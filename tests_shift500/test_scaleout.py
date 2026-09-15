"""World-size changes preserve samples, optimizer math and checkpoint compatibility."""
import copy
import io
import pytest
import torch
from shift500.runtime import accumulation, rank_indices, choose_candidate


def test_every_update_same_eight_samples_at_resume():
    for start in (0,40300,41017):
        for world in (1,2,4,8):
            per_rank=[list(rank_indices(start,start+3,r,world)) for r in range(world)]
            accum=accumulation(8,world)
            for step in range(3):
                samples=[idx for indices in per_rank for idx in indices[step*accum:(step+1)*accum]]
                assert sorted(samples)==list(range((start+step)*8,(start+step+1)*8))
    with pytest.raises(ValueError):accumulation(8,3)


def test_optimizer_resume_after_changing_world_size():
    torch.manual_seed(10)
    initial=torch.nn.Linear(3,2).double()
    x=torch.randn(8,3,dtype=torch.float64);y=torch.randn(8,2,dtype=torch.float64)
    optimizer=torch.optim.AdamW(initial.parameters(),lr=4e-4,betas=(.9,.99),weight_decay=0)
    (initial(x)-y).abs().mean().backward();optimizer.step()
    data=io.BytesIO();torch.save({'model':initial.state_dict(),'optimizer':optimizer.state_dict()},data)
    checkpoint=data.getvalue();outcomes=[]
    for world in (2,4,8):
        state=torch.load(io.BytesIO(checkpoint),weights_only=False)
        model=copy.deepcopy(initial);model.load_state_dict(state['model'])
        opt=torch.optim.AdamW(model.parameters(),lr=4e-4,betas=(.9,.99),weight_decay=0)
        opt.load_state_dict(state['optimizer']);gradients=[]
        for rank in range(world):
            replica=copy.deepcopy(model);replica.zero_grad(set_to_none=True)
            for i in rank_indices(0,1,rank,world):
                ((replica(x[i:i+1])-y[i:i+1]).abs().mean()/accumulation(8,world)).backward()
            gradients.append([p.grad for p in replica.parameters()])
        for j,p in enumerate(model.parameters()):p.grad=torch.stack([g[j] for g in gradients]).mean(0)
        torch.nn.utils.clip_grad_norm_(model.parameters(),.01);opt.step()
        outcomes.append([p.detach().clone() for p in model.parameters()])
    for params in outcomes[1:]:
        for a,b in zip(outcomes[0],params):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def candidate(world, seconds, util=80, checkpointing='on'):
    return dict(status='PASS',world_size=world,seconds_per_update=seconds,gpu_util_mean=util,
                gpu_util_min_rank=util,amp_skipped_during_trial=0,peak_memory_fraction=.3,
                activation_checkpointing=checkpointing)


def test_selection_requires_incremental_eight_gpu_benefit():
    two=candidate(2,1.92);four=candidate(4,1.1);eight=candidate(8,.95)
    assert choose_candidate([two,four,eight]) is four
    eight=candidate(8,.65)
    assert choose_candidate([two,four,eight]) is eight
    eight['gpu_util_mean']=40
    assert choose_candidate([two,four,eight]) is four
    four['amp_skipped_during_trial']=1
    with pytest.raises(ValueError):choose_candidate([two,four,eight])


def test_disposable_benchmark_never_writes_production(tmp_path,monkeypatch):
    """Exercise the real trainer on CPU with a tiny model and CUDA-only calls adapted."""
    import argparse
    import json
    from contextlib import nullcontext
    from types import SimpleNamespace
    import shift500.train as trainer
    from shift500.data import sha256
    from shift500.protocol import training_settings

    class Tiny(torch.nn.Module):
        def __init__(self,*args,**kwargs):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.7))
        def forward(self,x):return x[:,1:-1].float()*self.weight

    class Samples(torch.utils.data.Dataset):
        def __init__(self,manifest,total,*args,**kwargs):self.total=total
        def __len__(self):return self.total
        def __getitem__(self,index):
            x=torch.full((13,3,2,2),.5+index%3*.1)
            return {'blur':x,'gt':x*.9}

    class TorchCPU:
        cuda=SimpleNamespace(set_device=lambda *_:None,manual_seed_all=lambda *_:None,
            synchronize=lambda:None,reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:16,
            get_device_properties=lambda _:SimpleNamespace(total_memory=1000),
            amp=SimpleNamespace(GradScaler=lambda:torch.cuda.amp.GradScaler(enabled=False)))
        @staticmethod
        def device(*args):return torch.device('cpu')
        @staticmethod
        def autocast(*args,**kwargs):return nullcontext()
        def __getattr__(self,key):return getattr(torch,key)

    monkeypatch.setattr(trainer,'torch',TorchCPU())
    monkeypatch.setattr(trainer,'ShiftModel',Tiny);monkeypatch.setattr(trainer,'Clips',Samples)
    monkeypatch.setenv('WORLD_SIZE','1');monkeypatch.setenv('RANK','0');monkeypatch.setenv('LOCAL_RANK','0')
    # Do not install process-global SIGTERM handlers during this unit test.
    monkeypatch.setattr(trainer.signal,'signal',lambda *_:None)
    run=tmp_path/'run';prod=run/'quality';prod.mkdir(parents=True)
    manifest=run/'manifest.json';manifest.write_text('{}')
    c=dict(training_settings(),output=str(run),upstream='unused',manifest=str(manifest),manifest_sha256=sha256(manifest))
    config=run/'config.json';config.write_text(json.dumps(c))
    (prod/'preflight.json').write_text(json.dumps(dict(status='PASS',config_sha256=sha256(config))))
    (prod/'training.jsonl').write_text('existing production log\n')
    model=Tiny();opt=torch.optim.AdamW(model.parameters(),lr=4e-4,betas=(.9,.99),weight_decay=0)
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    source=prod/'latest.pth'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                    update=40300,amp_skipped_total=0,variant='quality',config=c),source)
    before={p.name:p.read_bytes() for p in prod.iterdir()}
    report=tmp_path/'trial'/'report.json'
    args=argparse.Namespace(config=str(config),variant='quality',resume=str(source),preflight=False,
        stop_after=None,benchmark_report=str(report),benchmark_warmup=1,benchmark_steps=2,
        activation_checkpointing='off',workers=0,prefetch_factor=2)
    trainer.train(args)
    assert {p.name:p.read_bytes() for p in prod.iterdir()}==before
    result=json.loads(report.read_text())
    assert result['start_update']==40300 and result['end_update']==40303
    assert result['measured_updates']==2 and result['amp_skipped_during_trial']==0 and result['amp_skipped_measured']==0
    assert result['world_size']==1 and result['accumulation_per_rank']==8
