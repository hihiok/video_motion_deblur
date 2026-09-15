"""Two-GPU Haar-domain Shift-Net compression with pretrained initialization and output distillation."""
import argparse
from datetime import timedelta
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from shift500.data import DOMAINS, load_training_clip, sha256
from .data import Clips
from .protocol import validate_training_settings, training_targets, learning_rate, distillation_weight
from .losses import loss_terms
from .model import ShiftWave, initialize_pretrained
from shift500.model import ShiftModel, load_teacher
from shift500.runtime import accumulation, rank_indices

STOP=False

def request_stop(*_):
    global STOP
    STOP=True


def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(path)


def worker_init(_):
    torch.set_num_threads(1)


def train(a):
    c=json.loads(Path(a.config).read_text())
    validate_training_settings(c)
    if sha256(c['manifest'])!=c['manifest_sha256']:
        raise ValueError('Input provenance changed')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local=int(os.environ.get('LOCAL_RANK',0))
    accum=accumulation(c['clips_per_update'],world)
    if (not a.preflight and world!=2) or (a.preflight and world!=1):
        raise ValueError('Training requires exactly two GPUs; disposable memory preflight uses one')
    if sha256(c['teacher_checkpoint'])!=c['teacher_sha256']:
        raise ValueError('Teacher checkpoint changed')
    benchmarking=bool(a.benchmark_report)
    if benchmarking and (a.preflight or a.stop_after or not a.resume):
        raise ValueError('Benchmark requires explicit --resume and cannot combine with preflight/stop-after')
    if a.benchmark_warmup<0 or a.benchmark_steps<1:raise ValueError('Invalid benchmark lengths')
    if benchmarking and Path(a.benchmark_report).exists():raise FileExistsError(a.benchmark_report)
    workers=c['workers'] if a.workers is None else a.workers
    if workers<0 or a.prefetch_factor<1:raise ValueError('Invalid loader execution controls')
    execution=dict(world_size=world,activation_checkpointing=a.activation_checkpointing,
                   workers_per_rank=workers,prefetch_factor=a.prefetch_factor,global_batch=c['clips_per_update'],
                   accumulation_per_rank=accum)
    torch.set_num_threads(2);torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    manifest=json.loads(Path(c['manifest']).read_text())
    out=Path(c['output'])/a.variant
    if not benchmarking:out.mkdir(parents=True,exist_ok=True)
    model=ShiftWave(c['upstream'],activation_checkpointing=a.activation_checkpointing=='on',training_context=c['training_context']).to(device)
    transfer=initialize_pretrained(model,c['teacher_checkpoint'])
    teacher=ShiftModel(c['upstream'],'teacher').to(device).eval()
    load_teacher(teacher,c['teacher_checkpoint']);teacher.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],betas=(.9,.99),weight_decay=0)
    scaler=torch.cuda.amp.GradScaler()
    start=0; amp_skipped_total=0
    resume=Path(a.resume) if a.resume else out/'latest.pth'
    if resume.exists() and not a.preflight:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['variant']!=a.variant or state['config']!=c:raise ValueError('Resume provenance/config mismatch')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start=state['update'];amp_skipped_total=state['amp_skipped_total']
        if state.get('model_id')!=c['model_id']:raise ValueError('Wrong model identity')
    elif a.resume and not a.preflight:
        raise FileNotFoundError(a.resume)
    if rank==0 and not benchmarking and not start:(out/'initialization.json').write_text(json.dumps(transfer,indent=2)+'\n')
    initial_amp_skipped=amp_skipped_total
    total=c['total_updates']
    if a.preflight:
        if world!=1:raise ValueError('Preflight is single GPU; later DDP uses same per-GPU microbatch')
        reports=[]
        for d in DOMAINS:
            r=max((r for r in manifest['train'] if r['domain']==d),key=lambda r:r['height']*r['width'])
            print(json.dumps({'preflight_phase':'crop_backward','domain':d,'sequence':r['name']}),flush=True)
            sample=load_training_clip(r,0,c['frames'],c['crop_size'],random.Random(c['seed']),augment=False)
            x=sample['blur'][None].to(device).half();y=training_targets(sample['gt'][None],c).to(device)
            torch.cuda.reset_peak_memory_stats();opt.zero_grad(set_to_none=True);t=time.monotonic()
            # An initial FP16 scale overflow is recoverable, as in upstream.
            # Require an actual finite optimizer step before declaring PASS.
            for attempt in range(10):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    with torch.no_grad():
                        target=teacher(x,context=c['training_context'])
                    if not torch.isfinite(target).all():raise RuntimeError('NONFINITE_TEACHER')
                    pred=model(x);loss,terms=loss_terms(pred,y,target,c['kd_weight'],c['hf_weight'])
                if not torch.isfinite(loss):raise RuntimeError('NONFINITE_PREFLIGHT_LOSS')
                scaler.scale(loss).backward();scaler.unscale_(opt)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'])
                finite=bool(torch.isfinite(norm))
                scaler.step(opt);scaler.update()
                if finite:break
                del pred,loss,terms
            else:raise RuntimeError('NONFINITE_PREFLIGHT_GRAD_AFTER_SCALE_BACKOFF')
            torch.cuda.synchronize()
            reports.append(dict(domain=d,sequence=r['name'],source_resolution=[r['height'],r['width']],crop_box=sample['crop_box'],
                                shape=list(x.shape),output_shape=list(pred.shape),
                                amp_scale=scaler.get_scale(),scale_backoffs=attempt,
                                seconds=time.monotonic()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
            print(json.dumps(reports[-1]),flush=True)
            (out/'preflight_progress.json').write_text(json.dumps({'training_recipe':c['training_recipe'],'crop_reports':reports},indent=2)+'\n')
            del x,y,pred,loss,terms,target
        # Separately verify deployment forward memory. A training crop is not
        # evidence that native 1080p inference fits. No full-frame backward here.
        inference_reports=[]
        r=max(manifest['train']+manifest['val']+manifest['test'],key=lambda r:r['height']*r['width'])
        opt.zero_grad(set_to_none=True)
        teacher.cpu();torch.cuda.empty_cache()
        native_h,native_w=max(((1080,1920),(r['height'],r['width'])),key=lambda hw:hw[0]*hw[1])
        for label,network in ((a.variant,model),):
            print(json.dumps({'preflight_phase':'native_inference','model':label,'height':native_h,'width':native_w}),flush=True)
            was_training=network.training;network.eval()
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                x=torch.zeros(1,16,3,native_h,native_w,device=device)
                pred=network(x)
                if pred.shape != (1,12,3,native_h,native_w) or not torch.isfinite(pred).all():
                    raise RuntimeError('NATIVE_INFERENCE_PREFLIGHT_FAILED')
                inference_reports.append(dict(model=label,shape=list(x.shape),output_shape=list(pred.shape),
                                               peak_GiB=torch.cuda.max_memory_allocated()/2**30))
                del x,pred
            network.train(was_training)
        (out/'preflight.json').write_text(json.dumps({'status':'PASS','config_sha256':sha256(a.config),
            'training_recipe':c['training_recipe'],'reports':reports,'native_inference':inference_reports},indent=2)+'\n')
        print(json.dumps(reports,indent=2));return
    gate=out/'preflight.json'
    if not gate.exists() or json.loads(gate.read_text()).get('status')!='PASS' or json.loads(gate.read_text()).get('config_sha256')!=sha256(a.config):
        raise ValueError('Run this variant preflight first')
    if world>1:model=DDP(model,device_ids=[local],broadcast_buffers=False)
    raw=model.module if world>1 else model
    end=total if not a.stop_after else min(total,start+a.stop_after)
    if benchmarking:
        end=min(total,start+a.benchmark_warmup+a.benchmark_steps)
        if end-start<=a.benchmark_warmup:raise ValueError('Insufficient remaining iterations for benchmark')
    indices=rank_indices(start,end,rank,world,c['clips_per_update'])
    dataset=Clips(manifest,total*c['clips_per_update'],c['frames'],c['seed'],crop_size=c['crop_size'],
                  n_frames_per_video=c['n_frames_per_video'])
    loader=iter(DataLoader(dataset,batch_size=1,sampler=indices,num_workers=workers,pin_memory=True,
                           worker_init_fn=worker_init,**({'prefetch_factor':a.prefetch_factor} if workers else {})))
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    began=time.monotonic();last_log_time=began;last_log_update=start
    measurement_start=None;measurement_wall=None;data_wait=0.;measurement_initial_skips=initial_amp_skipped
    if rank==0 and not benchmarking:
        (out/'runtime.json').write_text(json.dumps(dict(execution,start_update=start,
            resumed_from=str(resume) if start else None),indent=2)+'\n')
        print(json.dumps({'execution':execution,'resume_update':start,'amp_skipped_total':amp_skipped_total}),flush=True)
    def save(update,filename='latest.pth'):
        if rank==0 and not benchmarking:
            atomic_save({'model':raw.state_dict(),'optimizer':opt.state_dict(),'update':update,
                         'model_id':c['model_id'],'scaler':scaler.state_dict(),'amp_skipped_total':amp_skipped_total,'variant':a.variant,'config':c,
                         'execution':execution,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},out/filename)
    for step in range(start,end):
        if step>=c['kd_end_update'] and teacher is not None:
            teacher=None;torch.cuda.empty_cache()
        if benchmarking and step==start+a.benchmark_warmup:
            if world>1:dist.barrier()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            measurement_start=time.monotonic();measurement_wall=time.time();measurement_initial_skips=amp_skipped_total
        lr=learning_rate(step,c)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);meter={}
        for micro in range(accum):
            load_started=time.monotonic();batch=next(loader)
            if measurement_start is not None:data_wait+=time.monotonic()-load_started
            x=batch['blur'].to(device,non_blocking=True).half();gt=training_targets(batch['gt'],c).to(device,non_blocking=True)
            sync=model.no_sync() if world>1 and micro<accum-1 else nullcontext()
            with sync:
                with torch.autocast('cuda',dtype=torch.float16):
                    kd_weight=distillation_weight(step,c)
                    with torch.no_grad():
                        target=teacher(x,context=c['training_context']) if kd_weight else None
                    if target is not None and not torch.isfinite(target).all():raise RuntimeError('NONFINITE_TEACHER')
                    pred=model(x);loss,terms=loss_terms(pred,gt,target,kd_weight,c['hf_weight'])
                finite=torch.tensor(int(torch.isfinite(loss)),device=device)
                if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():raise RuntimeError('NONFINITE_LOSS; resume last valid checkpoint')
                scaler.scale(loss/accum).backward()
            for k,v in terms.items():meter[k]=meter.get(k,0.)+v/accum
            del x,gt,pred,loss,terms,batch,target
        scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'])
        # Upstream AMP skips overflowing updates and adjusts the scale.
        scale_before=scaler.get_scale()
        scaler.step(opt);scaler.update();update=step+1
        amp_skipped=scaler.get_scale()<scale_before
        amp_skipped_total+=int(amp_skipped)
        if update%20==0 or update==start+1:
            if world>1:
                values=torch.stack(list(meter.values()))
                dist.all_reduce(values);values/=world
                meter=dict(zip(meter,values.tolist()))
            else:meter={k:v.item() for k,v in meter.items()}
        if rank==0 and (update%20==0 or update==start+1):
            row=dict(update=update,total=total,lr=lr,kd_weight=distillation_weight(step,c),grad_norm=norm.item() if torch.isfinite(norm) else None,
                     amp_scale=scaler.get_scale(),amp_skipped=amp_skipped,amp_skipped_total=amp_skipped_total,
                     seconds_per_update=(time.monotonic()-began)/(update-start),
                     recent_seconds_per_update=(time.monotonic()-last_log_time)/(update-last_log_update),**meter)
            last_log_time=time.monotonic();last_log_update=update
            print(json.dumps(row),flush=True)
            if not benchmarking:
                with open(out/'training.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        stop=torch.tensor(int(STOP),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if update%c['save_every']==0 or update==end or stop.item():save(update)
        if update%20000==0:save(update,f'update_{update:06d}.pth')
        if stop.item():
            save(update)
            if world>1:dist.barrier();dist.destroy_process_group()
            raise SystemExit(75)
    if benchmarking:
        torch.cuda.synchronize()
        duration=time.monotonic()-measurement_start
        memory=torch.cuda.max_memory_allocated()
        fraction=memory/torch.cuda.get_device_properties(device).total_memory
        metrics=torch.tensor([duration,data_wait,memory/2**30,fraction],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(metrics,op=dist.ReduceOp.MAX)
        if rank==0:
            elapsed,wait,peak,fraction=metrics.tolist()
            report=dict(execution,status='PASS',variant=a.variant,start_update=start,end_update=end,
                        measured_updates=end-start-a.benchmark_warmup,seconds_per_update=elapsed/(end-start-a.benchmark_warmup),
                        measurement_start_time=measurement_wall,measurement_end_time=time.time(),
                        max_rank_data_wait_fraction=wait/elapsed,peak_GiB=peak,peak_memory_fraction=fraction,
                        amp_skipped_during_trial=amp_skipped_total-initial_amp_skipped,
                        amp_skipped_measured=amp_skipped_total-measurement_initial_skips,
                        source_checkpoint=str(resume),config_sha256=sha256(a.config))
            target=Path(a.benchmark_report);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(report,indent=2)+'\n')
    if world>1:dist.barrier();dist.destroy_process_group()
    if STOP:raise SystemExit(75)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--variant',choices=['student'],default='student')
    p.add_argument('--preflight',action='store_true');p.add_argument('--resume')
    p.add_argument('--stop-after',type=int,help='Optional bounded run; does not alter schedule')
    p.add_argument('--activation-checkpointing',choices=['on','off'],default='on')
    p.add_argument('--workers',type=int,help='Execution-only workers per rank; config stays unchanged')
    p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--benchmark-report',help='Disposable trial: write only this report, never production logs/checkpoints')
    p.add_argument('--benchmark-warmup',type=int,default=20)
    p.add_argument('--benchmark-steps',type=int,default=80)
    train(p.parse_args())

if __name__=='__main__':main()
