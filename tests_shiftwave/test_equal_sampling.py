from collections import Counter
from shiftwave.data import Clips
from shiftwave.protocol import training_settings
from shift500.runtime import rank_indices, accumulation


def dataset():
    records=[]
    for domain, sizes in [('gopro',[15,18]),('dvd',[14,16]),('bsd',[17,20])]:
        for i,n in enumerate(sizes):
            records.append(dict(domain=domain,name=str(i),blur=list(range(n))))
    return Clips({'train':records},600,frames=13,seed=10,crop_size=256,n_frames_per_video=100)


def test_equal_every_update_and_each_rank():
    c=training_settings();assert c['clips_per_update']==6
    assert accumulation(6,2)==3
    ds=dataset()
    for step in range(30):
        domains=[ds.locate(i)[0] for i in range(step*6,(step+1)*6)]
        assert Counter(domains)=={'gopro':2,'dvd':2,'bsd':2}
        for rank in range(2):
            assert Counter(ds.locate(i)[0] for i in rank_indices(step,step+1,rank,2,6))=={'gopro':1,'dvd':1,'bsd':1}


def test_domain_windows_no_repeats_per_epoch_and_resume():
    ds=dataset()
    for offset,domain in enumerate(ds.cycle):
        count=ds.ends[domain][-1]
        windows=[ds.locate(3*i+offset)[1:] for i in range(count)]
        assert len({(r['name'],start) for r,start in windows})==count
    resumed=dataset()
    for i in range(66,120):
        assert ds.locate(i)==resumed.locate(i)


def test_sampling_migration_preserves_checkpoint(tmp_path, monkeypatch):
    import json
    import torch
    from shiftwave import prepare as module
    from shiftwave.model import MODEL_ID
    from shift500.data import sha256
    source=tmp_path/'old';source.mkdir()
    def record(domain,name,n):
        return dict(domain=domain,name=name,blur=list(range(n)),gt=list(range(n)),height=256,width=256)
    manifest={'train':[record('gopro',f'train{i}',15) for i in range(22)]+[record(d,'train',15) for d in ('dvd','bsd')],
              'val':[], 'test':[record('gopro',f'test{i}',101) for i in range(11)]+[record(d,'test',101) for d in ('dvd','bsd')],
              'split_audit':{'gt_file_sha256_cross_split_check':'passed'}}
    mp=source/'manifest.json';mp.write_text(json.dumps(manifest))
    teacher=source/'teacher.pth';teacher.write_bytes(b'test provenance')
    c=dict(output=str(source),manifest=str(mp),manifest_sha256=sha256(mp),teacher_checkpoint=str(teacher),teacher_sha256=sha256(teacher),upstream='fake',clips_per_update=8)
    cp=source/'config.json';cp.write_text(json.dumps(c))
    state=dict(config=c,model_id=MODEL_ID,variant='student',update=2000,
               model={'w':torch.tensor([2.])},optimizer={'state':{'moments':torch.tensor([3.])}},scaler={'scale':128},amp_skipped_total=2)
    ck=source/'latest.pth';torch.save(state,ck);before=sha256(ck)
    monkeypatch.setattr(module,'profile',lambda _:dict(arithmetic_GFLOPs_per_output=444.41))
    monkeypatch.setattr(module,'teacher_profile',lambda *args:dict(arithmetic_GFLOPs_per_output=3248.66))
    target=tmp_path/'equal';out=module.prepare(cp,target,continue_checkpoint=ck)
    moved=torch.load(target/'student/latest.pth',weights_only=False)
    assert out['clips_per_update']==6 and moved['config']==out and moved['update']==2000
    assert moved['scaler']==state['scaler'] and moved['amp_skipped_total']==2
    torch.testing.assert_close(moved['model']['w'],state['model']['w'])
    torch.testing.assert_close(moved['optimizer']['state']['moments'],state['optimizer']['state']['moments'])
    assert sha256(ck)==before and (target/'sampling_transition.json').exists()
