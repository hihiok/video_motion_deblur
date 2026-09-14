import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import pytest
import torch
import yaml
from fair_benchmark import data, metrics, workflow
from rtf_temporal import posttrain
from rtf_temporal.model import TemporalRTFocuser


def make_split(root, domain='gopro', count=3, layout='sequence_modalities', offset=0, names=('scene0','scene1')):
    for sn,name in enumerate(names):
        for label in ('blur','sharp'):
            d=(root/'test'/name/label if layout=='sequence_modalities' else root/'test'/label/name)
            if domain=='bsd':d=d/'RGB'
            d.mkdir(parents=True)
            for i in range(count):
                rng=np.random.default_rng(offset+sn*100+i)
                a=rng.integers(0,200,(32,48,3),dtype=np.uint8)
                if label=='blur':a=np.clip(a.astype(int)+10,0,255).astype(np.uint8)
                Image.fromarray(a).save(d/f'{i+7:08d}.png')
    return {'root':str(root),'layout':layout,'lq_dir':'blur','gt_dir':'sharp',
            'variant':'3ms24ms' if domain=='bsd' else ('blur' if domain=='gopro' else 'standard'),
            'variant_evidence':'original BSD_3ms24ms archive'}


@pytest.mark.parametrize('layout',['sequence_modalities','split_modalities'])
def test_canonical_pairing_and_bsd_rgb(tmp_path,layout):
    spec=make_split(tmp_path/'source',domain='bsd',layout=layout)
    ds=data.prepare_dataset('bsd',spec,tmp_path/'canonical',expected=(2,6),check_official_names=False)
    assert ds['frames']==6
    f=ds['sequences'][0]['frames'][0]
    assert f['original_frame_id']=='00000007' and Path(f['blur']).name=='00000000.png'
    data.verify_files(ds)
    Image.fromarray(np.zeros((32,48,3),np.uint8)).save(f['source_blur'])
    with pytest.raises(ValueError,match='modified'):data.verify_files(ds)


def test_official_names_counts_and_no_train_fallback(tmp_path):
    spec=make_split(tmp_path/'source')
    with pytest.raises(ValueError,match='expected 11'):data.prepare_dataset('gopro',spec,tmp_path/'bad')
    shutil.move(tmp_path/'source/test',tmp_path/'source/train')
    with pytest.raises(ValueError,match='test'):data.pair_directories(spec)
    assert sum(v[0] for v in data.official_meta('gopro').values())==1111
    assert sum(v[0] for v in data.official_meta('dvd').values())==1000


def test_selection_overlap_and_reencoded_pixels(tmp_path):
    spec=make_split(tmp_path/'source')
    ds=data.prepare_dataset('gopro',spec,tmp_path/'canonical',expected=(2,6),check_official_names=False)
    f=ds['sequences'][0]['frames'][1]
    reencoded=tmp_path/'renamed.png'
    Image.fromarray(data.rgb(f['gt'])).save(reencoded,compress_level=0)
    manifest={'train':[],'val':[{'domain':'dvd','name':'renamed','gt':[str(reencoded)]}]}
    audit=data.overlap_audit(manifest,{'gopro':ds})
    assert audit['datasets']['gopro']['status']=='OVERLAP'
    assert audit['datasets']['gopro']['overlap_frames']==1


def test_metrics_inf_formula_and_missing_frames(tmp_path):
    a=np.zeros((16,16,3),np.uint8);b=a+10
    assert math.isinf(metrics.metric(a,a)[0])
    assert metrics.metric(b,a)[0]==pytest.approx(20*math.log10(25.5))
    spec=make_split(tmp_path/'source')
    ds=data.prepare_dataset('gopro',spec,tmp_path/'canonical',expected=(2,6),check_official_names=False)
    pred=tmp_path/'pred'
    for seq in ds['sequences']:
        (pred/seq['name']).mkdir(parents=True)
        for f in seq['frames']:shutil.copy2(f['blur'],pred/seq['name']/f'{f["index"]:08d}.png')
    summary=metrics.score(ds,pred,tmp_path/'scores')
    assert summary['frames']==6
    assert summary['psnr_frame_mean']==pytest.approx(20*math.log10(25.5))
    (pred/'scene0/00000000.png').unlink()
    with pytest.raises(ValueError,match='frame set mismatch'):metrics.score(ds,pred,tmp_path/'bad_scores')


def test_reference_mapping_no_positional_guess(tmp_path):
    spec=make_split(tmp_path/'source',count=16)
    ds=data.prepare_dataset('gopro',spec,tmp_path/'canonical',expected=(2,32),check_official_names=False)
    pred=tmp_path/'pred';mapping=[]
    for seq in ds['sequences']:
        (pred/seq['name']).mkdir(parents=True)
        for f in seq['frames']:
            shutil.copy2(f['blur'],pred/seq['name']/f'{f["index"]:08d}.png')
            mapping.append({'sequence':seq['name'],'original_frame_id':f['original_frame_id'],'official_prediction':f['blur']})
    assert metrics.reference_check(ds,pred,mapping)['passes']
    mapping[0]['original_frame_id']='wrong'
    with pytest.raises(ValueError,match='reference frame'):metrics.reference_check(ds,pred,mapping)


def test_producer_writes_native_rgb8_and_fails_nan(tmp_path):
    spec=importlib.util.spec_from_file_location('fair_producer',Path(__file__).parents[1]/'tools/infer_fair_video.py')
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    torch.set_num_threads(2)
    model=posttrain.RT_Focuser_Standard().eval();cp=tmp_path/'model.pth';torch.save(model.state_dict(),cp)
    frames=[]
    for i in range(2):
        p=tmp_path/f'{i:08d}.png';Image.fromarray(np.full((33,49,3),i*30,np.uint8)).save(p);frames.append(p)
    job={'kind':'rtf_official','checkpoint_snapshot':str(cp),'checkpoint_sha256':data.sha256(cp)}
    out=tmp_path/'pred';out.mkdir()
    with torch.inference_mode():info=producer.run(job,frames,out,'cpu')
    assert info['frames']==2 and Image.open(out/'00000000.png').size==(49,33)
    next(model.parameters()).data.fill_(float('nan'));torch.save(model.state_dict(),cp)
    job['checkpoint_sha256']=data.sha256(cp);bad=tmp_path/'bad';bad.mkdir()
    with pytest.raises(FloatingPointError,match='Nonfinite'):
        with torch.inference_mode():producer.run(job,frames,bad,'cpu')


def test_workflow_freezes_and_excludes_leaked_joint_only(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'git_identity',lambda _: {'commit':'test','working_changes_sha256':'test','tracked_dirty':False})
    raw_prepare=data.prepare_dataset
    monkeypatch.setattr(workflow,'prepare_dataset',lambda d,s,o:raw_prepare(d,s,o,expected=(2,4),check_official_names=False))
    specs={d:make_split(tmp_path/d,domain=d,count=2,offset=i*1000) for i,d in enumerate(data.EXPECTED)}
    run=tmp_path/'training';(run/'checkpoints').mkdir(parents=True)
    gt=tmp_path/'dvd/test/scene0/sharp/00000007.png'
    training={'train':[{'domain':'dvd','name':'scene0','gt':[str(gt)],'blur':[str(gt)]}],'val':[]}
    posttrain.write_json(run/'manifest.json',training)
    official=posttrain.RT_Focuser_Standard().eval();cp=tmp_path/'official.pth';torch.save(official.state_dict(),cp)
    monkeypatch.setattr(posttrain,'OFFICIAL_SHA',data.sha256(cp))
    temporal=TemporalRTFocuser(activation_checkpointing=False);temporal.load_official(cp)
    payload={'model':temporal.state_dict(),'update':17000,'git_commit':'train','baseline':{},'config':{
        'protocol':'rtfocuser_causal_temporal_finetune_v1','manifest_sha256':data.sha256(run/'manifest.json'),
        'model':{'hidden':32},'train':{'cut_threshold':.3}}}
    best=run/'checkpoints/best_stable.pth';torch.save(payload,best)
    jobs=[]
    for domain in data.EXPECTED:
        for kind,path in [('rtf_official',cp),('rtf_temporal',best)]:
            jobs.append({'id':kind+'_'+domain,'kind':kind,'domain':domain,'checkpoint':str(path),
                         'training_domains':['gopro'],'training_evidence':'test','causality':'causal','python':sys.executable})
    cfg={'protocol':'fair_video_deblur_rgb8_v1','training_run':str(run),'official_checkpoint':str(cp),'datasets':specs,'jobs':jobs}
    cf=tmp_path/'config.yml';cf.write_text(yaml.safe_dump(cfg));out=tmp_path/'benchmark'
    frozen=workflow.prepare(cf,out)
    status={j['id']:j['status'] for j in frozen['jobs']}
    assert status['rtf_temporal_dvd']=='UNAVAILABLE'
    assert status['rtf_official_dvd']=='READY'
    assert status['rtf_temporal_gopro']=='READY'
    statuses=workflow.run_jobs(out,phase='full',only=['rtf_official_gopro'],device='cpu')
    assert statuses[0]['status']=='INFERENCE_COMPLETE'
    report=workflow.summarize(out)
    row=next(r for r in report['rows'] if r['id']=='rtf_official_gopro')
    assert row['status']=='MEASURED' and row['frames']==4
    with (out/'frozen.json').open('a') as h:h.write(' ')
    with pytest.raises(ValueError,match='Frozen protocol changed'):workflow.load_frozen(out)


def test_checkpoint_container_ambiguity_requires_explicit_selection():
    spec=importlib.util.spec_from_file_location('fair_producer_state',Path(__file__).parents[1]/'tools/infer_fair_video.py')
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    payload={'params':{'weight':torch.tensor([1.])},'params_ema':{'weight':torch.tensor([2.])}}
    with pytest.raises(ValueError,match='Ambiguous'):producer.select_state(payload)
    state,key=producer.select_state(payload,'params')
    assert key=='params' and state['weight'].item()==1
