"""Shared spatial/temporal crop alignment and upstream training-output contract."""
import ast
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
import pytest
import torch
from shift500.data import load_training_clip, Clips, sha256
from shift500.model import ShiftModel, load_upstream, load_teacher, VARIANTS
from shift500.protocol import training_settings, training_targets, validate_training_settings
from shift500.prepare_crop import migrate

UPSTREAM=os.environ.get('SHIFT500_UPSTREAM',str(Path(__file__).resolve().parents[2]/'upstream_shift'))
torch.set_num_threads(2)


@pytest.fixture
def record(tmp_path):
    h,w=272,288
    xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    result=dict(domain='gopro',name='scene',height=h,width=w,blur=[],gt=[])
    for t in range(13):
        array=np.stack((xx%256,yy%256,np.full_like(xx,t)),axis=-1).astype('uint8')
        for key in ('blur','gt'):
            path=tmp_path/f'{key}_{t:04d}.png'
            Image.fromarray(array).save(path);result[key].append(str(path))
    return result


def test_same_crop_all_frames_gt_and_preflight_training(record):
    sample=load_training_clip(record,0,13,256,random.Random(1),augment=False)
    left,top,right,bottom=sample['crop_box']
    assert sample['blur'].shape==(13,3,256,256)
    torch.testing.assert_close(sample['blur'],sample['gt'],rtol=0,atol=0)
    assert right-left==bottom-top==256
    assert round(sample['blur'][0,0,0,0].item()*255)==left
    assert round(sample['blur'][0,1,0,0].item()*255)==top
    assert sample['blur'][:,2,0,0].mul(255).round().tolist()==list(range(13))
    manifest={'train':[dict(record,domain=d) for d in ('gopro','dvd','bsd')]}
    ds=Clips(manifest,4,13,seed=1,crop_size=256)
    for i in range(4):
        a,b=ds[i],ds[i]
        assert a['blur'].shape==(13,3,256,256)
        torch.testing.assert_close(a['blur'],b['blur'],rtol=0,atol=0)
        torch.testing.assert_close(a['blur'],a['gt'],rtol=0,atol=0)
        assert a['blur'][:,2,0,0].mul(255).round().tolist()==list(range(13))


def test_official_train_13_to_11_and_eval_16_to_12(tmp_path):
    original=load_upstream(UPSTREAM,VARIANTS['teacher']).GShiftNet().eval()
    path=tmp_path/'teacher.pth';torch.save({'params':original.state_dict()},path)
    teacher=ShiftModel(UPSTREAM,'teacher',training_context=1).eval();load_teacher(teacher,path)
    x=torch.rand(1,13,3,36,40)
    with torch.no_grad():
        actual=teacher(x,context=1)
        assert actual.shape==(1,11,3,36,40)
        torch.testing.assert_close(actual[0],original(x),rtol=0,atol=0)
    for v in ('quality','compact'):
        student=ShiftModel(UPSTREAM,v,True,training_context=1)
        gt=torch.rand_like(x)
        pred=student(x)
        assert pred.shape==training_targets(gt,training_settings()).shape==(1,11,3,36,40)
        (pred-training_targets(gt,training_settings())).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
        with torch.no_grad():
            assert student.eval()(torch.rand(1,16,3,36,40)).shape==(1,12,3,36,40)


def test_old_config_cannot_satisfy_crop_gate():
    with pytest.raises(ValueError,match='prepare_official'):
        validate_training_settings({'frames':16,'training_spatial_mode':'native_full_frame_no_crop_no_resize'})
    validate_training_settings(training_settings())


def test_migration_preserves_audited_data_and_old_config(record,tmp_path):
    old=tmp_path/'old';old.mkdir()
    teacher=old/'teacher.pth';teacher.write_bytes(b'already audited checkpoint fixture')
    manifest={'version':2,'frames':16,'train':[record],'val':[],'test':[],
              'split_audit':{'gt_file_sha256_cross_split_check':'passed'}}
    mp=old/'manifest.json';mp.write_text(json.dumps(manifest))
    c=dict(output=str(old),manifest=str(mp),manifest_sha256=sha256(mp),
           teacher_checkpoint=str(teacher),teacher_sha256=sha256(teacher),upstream=UPSTREAM,
           frames=16,clips_per_update=4,total_updates=180000)
    cp=old/'config.json';cp.write_text(json.dumps(c));before=cp.read_bytes();mbefore=mp.read_bytes()
    new=migrate(cp,tmp_path/'crop')
    assert cp.read_bytes()==before and mp.read_bytes()==mbefore
    nm=json.loads(Path(new['manifest']).read_text())
    assert all(nm[s]==manifest[s] for s in ('train','val','test'))
    assert new['frames']==13 and new['crop_size']==256 and new['training_context']==1
    assert new['teacher_sha256']==c['teacher_sha256'] and new['total_updates']==300000
    with pytest.raises(FileExistsError):migrate(cp,tmp_path/'crop')


def test_python39_syntax():
    for path in (Path(__file__).resolve().parents[1]/'shift500').glob('*.py'):
        ast.parse(path.read_text(),feature_version=(3,9))
