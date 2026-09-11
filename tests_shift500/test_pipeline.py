import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from shift500.data import discover, Clips
from shift500.evaluate import windows, score_frame, evaluate
from shift500.losses import loss_terms
from shift500.model import ShiftModel, load_upstream, load_teacher, VARIANTS
from shift500.profile import profile

UPSTREAM=os.environ.get('SHIFT500_UPSTREAM',str(Path(__file__).resolve().parents[2]/'upstream_shift'))
torch.set_num_threads(2)


def test_teacher_parity(tmp_path):
    mod=load_upstream(UPSTREAM,VARIANTS['teacher'])
    original=mod.GShiftNet(future_frames=2,past_frames=2).eval()
    path=tmp_path/'teacher.pth';torch.save({'params':original.state_dict()},path)
    adapted=ShiftModel(UPSTREAM,'teacher').eval();load_teacher(adapted,path)
    x=torch.rand(1,7,3,36,40)
    with torch.no_grad():
        torch.testing.assert_close(adapted(x)[0],original(x),rtol=0,atol=0)


@pytest.mark.parametrize('variant',['quality','compact'])
def test_gradient_and_checkpoint_equivalence(variant):
    a=ShiftModel(UPSTREAM,variant);b=ShiftModel(UPSTREAM,variant,True);b.load_state_dict(a.state_dict())
    x=torch.rand(1,7,3,35,41);gt=torch.rand(1,3,3,35,41)
    ya=a(x);yb=b(x)
    torch.testing.assert_close(ya,yb)
    loss,_=loss_terms(ya,gt,gt,30000,180000);loss.backward()
    loss2,_=loss_terms(yb,gt,gt,30000,180000);loss2.backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is None:pytest.fail('Unused trainable parameter would break DDP')
        else:
            assert torch.isfinite(pa.grad).all();torch.testing.assert_close(pa.grad,pb.grad)


def test_budget():
    for v in ('quality','compact'):
        r=profile(UPSTREAM,v)
        assert r['output_frames']==12 and r['arithmetic_GFLOPs_per_output']<=500
        assert r['conv_GFLOPs_per_output']==2*r['GMAC_per_output']


def test_frame_coverage_and_official_protocol():
    for n in (1,2,5,12,13,100,452):
        ws=list(windows(n))
        assert [i for _,valid in ws for i in valid]==list(range(n))
        assert all(len(indices)==16 for indices,_ in ws)
        for indices,valid in ws:assert indices[2:2+len(valid)]==valid
    assert [i for _,v in windows(200,96,'official_chunks') for i in v]==list(range(2,194))


def test_pairing_and_metrics(tmp_path):
    for kind in ('blur','gt'):
        folder=tmp_path/'train'/'scene'/kind;folder.mkdir(parents=True)
        for i in range(16):Image.fromarray(np.full((36,40,3),i,dtype='uint8')).save(folder/f'{i:04d}.png')
    records=discover(tmp_path,'gopro','train')
    ds=Clips({'train':[dict(records[0],domain=d) for d in ('gopro','dvd','bsd')]},8)
    assert [ds[i]['domain'] for i in range(4)]==['gopro','dvd','gopro','bsd']
    assert ds[0]['blur'].shape==(16,3,36,40)
    assert score_frame(torch.ones(3,4,4),torch.ones(3,4,4))==float('inf')
    assert score_frame(torch.zeros(3,4,4),torch.ones(3,4,4))==0
    (tmp_path/'train'/'scene'/'gt'/'0001.png').unlink()
    with pytest.raises(ValueError,match='pairing'):discover(tmp_path,'gopro','train')


def test_evaluation_uses_matching_center_gt(tmp_path):
    folder=tmp_path/'test'/'scene'
    for kind in ('blur','gt'):
        (folder/kind).mkdir(parents=True)
        for i in range(13):Image.fromarray(np.full((36,40,3),i*10,dtype='uint8')).save(folder/kind/f'{i:04d}.png')
    class Identity(torch.nn.Module):
        def forward(self,x):return x[:,2:-2]
    r=evaluate(Identity(),discover(tmp_path,'gopro','test'),torch.device('cpu'))
    assert r['summary']['gopro']['frames']==13
    assert r['summary']['gopro']['psnr']==float('inf')
    assert r['summary']['gopro']['gt_relative_unwarped_temporal_l1']==0


def test_ddp_two_updates(tmp_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    try:
        dist.init_process_group('gloo',init_method=(tmp_path/'rendezvous').as_uri(),rank=0,world_size=1)
    except RuntimeError as e:
        if 'Operation not permitted' in str(e):
            pytest.skip('Runtime blocks Gloo sockets; execute this test on target server')
        raise
    try:
        model=DistributedDataParallel(ShiftModel(UPSTREAM,'quality',True))
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            model(torch.rand(1,6,3,36,40)).square().mean().backward()
    finally:
        dist.destroy_process_group()
