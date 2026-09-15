import os
from pathlib import Path
import pytest
import torch
from shiftwave.model import haar, inverse_haar, ShiftWave, initialize_pretrained
from shiftwave.losses import loss_terms
from shiftwave.profile import profile
from shiftwave.protocol import training_settings, distillation_weight
from shiftwave.report import acceptance
from shift500.model import ShiftModel

UPSTREAM=os.environ.get('SHIFT500_UPSTREAM',str(Path(__file__).resolve().parents[2]/'upstream_shift'))
torch.set_num_threads(2)


def test_haar_inverse_and_gradients():
    x=torch.randn(2,14,68,72,requires_grad=True)
    ll,hf=haar(x);out=inverse_haar(ll,hf)
    torch.testing.assert_close(out,x,rtol=1e-5,atol=1e-6)
    out.square().mean().backward()
    torch.testing.assert_close(x.grad,2*x.detach()/x.numel())


def test_retained_pretrained_load(tmp_path):
    teacher=ShiftModel(UPSTREAM,'teacher')
    path=tmp_path/'teacher.pth';torch.save({'params':teacher.net.state_dict()},path)
    student=ShiftWave(UPSTREAM)
    report=initialize_pretrained(student,path)
    assert report['pretrained_elements']>2000000
    for k,v in student.net.state_dict().items():
        torch.testing.assert_close(v,teacher.net.state_dict()[k],rtol=0,atol=0)


def test_all_parameters_used_checkpoint_parity_and_temporal_context():
    torch.manual_seed(10)
    a=ShiftWave(UPSTREAM);b=ShiftWave(UPSTREAM,True);b.load_state_dict(a.state_dict())
    x=torch.rand(1,7,3,64,72);gt=torch.rand(1,5,3,64,72)
    ya=a(x);yb=b(x);assert ya.shape==gt.shape
    torch.testing.assert_close(ya,yb)
    la,_=loss_terms(ya,gt,gt,.1);lb,_=loss_terms(yb,gt,gt,.1)
    la.backward();lb.backward()
    for (n,pa),pb in zip(a.named_parameters(),b.parameters()):
        assert pa.grad is not None,n
        assert torch.isfinite(pa.grad).all(),n
        torch.testing.assert_close(pa.grad,pb.grad,atol=2e-5,rtol=2e-4)
    a.eval()
    with torch.no_grad():
        z=a(torch.rand(1,16,3,65,73))
    assert z.shape==(1,12,3,65,73) and torch.isfinite(z).all()


def test_budget_and_joint_gate():
    p=profile(UPSTREAM)
    assert p['arithmetic_GFLOPs_per_output']<450
    assert p['conv_GFLOPs_per_output']==2*p['GMAC_per_output']
    c=training_settings()
    report={'summary':{'gopro':{'frames':1111,'sequences':11,'rgb8_psnr':32.99,'float_psnr':33.1}},
            'sequences':[{'domain':'gopro','frames':1111,'windows':100}]}
    assert acceptance(c,report,p)['status']=='TARGET_NOT_MET'
    report['summary']['gopro']['rgb8_psnr']=33.01
    assert acceptance(c,report,p)['status']=='TARGET_MET'
    report['sequences'][0]['windows']=200
    assert acceptance(c,report,p)['status']=='TARGET_NOT_MET'
    assert distillation_weight(0,c)==.1 and distillation_weight(100000,c)==0
