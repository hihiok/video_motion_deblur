"""Executable upstream objective, schedule and full official-train data contract."""
import ast
from pathlib import Path
import os
import torch
from shift500.losses import loss_terms
from shift500.protocol import training_settings, learning_rate, official_training_partition
from shift500.data import Clips

UPSTREAM=Path(os.environ.get('SHIFT500_UPSTREAM','../upstream_shift'))


def test_actual_upstream_l1_value_and_gradient():
    tree=ast.parse((UPSTREAM/'basicsr/loss/__init__.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Loss2')
    namespace={'torch':torch,'nn':torch.nn}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'upstream_Loss2','exec'),namespace)
    original=namespace['Loss2']('1*L1')
    a=torch.randn(1,11,3,12,16,requires_grad=True)
    b=a.detach().clone().requires_grad_();gt=torch.randn_like(a)
    actual,_=loss_terms(a,gt);expected=original(b[0],gt[0])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.backward();expected.backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_no_warmup_cosine_matches_pytorch_upstream_schedule():
    c=training_settings();c['total_updates']=10
    p=torch.nn.Parameter(torch.ones(()))
    opt=torch.optim.AdamW([p],lr=c['lr'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=c['min_lr'])
    for step in range(11):
        assert abs(learning_rate(step,c)-opt.param_groups[0]['lr'])<1e-15
        opt.step();schedule.step()
    assert learning_rate(0,c)==4e-4
    assert learning_rate(10,c)==1e-7


def test_partition_reunites_only_internal_holdout():
    train=dict(domain='gopro',name='a');holdout=dict(domain='gopro',name='b')
    test=dict(domain='gopro',name='c')
    source=dict(train=[train],val=[holdout],test=[test],split_audit={
        'gt_file_sha256_cross_split_check':'passed',
        'gopro':{'train_val_holdout_groups':['b'],'official_train_test_shared_acquisitions':['shared']}})
    result=official_training_partition(source)
    assert result['train']==[train,holdout] and result['val']==[] and result['test']==[test]
    assert source['val']==[holdout] and source['train']==[train]
    assert result['split_audit']['gopro']['official_train_test_shared_acquisitions']==['shared']


def test_shuffled_window_epochs_and_global_eight_domain_mix():
    records=[]
    for d in ('gopro','dvd','bsd'):
        records += [dict(domain=d,name='long',blur=list(range(105))),
                    dict(domain=d,name='short',blur=list(range(14)))]
    ds=Clips({'train':records},1000,13,10,256,n_frames_per_video=100)
    assert [ds.locate(i)[0] for i in range(8)]==['gopro','dvd','gopro','bsd']*2
    # Long video contributes starts 0..87; short contributes 0..1, exactly once/epoch.
    expected={('long',i) for i in range(88)}|{('short',0),('short',1)}
    for epoch in (0,1):
        positions=[ds.locate(2*i) for i in range(epoch*90,(epoch+1)*90)]
        assert {(r['name'],start) for _,r,start in positions}==expected
