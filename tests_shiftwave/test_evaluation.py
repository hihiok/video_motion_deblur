import math
import torch
import numpy as np
from PIL import Image
from shiftwave.evaluate import evaluate
from shift500.data import discover


def test_full_frame_rgb8_identity_and_tail(tmp_path):
    for kind in ('blur','gt'):
        p=tmp_path/'test'/'scene'/kind;p.mkdir(parents=True)
        for i in range(13):
            Image.fromarray(np.full((65,73,3),i*10,dtype=np.uint8)).save(p/f'{i:04d}.png')
    class Identity(torch.nn.Module):
        def forward(self,x):return x[:,2:-2]
    r=evaluate(Identity(),discover(tmp_path,'gopro','test'),torch.device('cpu'))
    assert r['summary']['gopro']['frames']==13
    assert math.isinf(r['summary']['gopro']['rgb8_psnr'])
    assert r['sequences'][0]['windows']==2


def test_raw_nonfinite_rejected_before_clamp(tmp_path):
    import pytest
    for kind in ('blur','gt'):
        p=tmp_path/'test'/'scene'/kind;p.mkdir(parents=True)
        Image.fromarray(np.zeros((64,64,3),dtype=np.uint8)).save(p/'0000.png')
    class Broken(torch.nn.Module):
        def forward(self,x):return x[:,2:-2]+float('inf')
    with pytest.raises(RuntimeError,match='Nonfinite raw'):
        evaluate(Broken(),discover(tmp_path,'gopro','test'),torch.device('cpu'))
