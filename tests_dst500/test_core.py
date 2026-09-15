"""CPU semantic tests; pass DST500_UPSTREAM pointing at the pinned source."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
import torch
from dst500.model import ARCHITECTURE, Student, initialize, teacher, load_student
from dst500.run import losses
from dst500.evaluate import rgb8, psnr
from dst500.profile import profile, CounterMode
from dst500.data import load_training_clip
from PIL import Image
import numpy as np


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(17)
        cls.upstream=os.environ['DST500_UPSTREAM']
        cls.temp=tempfile.TemporaryDirectory();cls.ckpt=Path(cls.temp.name)/'teacher.pth'
        cls.teacher=teacher(cls.upstream)
        torch.save({'params':cls.teacher.state_dict()},cls.ckpt)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_train_fusion_and_checkpoint(self):
        model=Student(self.upstream);report=initialize(model,self.ckpt)
        self.assertEqual(len(report['svd_factorized']),52)
        x=torch.rand(1,3,3,32,40);gt=torch.rand_like(x)
        loss,_=losses(model(x),gt,self.teacher(x));loss.backward()
        for name,p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        model.eval();fused=copy.deepcopy(model).deploy()
        with torch.no_grad():torch.testing.assert_close(model(x),fused(x),atol=2e-5,rtol=2e-4)
        for deployment,m in [(False,model),(True,fused)]:
            path=Path(self.temp.name)/f'student_{deployment}.pth'
            torch.save({'architecture':ARCHITECTURE,'deployed':deployment,'model':m.state_dict()},path)
            restored,_=load_student(self.upstream,path)
            with torch.no_grad():torch.testing.assert_close(restored(x),fused(x),atol=2e-5,rtol=2e-4)
        for shape in [(1,1,3,33,41),(1,1,3,16,16)]:
            self.assertEqual(tuple(model(torch.rand(shape)).shape),shape)

    def test_wavelet_roundtrip(self):
        wave=self.teacher.wave;x=torch.randn(2,64,32,40)
        lo,hi=wave(x);back=wave(torch.cat([lo,hi],1),rev=True)
        torch.testing.assert_close(back,x,atol=1e-6,rtol=1e-6)

    def test_rgb8_metric(self):
        a=torch.zeros(3,2,2);b=torch.ones_like(a)/255
        self.assertEqual(psnr(a,a),float('inf'))
        self.assertAlmostEqual(psnr(a,b),48.1308036087,places=5)
        self.assertEqual(float(rgb8(torch.tensor(.5/255))),float(torch.tensor(1/255)))
        self.assertEqual(float(rgb8(torch.tensor(-.5))),0.)

    def test_counter_known_and_1080p(self):
        with torch.device('meta'):
            conv=torch.nn.Conv2d(3,4,3,padding=1);x=torch.ones(1,3,8,8)
        counter=CounterMode()
        with counter:conv(x)
        self.assertFalse(counter.unknown)
        self.assertEqual(sum(counter.flops.values()),2*4*8*8*3*3*3+4*8*8)
        r=profile(self.upstream)
        self.assertLessEqual(r['gflops_per_output'],500)
        self.assertGreater(r['gflops_per_output'],480)
        self.assertAlmostEqual(r['clip_gflops']/6,r['gflops_per_output'])

    def test_native_paired_augmentation(self):
        import random
        folder=Path(self.temp.name)/'frames';folder.mkdir(exist_ok=True)
        paths=[]
        for i in range(6):
            path=folder/f'{i:03}.png';Image.fromarray(np.full((24,40,3),i,dtype=np.uint8)).save(path);paths.append(str(path))
        r={'blur':paths,'gt':paths,'domain':'gopro','name':'test','height':24,'width':40}
        batch=load_training_clip(r,0,6,0,random.Random(1))
        self.assertEqual(tuple(batch['blur'].shape),(6,3,24,40))
        self.assertTrue(torch.equal(batch['blur'],batch['gt']))
        self.assertEqual(batch['crop_box'],(0,0,40,24))

if __name__=='__main__':unittest.main()
