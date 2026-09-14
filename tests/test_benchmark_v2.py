import argparse
import copy
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from benchmark_v2 import data, bsd_gt, evaluate

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'nanovsr_deblur/models/network_nanovnr_nafnet_rgb.py'


def image(path, seed=1, array=None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = np.random.default_rng(seed).integers(10, 220, (17, 25, 3), dtype=np.uint8)
    Image.fromarray(array).save(p)
    return array


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def config(self):
        cfg = {'datasets': {}}
        for j, name in enumerate(('GoPro', 'DVD', 'BSD')):
            for k, seq in enumerate(('video1', 'video2')):
                for i in range(2+k):
                    a = image(self.root/name/'test/blur'/seq/f'{i:08d}.png', 100*j+10*k+i)
                    image(self.root/name/'test/gt'/seq/f'{i:08d}.png', array=a+1)
            cfg['datasets'][name] = {'test_root': str(self.root/name/'test')}
            if name == 'BSD':
                cfg['datasets'][name]['exposure'] = '3ms24ms'
        path = self.root/'config.json'; data.write_json(path, cfg)
        return path
    def manifest(self):
        cfg = self.config(); path = self.root/'manifest.json'; data.build_manifest(cfg, path)
        return path
    def test_metric(self):
        a = np.zeros((2, 2, 3)); b = np.full_like(a, 0.1)
        self.assertAlmostEqual(evaluate.psnr(a, b), 20.0)
        self.assertTrue(math.isinf(evaluate.psnr(a, a)))
        with self.assertRaises(ValueError): evaluate.quantize(a + np.nan)
        self.assertEqual(evaluate.quantize(np.full_like(a, 0.5))[0, 0, 0], 128)
    def test_aggregations(self):
        rows = []
        for seq, count, value in [('a', 3, 10), ('b', 1, 30)]:
            rows.append({'dataset': 'D', 'sequence': seq, 'frames': [
                {'input_psnr': value, 'output_psnr_rgb8': value, 'output_psnr_float': value} for _ in range(count)]})
        out = evaluate.summarize(rows, self.root/'s.json', {}, True)
        self.assertEqual(out['datasets']['D']['output_psnr_rgb8_frame_mean'], 15)
        self.assertEqual(out['datasets']['D']['output_psnr_rgb8_video_mean'], 20)
    def test_full_manifest_and_mutation(self):
        p = self.manifest(); m = data.load_manifest(p)
        self.assertEqual(len(m['sequences']), 6)
        self.assertEqual(m['datasets']['GoPro']['frames'], 5)
        r = m['sequences'][0]['frames'][0]
        image(r['lq'], seed=999)
        with self.assertRaises(ValueError): data.checked_frame(r, 'lq')
        m['scope'] = 'modified'; data.write_json(p, m)
        with self.assertRaises(ValueError): data.load_manifest(p)
    def test_missing_gt_not_skipped(self):
        cfg = self.config()
        (self.root/'BSD/test/gt/video2/00000000.png').unlink()
        with self.assertRaises(ValueError): data.build_manifest(cfg, self.root/'m.json')
    def test_split_no_fallback(self):
        root = self.root/'BSD'; image(root/'config/test/Blur/RGB/000.png')
        with self.assertRaises(FileNotFoundError): data.discover(root/'test')
    def test_native_rgb_no_auto_recolor(self):
        p = self.root/'gray.png'; Image.new('L', (4, 4)).save(p)
        with self.assertRaises(ValueError): data.rgb(p)
    def test_bsd_copy_exact_offset(self):
        root = self.root/'BSD'; raw = root/'BSD_3ms24ms/test/012'
        for i in range(4):
            a = image(raw/'Blur/RGB'/f'{i:08d}.png', i+20)
            image(raw/'Sharp/RGB'/f'{i:08d}.png', array=a+2)
            if i in (1, 2): image(root/'test/blur/Scene1'/f'{i-1:08d}.png', array=a)
        mp4 = root/'test/gt_mp4'; mp4.mkdir(parents=True); (mp4/'Scene1.mp4').write_bytes(b'PROVENANCE_ONLY')
        inv = bsd_gt.inventory(root, [root], mp4)
        data.write_json(self.root/'inv.json', inv)
        plan = bsd_gt.plan(self.root/'inv.json', self.root/'plan.json')
        self.assertEqual(plan['sequences'][0]['raw_start_index'], 1)
        self.assertFalse(plan['sequences'][0]['is_complete_original_video'])
        bsd_gt.apply(self.root/'plan.json'); bsd_gt.apply(self.root/'plan.json')
        self.assertEqual(data.sha(root/'test/gt/Scene1/00000000.png'), data.sha(raw/'Sharp/RGB/00000001.png'))
        image(root/'test/gt/Scene1/00000000.png', seed=600)
        with self.assertRaises(ValueError): bsd_gt.apply(self.root/'plan.json')
    def test_no_similarity_based_alignment(self):
        a = image(self.root/'local/0.png', 1)
        image(self.root/'raw/blur/0.png', array=a+1)
        image(self.root/'raw/sharp/0.png', array=a+3)
        self.assertIsNone(bsd_gt.align_exact([self.root/'local/0.png'],
            {'blur_dir': str(self.root/'raw/blur'), 'sharp_dir': str(self.root/'raw/sharp')}))
    def test_overlap_not_acquisition_prefix(self):
        cfg_path = self.config(); cfg = data.read_json(cfg_path)
        # Different chunk IDs are not collapsed to a shared acquisition prefix.
        image(self.root/'GoPro/train/blur/video3/00000000.png', 990)
        image(self.root/'GoPro/train/gt/video3/00000000.png', 991)
        cfg['datasets']['GoPro']['train_root'] = str(self.root/'GoPro/train')
        data.write_json(cfg_path, cfg)
        data.build_manifest(cfg_path, self.root/'clean.json')
        gt = self.root/'GoPro/test/gt/video1/00000000.png'
        image(self.root/'GoPro/train/gt/video3/00000000.png', array=data.rgb(gt))
        with self.assertRaises(ValueError): data.build_manifest(cfg_path, self.root/'leak.json')
    def test_prediction_indexing_shift_all_and_fixed16(self):
        m = data.load_manifest(self.manifest()); frames = m['sequences'][0]['frames']
        class IdentityShift(torch.nn.Module):
            def forward(self, x): return x[0, 2:-2]
        for context in (0, 16):
            args = argparse.Namespace(method='shift-small', one_len=16, chunk=15, precision='fp32', context=context)
            got = list(evaluate.predict_sequence(args, frames, IdentityShift(), torch.device('cpu')))
            self.assertEqual([i for i,_ in got], list(range(len(frames))))
            for i,p in got: np.testing.assert_array_equal(evaluate.quantize(p), data.rgb(frames[i]['lq']))
    def test_nano_end_to_end_resume_external(self):
        torch.set_num_threads(1)
        self.assertEqual(data.blob(MODEL), evaluate.MODEL_BLOB)
        model = evaluate.import_file(MODEL, 'test_nano').NanoVNRNAFNetRGB()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 414923)
        checkpoint = self.root/'test.pth'
        torch.save({'architecture': 'NanoVNRNAFNetRGB', 'step': 125000, 'model': model.state_dict()}, checkpoint)
        mp = self.manifest(); meta = self.root/'meta.json'
        data.write_json(meta, {'training_data': ['SYNTHETIC_TEST'], 'checkpoint_origin': 'random_test_only',
                              'checkpoint_selection': 'TEST_ONLY'})
        args = argparse.Namespace(manifest=str(mp), metadata=str(meta), method='nano', checkpoint=str(checkpoint),
            out=str(self.root/'run'), model_file=str(MODEL), precision='fp32', device='cpu',
            chunk=2, one_len=16, context=0, resume=False)
        evaluate.run(args); args.resume=True; evaluate.run(args)
        report = data.read_json(self.root/'run/summary.json'); self.assertEqual(report['status'], 'COMPLETE')
        m = data.load_manifest(mp); idx = {'method': 'EXTERNAL_TEST', 'manifest_sha256': m['manifest_sha256'],
            'training_data': ['TEST'], 'checkpoint_sha256': data.sha(checkpoint), 'protocol': 'TEST',
            'precision': 'fp32', 'source_code_commit': 'TEST', 'outputs': []}
        for s in m['sequences']:
            for r in s['frames']:
                p = self.root/'run/sequences'/evaluate.seq_key(s)/'pred'/f'{r["index"]:08d}.png'
                idx['outputs'].append({'dataset': s['dataset'], 'sequence': s['sequence'], 'frame':r['frame'],
                                       'path': str(p), 'sha256':data.sha(p)})
        data.write_json(self.root/'index.json', idx)
        evaluate.score_external(argparse.Namespace(manifest=mp, index=self.root/'index.json', out=self.root/'external.json'))
        ext = data.read_json(self.root/'external.json')
        self.assertEqual(ext['datasets']['GoPro']['output_psnr_rgb8_frame_mean'],
                         report['datasets']['GoPro']['output_psnr_rgb8_frame_mean'])
        evaluate.compare(argparse.Namespace(reports=[self.root/'run/summary.json', self.root/'external.json'], out=self.root/'compare.json'))
        args.out = str(self.root/'fixed'); args.context=16; args.resume=False
        evaluate.run(args)
        with self.assertRaises(ValueError):
            evaluate.compare(argparse.Namespace(reports=[self.root/'run/summary.json', self.root/'fixed/summary.json'], out=self.root/'bad.json'))

if __name__ == '__main__':
    unittest.main()
