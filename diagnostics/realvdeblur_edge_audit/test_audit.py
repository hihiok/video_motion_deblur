"""CPU unit tests only; these do NOT certify actual CUDA/RealVDeblur execution."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("audit_realvdeblur.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metrics_identity(self):
        a = np.zeros((4, 5, 3), np.uint8)
        m = audit.metrics(a, a)
        self.assertEqual(m['mae_8bit'], 0)
        self.assertEqual(m['psnr_between_images_db_NOT_quality'], 'inf')
        json.dumps(m, allow_nan=False)

    def test_metrics_shape_mismatch(self):
        self.assertFalse(audit.metrics(np.zeros((2, 3)), np.zeros((3, 2)))['comparable'])

    def test_metrics_unsigned_subtraction(self):
        a = np.zeros((2, 2), np.uint8)
        b = np.full_like(a, 255)
        self.assertEqual(audit.metrics(a, b)['mae_8bit'], 255)

    def test_symlinks_and_natural_order(self):
        imgs = self.root / 'imgs'
        imgs.mkdir()
        Image.new('RGB', (20, 16)).save(self.root / 'source.png')
        for n in ('1.png', '2.png', '10.png'):
            (imgs / n).symlink_to(self.root / 'source.png')
        (imgs / 'bad.png').symlink_to(self.root / 'absent')
        d = audit.inventory(imgs)
        self.assertEqual(d['count'], 3)
        self.assertFalse(d['lexical_matches_natural'])
        self.assertEqual(d['broken_image_symlinks'], ['bad.png'])
        self.assertEqual(d['sizes'], {'20x16': 3})

    def test_native_sheet_does_not_resize(self):
        a = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        audit.comparison_sheet([('a', a), ('b', a)], self.root / 'sheet.png')
        canvas = audit.rgb(self.root / 'sheet.png')
        np.testing.assert_array_equal(canvas[36:, :10], a)
        np.testing.assert_array_equal(canvas[36:, 10:], a)

    def test_nearest_is_only_display_zoom(self):
        a = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        audit.comparison_sheet([('a', a)], self.root / 'sheet.png', zoom=4)
        result = audit.rgb(self.root / 'sheet.png')[36:]
        np.testing.assert_array_equal(result, np.repeat(np.repeat(a, 4, 0), 4, 1))

    def test_roi_in_bounds(self):
        for r in audit.rois(np.zeros((720, 1280, 3), np.uint8)):
            x1, y1, x2, y2 = r
            self.assertTrue(0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720)
            self.assertEqual(x2-x1, 160)

    def test_fp32_export_same_truncation(self):
        a = torch.linspace(-1, 1, 60).reshape(3, 1, 4, 5).to(torch.bfloat16)
        result = audit.save_rgb_fp32(a, self.root / 'image.png')
        expected = ((a.float().squeeze(1).permute(1, 2, 0) + 1) * 127.5).clamp(0,255).to(torch.uint8).numpy()
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (4,5,3))

    def test_export_rejects_nan_before_uint8(self):
        a = torch.zeros(3, 1, 4, 5)
        a[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            audit.save_rgb_fp32(a, self.root / 'bad.png')

    def test_gpu_selection_numeric_and_uuid(self):
        output = '4, GPU-four, GPU, 98000, 83074, 580\n6, GPU-six, GPU, 98000, 97230, 580\n0, GPU-zero, GPU, 98000, 999, 580\n'
        args = argparse.Namespace(out=self.root, mode='capture', precision='bfloat16', allowed_gpus=None, min_free_mib=24576)
        with patch.object(audit, 'command', return_value=argparse.Namespace(stdout=output)), patch.dict(os.environ, {}, clear=True):
            r = audit.select_gpu(args)
            self.assertEqual(r['index'], 6)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-six')

    def test_gpu_respects_allocation(self):
        output = '4, GPU-four, GPU, 98000, 83074, 580\n6, GPU-six, GPU, 98000, 97230, 580\n'
        args = argparse.Namespace(out=self.root, mode='capture', precision='bfloat16', allowed_gpus='4', min_free_mib=24576)
        with patch.object(audit, 'command', return_value=argparse.Namespace(stdout=output)), patch.dict(os.environ, {'SLURM_JOB_ID': '42'}, clear=True):
            self.assertEqual(audit.select_gpu(args)['index'], 4)

    def test_gpu_no_free_stops(self):
        args = argparse.Namespace(out=self.root, mode='capture', precision='bfloat16', allowed_gpus=None, min_free_mib=24576)
        with patch.object(audit, 'command', return_value=argparse.Namespace(stdout='0, GPU-zero, GPU, 98000, 10, 580')), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'NO_FREE_GPU'):
                audit.select_gpu(args)

    def test_source_exact_and_changed_bytes(self):
        repo = self.root / 'repo'
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()
        git('init')
        git('config', 'user.email', 'test@example.invalid')
        git('config', 'user.name', 'Unit Test')
        (repo / 'inference.py').write_text('print(1)\n')
        (repo / 'model').mkdir()
        (repo / 'model/a.py').write_text('a=1\n')
        git('add', '.')
        git('commit', '-m', 'fixture')
        sha = git('rev-parse', 'HEAD')
        with patch.object(audit, 'PIN', sha):
            self.assertEqual(audit.source_state(repo, self.root/'audit1')['gpu_gate'], 'PASS')
            (repo/'model/a.py').write_text('a=2\n')
            r = audit.source_state(repo, self.root/'audit2')
            self.assertEqual(r['gpu_gate'], 'BLOCKED_LOCAL_SOURCE_DIFF')
            self.assertIn('a=2', (self.root/'audit2/local_code_changes_PRIVATE.diff').read_text())

    def test_untracked_importable_source_is_flagged(self):
        repo = self.root / 'repo'
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()
        git('init'); git('config', 'user.email', 'test@example.invalid'); git('config', 'user.name', 'Test')
        (repo/'inference.py').write_text('pass\n'); (repo/'model').mkdir(); (repo/'model/a.py').write_text('pass\n')
        git('add', '.'); git('commit', '-m', 'fixture')
        sha = git('rev-parse','HEAD')
        (repo/'model/new.py').write_text('pass\n')
        with patch.object(audit, 'PIN', sha):
            r = audit.source_state(repo, self.root/'result')
            self.assertEqual(r['unexpected_python'], ['model/new.py'])
            self.assertNotEqual(r['gpu_gate'], 'PASS')

    def test_protected_output_rejected(self):
        proc = subprocess.run([os.sys.executable, str(Path(audit.__file__)), 'summary', '--root', str(self.root),
                               '--out', str(self.root/'benchmark/outputs/realvdeblur_blackwell/frames')], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('isolated NEW diagnostic directory', proc.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
