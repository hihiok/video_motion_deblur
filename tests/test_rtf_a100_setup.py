import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from rtf_t6.protocol import a100_world_size_config, rank_report_path
from tools import prepare_rtf_t3_a100 as setup
from tools.audit_rtf_t6_data import resolution_inventory
from tools import select_rtf_a100_mode as selector


class A100SetupTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def test_new_paths_and_split_roots_are_external_and_no_overwrite(self):
        root = self.root / 'newserver/motion_deblur'
        datasets = self.root / 'newserver/dataset'
        for name in ('GoPro', 'BSD_train', 'BSD_val', 'DeepVideoDeblurring_Dataset'):
            (datasets / name).mkdir(parents=True)
        pretrained = root / 'weights/GoPro_RT_Focuser_Standard_256.pth'
        pretrained.parent.mkdir(parents=True)
        pretrained.write_bytes(b'fixture, not a pretrained neural network')
        argv = ['prepare', '--root', str(root), '--dataset-base', str(datasets),
            '--bsd-root', str(datasets / 'BSD_train'), '--bsd-root', str(datasets / 'BSD_val'), '--gpus', '2']
        with patch.object(setup, 'OFFICIAL_SHA', setup.sha256_file(pretrained)), patch.object(sys, 'argv', argv):
            setup.main()
            run = root / 'runs/rtfocuser_shift_dst_t3_gopro_bsd_dvd_a100_v4_2gpu'
            cfg = yaml.safe_load((run / 'runtime_config.yaml').read_text())
            self.assertEqual(cfg['datasets']['bsd']['roots'], [str(datasets / 'BSD_train'), str(datasets / 'BSD_val')])
            self.assertEqual(cfg['pretrained'], str(pretrained))
            self.assertEqual(cfg['output'], str(run))
            self.assertEqual(cfg['train']['clip_length'], 3)
            self.assertEqual(cfg['train']['total_iters'], 180000)
            self.assertNotIn('/mnt/ssd1', (run / 'runtime_config.yaml').read_text())
            metadata = json.loads((run / 'a100_setup.json').read_text())
            self.assertEqual(metadata['config'], str(run / 'runtime_config.yaml'))
            setup.main()
            (run / 'runtime_config.yaml').write_text('different: config\n')
            with self.assertRaises(FileExistsError):
                setup.main()

    def test_missing_dataset_or_wrong_weight_rejected(self):
        with self.assertRaises(FileNotFoundError):
            setup.dataset_roots(self.root, 'dvd', None)
        bad = self.root / 'incorrect.pth'
        bad.write_bytes(b'not the official checkpoint')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            setup.official_checkpoint(self.root, str(bad))

    def test_inventory_finds_1080p_and_checks_nonfirst_frame(self):
        sequences = []
        for name, size in [('C0001', (1280, 720)), ('C0041', (1920, 1080))]:
            pairs = []
            for kind in ('blur', 'gt'):
                folder = self.root / name / kind
                folder.mkdir(parents=True)
                files = []
                for index in range(3):
                    path = folder / f'{index:03d}.png'
                    Image.new('RGB', size, color=(index, 2, 3)).save(path)
                    files.append(path)
                pairs.append(tuple(files))
            sequences.append(SimpleNamespace(domain='dvd', name=name, blur=pairs[0], gt=pairs[1], length=3))
        errors = []
        inventory = resolution_inventory(sequences, errors)
        self.assertEqual(errors, [])
        self.assertEqual(inventory['1080x1920'], {'sequences': ['C0041'], 'frames': 3})
        Image.new('RGB', (64, 48)).save(sequences[1].gt[-1])
        resolution_inventory(sequences, errors)
        self.assertTrue(any('header sizes differ' in e for e in errors))

    def test_two_gpu_budget_and_schedule_match_one_gpu(self):
        base = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/rtf_t3_gopro_bsd_dvd_a100.yaml').read_text())
        for size in (1, 2):
            cfg = a100_world_size_config(base, size)
            t = cfg['train']
            self.assertEqual(t['clip_length'] * t['gradient_accumulation'] * size, 12)
            self.assertEqual(t['total_iters'] // t['gradient_accumulation'], 90000)
            self.assertEqual(t['total_iters'] * size * t['clip_length'], 1080000)
            self.assertEqual(t['samples_per_epoch'] // size // t['gradient_accumulation'], 5000)
            self.assertEqual(t['workers'] * size, 4)
            for key in ('warmup_iters', 'save_every', 'validate_every'):
                self.assertEqual(t[key] // t['gradient_accumulation'], base['train'][key] // 4)
            for key in ('temporal_start_iter', 'temporal_ramp_iters'):
                self.assertEqual(cfg['loss'][key] // t['gradient_accumulation'], base['loss'][key] // 4)
        with self.assertRaises(ValueError):
            a100_world_size_config(base, 8)
        self.assertEqual(str(rank_report_path('report.json', 0, 1)), 'report.json')
        self.assertNotEqual(rank_report_path('report.json', 0, 2), rank_report_path('report.json', 1, 2))

    def test_selection_uses_measured_update_time_and_rejects_stale_config(self):
        base = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/rtf_t3_gopro_bsd_dvd_a100.yaml').read_text())
        for size, seconds in [(1, 1.0), (2, 0.6)]:
            run = self.root / f'{size}gpu'
            (run / 'benchmark').mkdir(parents=True)
            cfg_path = run / 'runtime_config.yaml'
            cfg_path.write_text(yaml.safe_dump(a100_world_size_config(base, size)))
            report = {'status': 'BENCHMARK_PASS', 'world_size': size, 'frames_per_update': 12,
                      'weights_discarded': True, 'config_sha256': setup.sha256_file(cfg_path),
                      'git_commit': 'fixture_commit', 'measured_updates': 12,
                      'mean_seconds_per_update': seconds, 'pretrained_sha256': 'fixture_weight',
                      'torch_version': 'fixture', 'cuda_version': 'fixture',
                      'cuda_visible_devices': '2' if size == 1 else '2,5'}
            (run / 'benchmark/benchmark.json').write_text(json.dumps(report))
        output = self.root / 'selection.json'
        argv = ['select', '--single-run', str(self.root / '1gpu'), '--dual-run', str(self.root / '2gpu'), '--output', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(selector.subprocess, 'check_output', return_value='fixture_commit\n'):
            selector.main()
            self.assertEqual(json.loads(output.read_text())['world_size'], 2)
            dual = self.root / '2gpu/benchmark/benchmark.json'
            report = json.loads(dual.read_text())
            report['mean_seconds_per_update'] = 0.97
            dual.write_text(json.dumps(report))
            selector.main()
            self.assertEqual(json.loads(output.read_text())['world_size'], 1)
            (self.root / '2gpu/runtime_config.yaml').write_text('changed: true\n')
            with self.assertRaisesRegex(ValueError, 'config_sha256'):
                selector.main()


if __name__ == '__main__':
    unittest.main()
