"""Run on CPU by default; WAVESHIFT_TEST_DEVICE=cuda requires real CUDA."""

import argparse
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amp_training import restore_scaler, training_update
from train_nanovnr_waveshift_pagf_fullframe import build_model, run_model, save_checkpoint


class AmpRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get('WAVESHIFT_TEST_DEVICE', 'cpu'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            self.fail('CUDA explicitly required but unavailable')
        torch.manual_seed(17)
        self.model = torch.nn.Linear(2, 1, bias=False).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=150000, eta_min=1e-7)
        self.scaler = torch.amp.GradScaler(
            self.device.type, init_scale=4096, growth_interval=10000)
        self.x = torch.ones(2, 2, device=self.device)
        self.records = []

    def loss(self):
        return self.model(self.x).square().mean()

    def update(self, **kwargs):
        return training_update(self.model, self.optimizer, self.scaler,
                               kwargs.pop('loss_closure', self.loss),
                               emit=self.records.append, **kwargs)

    def test_overflow_retries_same_batch_then_exactly_one_adam_update(self):
        reference = copy.deepcopy(self.model)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=3e-4)
        reference(self.x).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
        reference_optimizer.step()
        calls = []
        before = self.model.weight.detach().clone()

        def inject_once(grad):
            calls.append(id(self.x))
            if len(calls) == 1:
                return torch.full_like(grad, float('inf'))
            self.assertEqual(len(self.optimizer.state), 0)
            self.assertEqual(self.scheduler.last_epoch, 0)
            torch.testing.assert_close(self.model.weight, before, rtol=0, atol=0)
            return grad

        self.model.weight.register_hook(inject_once)
        result = self.update()
        self.scheduler.step()
        self.assertEqual(calls, [id(self.x), id(self.x)])
        self.assertEqual(result['overflow_retries'], 1)
        self.assertEqual(self.scaler.get_scale(), 2048)
        self.assertEqual(self.scheduler.last_epoch, 1)
        self.assertEqual(self.optimizer.state[self.model.weight]['step'].item(), 1)
        torch.testing.assert_close(self.model.weight, reference.weight)
        self.assertEqual([r['event'] for r in self.records], ['AMP_OVERFLOW', 'AMP_RECOVERED'])

    def test_persistent_overflow_stops_after_initial_plus_eight_retries(self):
        before = self.model.weight.detach().clone()
        self.model.weight.register_hook(lambda g: torch.full_like(g, float('inf')))
        with self.assertRaisesRegex(RuntimeError, 'RETRY_EXHAUSTED'):
            self.update()
        self.assertEqual(len(self.records), 9)
        self.assertEqual(self.scheduler.last_epoch, 0)
        self.assertEqual(len(self.optimizer.state), 0)
        self.assertIsNone(self.model.weight.grad)
        torch.testing.assert_close(self.model.weight, before, rtol=0, atol=0)

    def test_nonfinite_loss_stops_without_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'NON_FINITE_LOSS'):
            self.update(loss_closure=lambda: self.loss() * float('nan'))
        self.assertEqual(len(self.records), 1)
        self.assertEqual(self.scaler.get_scale(), 4096)
        self.assertEqual(len(self.optimizer.state), 0)

    def test_finite_element_norm_overflow_cannot_update_optimizer(self):
        self.model.weight.register_hook(lambda g: torch.full_like(g, 1e30))
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            self.update()
        self.assertEqual(self.records[0]['event'], 'GRADIENT_NORM_FAILURE')
        self.assertEqual(len(self.optimizer.state), 0)
        self.assertEqual(self.scaler.get_scale(), 4096)

    def test_nonfinite_gradient_without_scaler_is_fatal(self):
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        self.model.weight.register_hook(lambda g: torch.full_like(g, float('inf')))
        with self.assertRaisesRegex(RuntimeError, 'WITHOUT_AMP'):
            self.update()
        self.assertEqual(len(self.optimizer.state), 0)

    def test_scaler_legacy_and_roundtrip(self):
        self.assertEqual(restore_scaler(self.scaler, {}), 'LEGACY_CHECKPOINT_SCALER_INITIALIZED')
        self.assertEqual(self.scaler.get_scale(), 4096)
        self.update()
        state = self.scaler.state_dict()
        restored = torch.amp.GradScaler(self.device.type, init_scale=2)
        self.assertEqual(restore_scaler(restored, {'scaler': state}), 'CHECKPOINT_SCALER_RESTORED')
        self.assertEqual(restored.state_dict(), state)
        with self.assertRaisesRegex(RuntimeError, 'MODE_MISMATCH'):
            restore_scaler(restored, {'scaler': {}})

    def test_six_frame_checkpoint_backward_and_checkpoint_state(self):
        model = build_model('waveshift_edge').to(self.device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000)
        x = torch.rand(1, 6, 3, 16, 16, device=self.device)
        target = torch.rand_like(x)

        def loss():
            with torch.autocast(self.device.type, dtype=torch.float16,
                                enabled=self.device.type == 'cuda'):
                prediction, _ = run_model(model, x, True)
                return torch.sqrt((prediction - target).square() + 1e-6).mean()

        result = training_update(model, optimizer, self.scaler, loss, emit=self.records.append)
        scheduler.step()
        self.assertTrue(all(torch.isfinite(p).all().item() for p in model.parameters()))
        args = argparse.Namespace(variant='waveshift_edge')
        stats = {'overflow_attempts': result['overflow_retries'],
                 'recovered_batches': int(result['overflow_retries'] > 0)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            save_checkpoint(path, model, optimizer, scheduler, 1, args, self.scaler, stats)
            restored = torch.load(path, map_location=self.device, weights_only=False)
        self.assertEqual(restored['scaler'], self.scaler.state_dict())
        self.assertEqual(restored['amp_stats'], stats)
        self.assertEqual(restored['scheduler']['last_epoch'], 1)
        self.assertEqual(restored['step'], 1)
        self.assertEqual(restored['model_config'], model.config_dict())


if __name__ == '__main__':
    unittest.main()
