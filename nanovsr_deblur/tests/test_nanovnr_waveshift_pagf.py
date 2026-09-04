import copy
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.network_nanovnr_waveshift_pagf import (  # noqa: E402
    GroupedSpatialTemporalShift,
    HaarWavelet,
    NanoVNRWaveShiftPAGF,
    PAGF,
)


class WaveShiftPAGFTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_haar_round_trip(self):
        module = HaarWavelet(3)
        x = torch.randn(2, 3, 18, 26)
        ll, hf = module.decompose(x)
        restored = module.reconstruct(ll, hf)
        self.assertEqual(tuple(ll.shape), (2, 3, 9, 13))
        self.assertEqual(tuple(hf.shape), (2, 9, 9, 13))
        self.assertLess((x - restored).abs().max().item(), 1e-6)

    def test_pagf_gate_is_bounded_and_has_gradient(self):
        module = PAGF(12, initial_history_weight=0.1)
        current = torch.randn(1, 12, 8, 8, requires_grad=True)
        history = torch.randn(1, 12, 8, 8, requires_grad=True)
        output, gate = module(current, history, return_gate=True)
        self.assertGreaterEqual(gate.min().item(), 0.0)
        self.assertLessEqual(gate.max().item(), 1.0)
        output.mean().backward()
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertTrue(torch.isfinite(history.grad).all())

    def test_gsts_shape_and_temporal_effect(self):
        module = GroupedSpatialTemporalShift(12, spatial_radius=2)
        x = torch.zeros(1, 5, 12, 12, 12)
        x[:, 2] = 1.0
        y = module(x)
        self.assertEqual(y.shape, x.shape)
        # The fusion path receives shifted neighbors; the operation must not be
        # an unconditional identity at its non-zero initial residual scale.
        self.assertGreater((y - x).abs().sum().item(), 0.0)

    def test_model_shape_state_core_and_backward(self):
        model = NanoVNRWaveShiftPAGF(grad_checkpoint=False).train()
        x = torch.randn(1, 5, 3, 33, 47, requires_grad=True)
        output, state = model(x, core_start=1, core_end=4)
        self.assertEqual(tuple(output.shape), (1, 3, 3, 33, 47))
        self.assertEqual(tuple(state.shape), (1, 12, 17, 24))
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_reparameterization_equivalence(self):
        model = NanoVNRWaveShiftPAGF().eval()
        deployed = copy.deepcopy(model)
        x = torch.randn(1, 3, 3, 24, 32)
        with torch.no_grad():
            expected, expected_state = model(x)
            deployed.switch_to_deploy()
            actual, actual_state = deployed(x)
        self.assertLess((expected - actual).abs().max().item(), 2e-5)
        self.assertLess((expected_state - actual_state).abs().max().item(), 2e-5)

    def test_config_locks_gsts_to_ll(self):
        config = NanoVNRWaveShiftPAGF().config_dict()
        self.assertEqual(config['gsts_branch'], 'LL_only')
        self.assertEqual(config['recurrent_resolution'], '1/2')
        self.assertEqual(config['gsts_radii'], [2, 4])
        self.assertTrue(config['edge_aware_hf'])


if __name__ == '__main__':
    unittest.main()
