"""Mandatory architecture and numerical audit before training."""

import argparse
import copy
import json

import torch

from models.network_nanovnr_waveshift_pagf import (
    HaarWavelet,
    NanoVNRWaveShiftPAGF,
    PAGF,
    RepConv2d,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--frames', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = NanoVNRWaveShiftPAGF().to(device).eval()
    config = model.config_dict()
    assert config['num_feat'] == 12
    assert config['input_channels'] == 3
    assert config['haar'] is True
    assert config['recurrent_resolution'] == '1/2'
    assert config['gsts_blocks'] == 2
    assert config['gsts_radii'] == [2, 4]
    assert config['gsts_branch'] == 'LL_only'
    assert config['pagf'] is True
    assert config['additive_recurrence'] is True
    assert config['prop_channels'] == [24, 32, 48, 72]
    assert config['edge_aware_hf'] is True
    assert config['repconv'] is True
    assert model.feat_extract.in_channels == 3
    assert model.feat_extract.out_channels == 12
    assert sum(isinstance(m, PAGF) for m in model.modules()) >= 4
    assert sum(isinstance(m, RepConv2d) for m in model.modules()) >= 5

    haar = HaarWavelet(12).to(device)
    even = torch.randn(1, 12, 32, 48, device=device)
    ll, hf = haar.decompose(even)
    round_trip = haar.reconstruct(ll, hf)
    haar_diff = (even - round_trip).abs().max().item()
    if haar_diff >= 1e-6:
        raise RuntimeError(f'Haar round-trip failed: {haar_diff}')

    x = torch.randn(
        1, args.frames, 3, args.height, args.width, device=device
    )
    with torch.no_grad():
        expected, expected_state = model(x)
        deployed = copy.deepcopy(model).switch_to_deploy()
        actual, actual_state = deployed(x)
    output_diff = (expected - actual).abs().max().item()
    state_diff = (expected_state - actual_state).abs().max().item()
    if output_diff >= 2e-5 or state_diff >= 2e-5:
        raise RuntimeError(
            f'Reparameterization mismatch: output={output_diff} state={state_diff}'
        )

    print('ARCHITECTURE_AUDIT=PASS')
    print('ARCHITECTURE=NanoVNRWaveShiftPAGF')
    print('MODEL_CONFIG=' + json.dumps(config, sort_keys=True))
    print('GSTS_BRANCH=LL_ONLY')
    print('GSTS_BLOCKS=2 RADII_HALF_RES=2,4')
    print('PAGF=STATE_AND_BIDIRECTIONAL_FUSION')
    print('EDGE_AWARE=HF_ONLY')
    print('ADDITIVE_RECURRENCE=YES')
    print('REP_CONV_DEPLOY_FUSION=PASS')
    print(f'HAAR_ROUNDTRIP_MAX_ABS_DIFF={haar_diff:.9g}')
    print(f'DEPLOY_OUTPUT_MAX_ABS_DIFF={output_diff:.9g}')
    print(f'DEPLOY_STATE_MAX_ABS_DIFF={state_diff:.9g}')


if __name__ == '__main__':
    main()
