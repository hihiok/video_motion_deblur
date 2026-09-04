import argparse

import torch
from thop import profile

from train_nanovnr_waveshift_pagf_fullframe import build_model


class OutputOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, _ = self.model(x)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument(
        '--variant',
        choices=('haar_pagf', 'waveshift', 'waveshift_edge'),
        default='waveshift_edge',
    )
    parser.add_argument('--deploy-reparam', action='store_true')
    args = parser.parse_args()

    model = build_model(args.variant, grad_checkpoint=False).eval()
    if args.deploy_reparam:
        model.switch_to_deploy()
    x = torch.randn(1, args.frames, 3, args.height, args.width)
    macs, params = profile(OutputOnly(model), inputs=(x,), verbose=False)
    macs_per_frame = macs / args.frames
    # THOP does not count the functional fixed Haar conv/transposed-conv. This
    # conservative count treats every Haar coefficient multiply as one MAC.
    padded_h = args.height + args.height % 2
    padded_w = args.width + args.width % 2
    haar_macs_per_frame = 2 * padded_h * padded_w * 12
    total_with_haar = macs_per_frame + haar_macs_per_frame

    print('ARCHITECTURE=NanoVNRWaveShiftPAGF')
    print(f'VARIANT={args.variant}')
    print(f'DEPLOY_REPARAM={args.deploy_reparam}')
    print('GSTS_SHIFT_OP_MACS=0')
    print('NOTE=GSTS fusion convolutions are included by THOP')
    print(f'PARAMS={int(params)}')
    print(f'PARAMS_M={params / 1e6:.6f}')
    print(f'PROFILE_SIZE={args.width}x{args.height} T={args.frames}')
    print(f'THOP_TOTAL_MACS_G={macs / 1e9:.6f}')
    print(f'THOP_MACS_PER_FRAME_G={macs_per_frame / 1e9:.6f}')
    print(f'FIXED_HAAR_MACS_PER_FRAME_G={haar_macs_per_frame / 1e9:.6f}')
    print(f'TOTAL_MACS_PER_FRAME_G={total_with_haar / 1e9:.6f}')
    print(f'FLOPS_PER_FRAME_G_IF_2X_MAC={2 * total_with_haar / 1e9:.6f}')


if __name__ == '__main__':
    main()
