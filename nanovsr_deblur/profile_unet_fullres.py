import argparse

import torch
from thop import profile

from models.nanovsr_unet_fullres_deblur import NanoVSRFullResUNetDeblur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--base-channels', type=int, default=48)
    ap.add_argument('--mid-channels', type=int, default=64)
    ap.add_argument('--bottleneck-channels', type=int, default=96)
    ap.add_argument('--fullres-blocks', type=int, default=2)
    ap.add_argument('--mid-blocks', type=int, default=2)
    ap.add_argument('--bottleneck-blocks', type=int, default=4)
    args = ap.parse_args()

    model = NanoVSRFullResUNetDeblur(
        base_channels=args.base_channels,
        mid_channels=args.mid_channels,
        bottleneck_channels=args.bottleneck_channels,
        fullres_blocks=args.fullres_blocks,
        mid_blocks=args.mid_blocks,
        bottleneck_blocks=args.bottleneck_blocks,
        grad_checkpoint=False,
    ).eval()
    x = torch.randn(1, args.frames, 3, args.height, args.width)
    macs, params = profile(model, inputs=(x,), verbose=False)
    macs_per_frame = macs / args.frames

    print('ARCHITECTURE=NanoVSRFullResUNetDeblur')
    print(
        f'CONFIG=C={args.base_channels}/{args.mid_channels}/{args.bottleneck_channels} '
        f'blocks={args.fullres_blocks}/{args.mid_blocks}/{args.bottleneck_blocks}'
    )
    print('RECURRENT_STATE=FULL_RESOLUTION')
    print(f'PARAMS={int(params)}')
    print(f'PARAMS_M={params/1e6:.6f}')
    print(f'PROFILE_SIZE={args.width}x{args.height} T={args.frames}')
    print(f'TOTAL_MACS_G={macs/1e9:.6f}')
    print(f'MACS_PER_FRAME_G={macs_per_frame/1e9:.6f}')
    print(f'FLOPS_PER_FRAME_G_IF_2X_MAC={2.0*macs_per_frame/1e9:.6f}')


if __name__ == '__main__':
    main()
