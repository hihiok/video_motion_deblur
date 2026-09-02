import argparse
import torch
from models.nanovsr_deblur import NanoVSRDeblur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-feat', type=int, default=48)
    ap.add_argument('--num-blocks', type=int, default=12)
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--frames', type=int, default=7)
    args = ap.parse_args()

    model = NanoVSRDeblur(args.num_feat, args.num_blocks).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f'params={params} ({params/1e6:.4f} M)')
    try:
        from thop import profile
        x = torch.randn(1, args.frames, 3, args.height, args.width)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        print(f'MACs_clip={macs/1e9:.3f} G @ T={args.frames}, {args.width}x{args.height}')
        print(f'MACs_per_frame={macs/args.frames/1e9:.3f} G')
        print(f'FLOPs_per_frame_2xMAC={2*macs/args.frames/1e9:.3f} G')
    except Exception as e:
        print('THOP profiling unavailable:', repr(e))

if __name__ == '__main__':
    main()
