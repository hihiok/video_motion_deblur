import argparse
import torch
from thop import profile

from models.network_nanovnr_nafnet_rgb import NanoVNRNAFNetRGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--frames', type=int, default=3)
    args = ap.parse_args()

    model = NanoVNRNAFNetRGB(num_feat=12, grad_checkpoint=False).eval()
    x = torch.randn(1, args.frames, 3, args.height, args.width)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, inp):
            y, _ = self.m(inp)
            return y

    wrapped = Wrapper(model)
    macs, params = profile(wrapped, inputs=(x,), verbose=False)
    macs_per_frame = macs / args.frames
    print('ARCHITECTURE=NanoVNRNAFNetRGB')
    print('MODEL_DIFF_VS_SUPPLIED=feat_extract input channels 4->3 ONLY')
    print('NUM_FEAT=12')
    print('PROP_CHANNELS=24,32,48,72')
    print(f'PARAMS={int(params)}')
    print(f'PARAMS_M={params/1e6:.6f}')
    print(f'PROFILE_SIZE={args.width}x{args.height} T={args.frames}')
    print(f'TOTAL_MACS_G={macs/1e9:.6f}')
    print(f'MACS_PER_FRAME_G={macs_per_frame/1e9:.6f}')
    print(f'FLOPS_PER_FRAME_G_IF_2X_MAC={2*macs_per_frame/1e9:.6f}')


if __name__ == '__main__':
    main()
