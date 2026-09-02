import argparse
import math
import torch
from torch.utils.data import DataLoader
from data.gopro_video import GoProVideoDataset
from models.nanovsr_deblur import NanoVSRDeblur


def psnr(a, b):
    mse = (a - b).pow(2).mean(dim=(-3,-2,-1)).clamp_min(1e-12)
    return (-10.0 * torch.log10(mse)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num-frames', type=int, default=7)
    ap.add_argument('--max-clips', type=int, default=0)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location='cpu')
    sa = ck.get('args', {})
    model = NanoVSRDeblur(int(sa.get('num_feat',48)), int(sa.get('num_blocks',12))).to(device).eval()
    model.load_state_dict(ck['model'], strict=True)
    ds = GoProVideoDataset(args.gopro_root, 'test', args.num_frames, patch_size=None)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    vals=[]
    with torch.no_grad():
        for i,b in enumerate(dl):
            if args.max_clips and i >= args.max_clips: break
            x=b['blur'].to(device); y=b['sharp'].to(device)
            pred=model(x)
            vals.append(psnr(pred,y).item())
            if (i+1)%20==0: print(f'clips={i+1} PSNR={sum(vals)/len(vals):.3f}')
    print(f'FINAL clips={len(vals)} PSNR_RGB={sum(vals)/max(1,len(vals)):.4f} dB')

if __name__ == '__main__': main()
