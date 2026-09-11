import argparse
import torch
from torch.utils.data import DataLoader

from data.gopro_video import GoProVideoDataset
from models.network_nanovnr_nafnet_rgb import NanoVNRNAFNetRGB


def psnr_per_frame(a, b):
    mse = (a - b).pow(2).mean(dim=(-3, -2, -1)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu')
    if ck.get('architecture') != 'NanoVNRNAFNetRGB':
        raise RuntimeError(f'Unexpected architecture: {ck.get("architecture")}')
    model = NanoVNRNAFNetRGB(num_feat=12, grad_checkpoint=False).to(device).eval()
    model.load_state_dict(ck['model'], strict=True)
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num-frames', type=int, default=15)
    ap.add_argument('--max-clips', type=int, default=100)
    ap.add_argument('--center-only', action='store_true')
    ap.add_argument('--fp16', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_model(args.checkpoint, device)
    ds = GoProVideoDataset(
        args.gopro_root,
        split='test',
        num_frames=args.num_frames,
        patch_size=None,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    vals = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if args.max_clips and i >= args.max_clips:
                break
            x = batch['blur'].to(device, non_blocking=True)
            y = batch['sharp'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                pred, _ = model(x)
            p = psnr_per_frame(pred.float().clamp(0, 1), y.float())[0]
            if args.center_only:
                vals.append(p[args.num_frames // 2].item())
            else:
                vals.extend(p.cpu().tolist())
            if (i + 1) % 20 == 0:
                print(f'clips={i+1} PSNR_RGB={sum(vals)/len(vals):.4f}', flush=True)

    mode = 'CENTER_ONLY' if args.center_only else 'ALL_FRAMES'
    clips = min(len(ds), args.max_clips) if args.max_clips else len(ds)
    print(f'ARCHITECTURE={ck.get("architecture")}', flush=True)
    print(f'CHECKPOINT_STEP={ck.get("step")}', flush=True)
    print(f'EVAL_MODE={mode}', flush=True)
    print(f'NUM_FRAMES={args.num_frames}', flush=True)
    print(f'CLIPS={clips}', flush=True)
    print(f'PSNR_RGB={sum(vals)/max(1,len(vals)):.4f} dB', flush=True)


if __name__ == '__main__':
    main()
