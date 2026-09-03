import argparse

import torch
from torch.utils.data import DataLoader

from data.gopro_video import GoProVideoDataset
from models.nanovsr_unet_deblur import NanoVSRUNetDeblur


def psnr_per_frame(a, b):
    mse = (a - b).pow(2).mean(dim=(-3, -2, -1)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu')
    if ck.get('architecture') != 'NanoVSRUNetDeblur':
        raise RuntimeError(f'Unexpected architecture: {ck.get("architecture")}')
    cfg = ck.get('model_config', {})
    model = NanoVSRUNetDeblur(
        base_channels=int(cfg.get('base_channels', 32)),
        mid_channels=int(cfg.get('mid_channels', 48)),
        bottleneck_channels=int(cfg.get('bottleneck_channels', 64)),
        num_temporal_blocks=int(cfg.get('num_temporal_blocks', 6)),
        grad_checkpoint=False,
    ).to(device).eval()
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

    values = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if args.max_clips and i >= args.max_clips:
                break
            x = batch['blur'].to(device, non_blocking=True)
            y = batch['sharp'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                pred = model(x)
            frame_psnr = psnr_per_frame(pred.float().clamp(0, 1), y.float())[0]
            if args.center_only:
                values.append(frame_psnr[args.num_frames // 2].item())
            else:
                values.extend(frame_psnr.cpu().tolist())
            if (i + 1) % 20 == 0:
                print(f'clips={i+1} PSNR_RGB={sum(values)/len(values):.4f}', flush=True)

    mode = 'CENTER_ONLY' if args.center_only else 'ALL_FRAMES'
    print(f'ARCHITECTURE={ck.get("architecture")}', flush=True)
    print(f'CHECKPOINT_STEP={ck.get("step")}', flush=True)
    print(f'EVAL_MODE={mode}', flush=True)
    print(f'NUM_FRAMES={args.num_frames}', flush=True)
    print(f'CLIPS={min(len(ds), args.max_clips) if args.max_clips else len(ds)}', flush=True)
    print(f'PSNR_RGB={sum(values)/max(1,len(values)):.4f} dB', flush=True)


if __name__ == '__main__':
    main()
