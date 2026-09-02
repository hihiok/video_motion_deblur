import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.gopro_video import GoProVideoDataset
from models.nanovsr_deblur import NanoVSRDeblur


def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b).pow(2) + eps * eps).mean()


def gradient_loss(a, b):
    ax = a[..., :, 1:] - a[..., :, :-1]
    bx = b[..., :, 1:] - b[..., :, :-1]
    ay = a[..., 1:, :] - a[..., :-1, :]
    by = b[..., 1:, :] - b[..., :-1, :]
    return charbonnier(ax, bx) + charbonnier(ay, by)


def temporal_delta_loss(pred, gt):
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    return charbonnier(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])


def save_ckpt(path, model, opt, step, args, best=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': step,
                'args': vars(args), 'best': best}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--teacher-root', default=None, help='Optional cached teacher outputs with split/seq/frame layout')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--num-feat', type=int, default=48)
    ap.add_argument('--num-blocks', type=int, default=12)
    ap.add_argument('--num-frames', type=int, default=7)
    ap.add_argument('--patch-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda-edge', type=float, default=0.05)
    ap.add_argument('--lambda-temp', type=float, default=0.10)
    ap.add_argument('--lambda-distill', type=float, default=0.20)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--save-every', type=int, default=5000)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = GoProVideoDataset(args.gopro_root, 'train', args.num_frames, args.patch_size, teacher_root=args.teacher_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    model = NanoVSRDeblur(args.num_feat, args.num_blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ck['model'], strict=True)
        if 'optimizer' in ck: opt.load_state_dict(ck['optimizer'])
        start = int(ck.get('step', 0))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    it = iter(dl)
    model.train()
    for step in range(start + 1, args.steps + 1):
        try: batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        blur = batch['blur'].to(device, non_blocking=True)
        sharp = batch['sharp'].to(device, non_blocking=True)
        teacher = batch.get('teacher')
        if teacher is not None: teacher = teacher.to(device, non_blocking=True)

        progress = step / max(1, args.steps)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups: pg['lr'] = max(lr, 1e-7)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(blur)
            l_rec = charbonnier(pred, sharp)
            l_edge = gradient_loss(pred, sharp)
            l_temp = temporal_delta_loss(pred, sharp)
            l_dist = pred.new_zeros(())
            if teacher is not None and args.lambda_distill > 0:
                # Blend teacher supervision with GT so student cannot inherit teacher artifacts blindly.
                l_dist = charbonnier(pred, teacher)
            loss = l_rec + args.lambda_edge*l_edge + args.lambda_temp*l_temp + args.lambda_distill*l_dist
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if step == 1 or step % 100 == 0:
            print(f'step={step} loss={loss.item():.6f} rec={l_rec.item():.6f} edge={l_edge.item():.6f} temp={l_temp.item():.6f} dist={l_dist.item():.6f} lr={lr:.3e}', flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(out / f'step_{step:07d}.pth', model, opt, step, args)
            save_ckpt(out / 'latest.pth', model, opt, step, args)


if __name__ == '__main__':
    main()
