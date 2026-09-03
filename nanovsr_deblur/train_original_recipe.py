import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mixed_deblur import build_mixed_dataset
from models.nanovsr_deblur import NanoVSRDeblur


RECIPE_ID = 'nanovsr_original_charbonnier_mixed_v1'


class CharbonnierLoss(torch.nn.Module):
    """Same loss form as the official NanoVSR training code."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.sqrt(diff * diff + self.eps).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_loader(roots, num_frames, patch_size, batch_size, workers):
    ds, sampler, audit = build_mixed_dataset(
        roots, split='train', num_frames=num_frames,
        patch_size=patch_size, train=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn if workers > 0 else None,
    )
    return loader, audit


def print_audit(tag, audit):
    print(f'[{tag}] dataset audit:', flush=True)
    family_windows = {}
    for row in audit:
        print(
            f"  {row['family']}: blur={row['blur_root']} gt={row['gt_root']} "
            f"seq={row['sequences']} windows={row['windows']} T={row['frames_per_window']}",
            flush=True,
        )
        family_windows[row['family']] = family_windows.get(row['family'], 0) + row['windows']
    print(f'[{tag}] family_windows={family_windows}', flush=True)


def save_checkpoint(path, model, optimizer, scheduler, step, args, phase):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'recipe_id': RECIPE_ID,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': int(step),
        'phase': phase,
        'args': vars(args),
    }, path)


def run_preflight(args, roots, device):
    print('PREFLIGHT_ONLY=YES', flush=True)
    loader, audit = make_loader(
        roots, args.long_frames, args.patch_size,
        args.batch_size, args.workers,
    )
    print_audit('PREFLIGHT_LONG', audit)
    model = NanoVSRDeblur(args.num_feat, args.num_blocks).to(device).train()
    criterion = CharbonnierLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    batch = next(iter(loader))
    blur = batch['blur'].to(device, non_blocking=True)
    sharp = batch['sharp'].to(device, non_blocking=True)
    torch.cuda.reset_peak_memory_stats(device)
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=args.amp):
        pred = model(blur)
        loss = criterion(pred, sharp)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f'PREFLIGHT_LOSS={loss.item():.6f}', flush=True)
    print(f'PREFLIGHT_PEAK_GPU_GIB={peak:.3f}', flush=True)
    print('PREFLIGHT_STATUS=PASS', flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--dvd-root', required=True)
    ap.add_argument('--bsd-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--resume', default=None)

    ap.add_argument('--num-feat', type=int, default=48)
    ap.add_argument('--num-blocks', type=int, default=12)
    ap.add_argument('--short-frames', type=int, default=7)
    ap.add_argument('--long-frames', type=int, default=30)
    ap.add_argument('--switch-iter', type=int, default=50000)
    ap.add_argument('--total-iterations', type=int, default=150000)

    ap.add_argument('--patch-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eta-min', type=float, default=1e-7)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--preflight-only', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this training recipe.')
    device = torch.device('cuda')

    roots = {
        'GoPro': args.gopro_root,
        'DVD': args.dvd_root,
        'BSD': args.bsd_root,
    }
    print('RECIPE_ID=' + RECIPE_ID, flush=True)
    print('SOURCE_ROOTS=' + json.dumps(roots), flush=True)
    print(
        f'ARCH=NanoVSRDeblur F={args.num_feat} N={args.num_blocks} '
        f'shortT={args.short_frames} longT={args.long_frames}', flush=True,
    )
    print(
        f'OPT=Adam betas=(0.9,0.99) LR={args.lr} eta_min={args.eta_min} '
        f'loss=CharbonnierOnly grad_clip=0.5 patch={args.patch_size} '
        f'batch={args.batch_size}', flush=True,
    )

    if args.preflight_only:
        run_preflight(args, roots, device)
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = NanoVSRDeblur(args.num_feat, args.num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iterations, eta_min=args.eta_min,
    )
    criterion = CharbonnierLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        if ck.get('recipe_id') != RECIPE_ID:
            raise RuntimeError(
                f'Refusing foreign checkpoint resume: recipe_id={ck.get("recipe_id")}. '
                f'This run must stay clean and cannot resume old edge/temporal-loss training.'
            )
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_step = int(ck.get('step', 0))
        print(f'RESUMED_FROM={args.resume} STEP={start_step}', flush=True)

    current_long = start_step >= args.switch_iter
    current_t = args.long_frames if current_long else args.short_frames
    phase = 'long' if current_long else 'short'
    loader, audit = make_loader(
        roots, current_t, args.patch_size, args.batch_size, args.workers,
    )
    print_audit(f'PHASE_{phase.upper()}', audit)
    train_iter = iter(loader)

    model.train()
    for step in range(start_step + 1, args.total_iterations + 1):
        should_long = step > args.switch_iter
        if should_long != current_long:
            del train_iter, loader
            torch.cuda.empty_cache()
            current_long = should_long
            current_t = args.long_frames
            phase = 'long'
            loader, audit = make_loader(
                roots, current_t, args.patch_size, args.batch_size, args.workers,
            )
            print(f'SWITCH_PHASE_AT_STEP={step}: T={args.short_frames} -> T={args.long_frames}', flush=True)
            print_audit('PHASE_LONG', audit)
            train_iter = iter(loader)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        blur = batch['blur'].to(device, non_blocking=True)
        sharp = batch['sharp'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(blur)
            loss = criterion(pred, sharp)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 100 == 0 or step in (args.switch_iter, args.switch_iter + 1):
            lr = optimizer.param_groups[0]['lr']
            src = batch.get('source')
            if isinstance(src, (list, tuple)):
                src_text = ','.join(src)
            else:
                src_text = str(src)
            print(
                f'step={step}/{args.total_iterations} phase={phase} T={current_t} '
                f'loss={loss.item():.6f} lr={lr:.3e} grad_norm={float(grad_norm):.4f} '
                f'source={src_text}', flush=True,
            )

        if step % args.save_every == 0 or step == args.total_iterations:
            ckpt = out / f'step_{step:07d}.pth'
            save_checkpoint(ckpt, model, optimizer, scheduler, step, args, phase)
            save_checkpoint(out / 'latest.pth', model, optimizer, scheduler, step, args, phase)
            print(f'SAVED={ckpt}', flush=True)


if __name__ == '__main__':
    main()
