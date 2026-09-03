import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.mixed_deblur import build_mixed_dataset
from models.nanovsr_unet_fullres_deblur import NanoVSRFullResUNetDeblur


RECIPE_ID = 'nanovsr_unet_fullres_recurrence_charbonnier_mix_v2'
ARCHITECTURE = 'NanoVSRFullResUNetDeblur'


class CharbonnierLoss(torch.nn.Module):
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


def build_model(args, device):
    return NanoVSRFullResUNetDeblur(
        base_channels=args.base_channels,
        mid_channels=args.mid_channels,
        bottleneck_channels=args.bottleneck_channels,
        fullres_blocks=args.fullres_blocks,
        mid_blocks=args.mid_blocks,
        bottleneck_blocks=args.bottleneck_blocks,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)


def build_loader(roots, num_frames, workers):
    dataset, sampler, audit = build_mixed_dataset(
        roots,
        split='train',
        num_frames=num_frames,
        patch_size=None,
        train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn if workers > 0 else None,
    )
    return dataset, loader, audit


def print_audit(tag, audit):
    print(f'[{tag}] dataset audit:', flush=True)
    totals = {}
    for row in audit:
        print(
            f"  family={row['family']} blur={row['blur_root']} gt={row['gt_root']} "
            f"seq={row['sequences']} windows={row['windows']} T={row['frames_per_window']}",
            flush=True,
        )
        totals[row['family']] = totals.get(row['family'], 0) + row['windows']
    print(f'[{tag}] FAMILY_WINDOWS={totals}', flush=True)


def save_checkpoint(path, model, optimizer, scheduler, step, args, phase):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'recipe_id': RECIPE_ID,
        'architecture': ARCHITECTURE,
        'model_config': model.config_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': int(step),
        'phase': phase,
        'args': vars(args),
    }, path)


def representative_components_by_family_resolution(dataset):
    reps = {}
    for component in dataset.datasets:
        if not component.samples:
            continue
        _, _, pairs = component.samples[0]
        blur_path = pairs[0][0]
        with Image.open(blur_path) as im:
            w, h = im.size
        key = (component.family, int(h), int(w))
        reps.setdefault(key, component)
    return reps


def run_preflight(args, roots, device):
    print('PREFLIGHT_ONLY=YES', flush=True)
    dataset, _, audit = build_loader(roots, args.long_frames, workers=0)
    print_audit('PREFLIGHT_T30', audit)
    reps = representative_components_by_family_resolution(dataset)
    print('PREFLIGHT_RESOLUTION_KEYS=' + json.dumps([list(k) for k in reps.keys()]), flush=True)

    failed = []
    for (family, h, w), component in reps.items():
        print(f'PREFLIGHT_BEGIN family={family} T={args.long_frames} H={h} W={w}', flush=True)
        torch.cuda.empty_cache()
        model = build_model(args, device).train()
        criterion = CharbonnierLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        sample = component[0]
        blur = sample['blur'].unsqueeze(0).to(device, non_blocking=True)
        sharp = sample['sharp'].unsqueeze(0).to(device, non_blocking=True)
        torch.cuda.reset_peak_memory_stats(device)
        pred = None
        try:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(blur)
                if pred.shape != sharp.shape:
                    raise RuntimeError(f'Output/GT shape mismatch: {tuple(pred.shape)} vs {tuple(sharp.shape)}')
                loss = criterion(pred, sharp)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f'PREFLIGHT_PASS family={family} T={args.long_frames} H={h} W={w} '
                f'loss={loss.item():.6f} peak_gpu_gib={peak:.3f}', flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            failed.append((family, h, w))
            print(f'PREFLIGHT_OOM family={family} T={args.long_frames} H={h} W={w}', flush=True)
        finally:
            del model, criterion, optimizer, scaler, blur, sharp
            if pred is not None:
                del pred
            torch.cuda.empty_cache()

    if failed:
        print('PREFLIGHT_STATUS=FAIL', flush=True)
        print('PREFLIGHT_FAILED=' + json.dumps([list(x) for x in failed]), flush=True)
        raise RuntimeError(
            'Native full-frame T=30 with full-resolution recurrent state OOM. '
            'Do NOT crop, resize, lower T, move recurrence to low resolution, or shrink the model. '
            'Report HUMAN_ACTION_REQUIRED.'
        )
    print('PREFLIGHT_STATUS=PASS', flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--dvd-root', required=True)
    ap.add_argument('--bsd-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--resume', default=None)

    ap.add_argument('--base-channels', type=int, default=48)
    ap.add_argument('--mid-channels', type=int, default=64)
    ap.add_argument('--bottleneck-channels', type=int, default=96)
    ap.add_argument('--fullres-blocks', type=int, default=2)
    ap.add_argument('--mid-blocks', type=int, default=2)
    ap.add_argument('--bottleneck-blocks', type=int, default=4)

    ap.add_argument('--short-frames', type=int, default=7)
    ap.add_argument('--long-frames', type=int, default=30)
    ap.add_argument('--switch-iter', type=int, default=50000)
    ap.add_argument('--total-iterations', type=int, default=150000)

    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eta-min', type=float, default=1e-7)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--grad-checkpoint', action='store_true')
    ap.add_argument('--preflight-only', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required.')
    device = torch.device('cuda')

    roots = {
        'GoPro': args.gopro_root,
        'DVD': args.dvd_root,
        'BSD': args.bsd_root,
    }
    print('RECIPE_ID=' + RECIPE_ID, flush=True)
    print('ARCHITECTURE=' + ARCHITECTURE, flush=True)
    print('SOURCE_ROOTS=' + json.dumps(roots), flush=True)
    print(
        f'CONFIG=C={args.base_channels}/{args.mid_channels}/{args.bottleneck_channels} '
        f'blocks={args.fullres_blocks}/{args.mid_blocks}/{args.bottleneck_blocks} '
        f'shortT={args.short_frames} longT={args.long_frames}', flush=True,
    )
    print(
        f'FULL_FRAME=YES RANDOM_CROP=NO RESIZE=NO BATCH=1 '
        f'RECURRENT_STATE=FULL_RESOLUTION GRAD_CHECKPOINT={args.grad_checkpoint} AMP={args.amp}', flush=True,
    )
    print(
        f'OPT=Adam betas=(0.9,0.99) LR={args.lr} eta_min={args.eta_min} '
        f'LOSS=CharbonnierOnly grad_clip=0.5', flush=True,
    )

    if args.preflight_only:
        run_preflight(args, roots, device)
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iterations, eta_min=args.eta_min,
    )
    criterion = CharbonnierLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        if ck.get('recipe_id') != RECIPE_ID or ck.get('architecture') != ARCHITECTURE:
            raise RuntimeError('Refusing incompatible resume checkpoint.')
        if ck.get('model_config') != model.config_dict():
            raise RuntimeError(
                f'Model config mismatch: checkpoint={ck.get("model_config")} current={model.config_dict()}'
            )
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_step = int(ck.get('step', 0))
        print(f'RESUMED_FROM={args.resume} STEP={start_step}', flush=True)

    current_long = start_step >= args.switch_iter
    current_t = args.long_frames if current_long else args.short_frames
    phase = 'long' if current_long else 'short'
    _, loader, audit = build_loader(roots, current_t, args.workers)
    print_audit(f'PHASE_{phase.upper()}', audit)
    train_iter = iter(loader)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step + 1, args.total_iterations + 1):
        should_long = step > args.switch_iter
        if should_long != current_long:
            del train_iter, loader
            torch.cuda.empty_cache()
            current_long = should_long
            current_t = args.long_frames
            phase = 'long'
            _, loader, audit = build_loader(roots, current_t, args.workers)
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
        if blur.shape != sharp.shape:
            raise RuntimeError(f'Blur/GT shape mismatch: {tuple(blur.shape)} vs {tuple(sharp.shape)}')

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
            source = batch.get('source')
            source_text = source[0] if isinstance(source, (list, tuple)) else str(source)
            _, t, _, h, w = blur.shape
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f'step={step}/{args.total_iterations} phase={phase} source={source_text} '
                f'T={t} H={h} W={w} loss={loss.item():.6f} lr={lr:.3e} '
                f'grad_norm={float(grad_norm):.4f} peak_gpu_gib={peak:.3f}', flush=True,
            )

        if step % args.save_every == 0 or step == args.total_iterations:
            ckpt = out / f'step_{step:07d}.pth'
            save_checkpoint(ckpt, model, optimizer, scheduler, step, args, phase)
            save_checkpoint(out / 'latest.pth', model, optimizer, scheduler, step, args, phase)
            print(f'SAVED={ckpt}', flush=True)


if __name__ == '__main__':
    main()
