# Legacy entrypoint retained for compatibility.
#
# For the current experiment use:
#   train_nanovnr_nafnet_rgb_fullframe_bsd_splits.py
# which sets recipe id nanovnr_nafnet_rgb_native_fullframe_mix_bsd_train_test_v2.
# Dataset selection itself is enforced in data/mixed_deblur.py, where family BSD
# can only read the direct <BSD_ROOT>/train or <BSD_ROOT>/test split.

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.mixed_deblur import build_mixed_dataset
from models.network_nanovnr_nafnet_rgb import NanoVNRNAFNetRGB


RECIPE_ID = 'nanovnr_nafnet_rgb_native_fullframe_mix_v1'
ARCHITECTURE = 'NanoVNRNAFNetRGB'


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        d = pred - target
        return torch.sqrt(d * d + self.eps).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def roots_from_args(args):
    return {'GoPro': args.gopro_root, 'DVD': args.dvd_root, 'BSD': args.bsd_root}


def build_loader(roots, num_frames, workers):
    ds, sampler, audit = build_mixed_dataset(
        roots,
        split='train',
        num_frames=num_frames,
        patch_size=None,
        train=True,
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn if workers > 0 else None,
    )
    return ds, dl, audit


def print_audit(tag, audit):
    totals = {}
    print(f'[{tag}] DATASET_AUDIT_BEGIN', flush=True)
    for row in audit:
        print(
            f"family={row['family']} blur={row['blur_root']} gt={row['gt_root']} "
            f"sequences={row['sequences']} windows={row['windows']} T={row['frames_per_window']} "
            f"strict_root_split={row.get('strict_root_split', False)}",
            flush=True,
        )
        totals[row['family']] = totals.get(row['family'], 0) + row['windows']
    print(f'[{tag}] FAMILY_WINDOWS={totals}', flush=True)
    print(f'[{tag}] DATASET_AUDIT_END', flush=True)


def save_checkpoint(path, model, optimizer, scheduler, step, args, phase):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'recipe_id': RECIPE_ID,
            'architecture': ARCHITECTURE,
            'model_config': model.config_dict(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': int(step),
            'phase': phase,
            'args': vars(args),
        },
        path,
    )


def _representatives_by_family_resolution(dataset):
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


def _run_model(model, blur, grad_checkpoint):
    if not grad_checkpoint:
        return model(blur)

    # Model-level temporal checkpointing without changing architecture/math.
    # The wrapped function returns only the video output; prev_forward_feat is
    # intentionally None during training/preflight, matching ordinary training.
    from torch.utils.checkpoint import checkpoint

    def fn(inp):
        out, _ = model(inp, prev_forward_feat=None)
        return out

    return checkpoint(fn, blur, use_reentrant=False), None


def run_preflight(args, roots, device):
    print('PREFLIGHT_ONLY=YES', flush=True)
    ds, _, audit = build_loader(roots, args.long_frames, workers=0)
    print_audit('PREFLIGHT_T30', audit)
    reps = _representatives_by_family_resolution(ds)
    print('PREFLIGHT_RESOLUTION_KEYS=' + json.dumps([list(k) for k in reps]), flush=True)

    failures = []
    for (family, h, w), component in reps.items():
        print(
            f'PREFLIGHT_BEGIN family={family} T={args.long_frames} H={h} W={w}',
            flush=True,
        )
        torch.cuda.empty_cache()
        model = NanoVNRNAFNetRGB().to(device).train()
        criterion = CharbonnierLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        sample = component[0]
        blur = sample['blur'].unsqueeze(0).to(device, non_blocking=True)
        sharp = sample['sharp'].unsqueeze(0).to(device, non_blocking=True)
        torch.cuda.reset_peak_memory_stats(device)
        pred = None
        loss = None
        try:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred, _ = _run_model(model, blur, args.grad_checkpoint)
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
                f'loss={loss.item():.6f} peak_gpu_gib={peak:.3f}',
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            failures.append((family, h, w))
            print(
                f'PREFLIGHT_OOM family={family} T={args.long_frames} H={h} W={w}',
                flush=True,
            )
        finally:
            del model, criterion, optimizer, scaler, blur, sharp
            if pred is not None:
                del pred
            if loss is not None:
                del loss
            torch.cuda.empty_cache()

    if failures:
        print('PREFLIGHT_STATUS=FAIL', flush=True)
        print('PREFLIGHT_FAILED=' + json.dumps([list(x) for x in failures]), flush=True)
        raise RuntimeError(
            'Native full-frame T=30 OOM. Do not crop/resize/change T or alter model.'
        )
    print('PREFLIGHT_STATUS=PASS', flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True)
    ap.add_argument('--dvd-root', required=True)
    ap.add_argument('--bsd-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--resume', default=None)

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
    roots = roots_from_args(args)

    print('RECIPE_ID=' + RECIPE_ID, flush=True)
    print('ARCHITECTURE=' + ARCHITECTURE, flush=True)
    print('MODEL_CONFIG=' + json.dumps(NanoVNRNAFNetRGB().config_dict()), flush=True)
    print('INPUT_CHANNELS=3 RGB', flush=True)
    print('FULL_FRAME=YES RANDOM_CROP=NO RESIZE=NO BATCH=1', flush=True)
    print('BSD_POLICY=STRICT_DIRECT_TRAIN_TEST_ONLY', flush=True)
    print(
        f'TRAIN shortT={args.short_frames} longT={args.long_frames} '
        f'switch={args.switch_iter} total={args.total_iterations}',
        flush=True,
    )
    print(
        f'LOSS=CharbonnierOnly OPT=Adam betas=(0.9,0.99) '
        f'LR={args.lr}->{args.eta_min} grad_clip=0.5 '
        f'AMP={args.amp} GRAD_CHECKPOINT={args.grad_checkpoint}',
        flush=True,
    )

    if args.preflight_only:
        run_preflight(args, roots, device)
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = NanoVNRNAFNetRGB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iterations, eta_min=args.eta_min
    )
    criterion = CharbonnierLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        if ck.get('recipe_id') != RECIPE_ID or ck.get('architecture') != ARCHITECTURE:
            raise RuntimeError(
                f'Refusing incompatible resume: recipe={ck.get("recipe_id")} '
                f'architecture={ck.get("architecture")}'
            )
        if ck.get('model_config') != model.config_dict():
            raise RuntimeError(
                f'Model config mismatch: checkpoint={ck.get("model_config")} '
                f'current={model.config_dict()}'
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
    for step in range(start_step + 1, args.total_iterations + 1):
        should_long = step > args.switch_iter
        if should_long != current_long:
            del train_iter, loader
            torch.cuda.empty_cache()
            current_long = should_long
            current_t = args.long_frames
            phase = 'long'
            _, loader, audit = build_loader(roots, current_t, args.workers)
            print(
                f'SWITCH_PHASE_AT_STEP={step}: T={args.short_frames} -> T={args.long_frames}',
                flush=True,
            )
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
            raise RuntimeError(
                f'Blur/GT shape mismatch: {tuple(blur.shape)} vs {tuple(sharp.shape)}'
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred, _ = _run_model(model, blur, args.grad_checkpoint)
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
                f'grad_norm={float(grad_norm):.4f} peak_gpu_gib={peak:.3f}',
                flush=True,
            )

        if step % args.save_every == 0 or step == args.total_iterations:
            ckpt = out / f'step_{step:07d}.pth'
            save_checkpoint(ckpt, model, optimizer, scheduler, step, args, phase)
            save_checkpoint(out / 'latest.pth', model, optimizer, scheduler, step, args, phase)
            print(f'SAVED={ckpt}', flush=True)


if __name__ == '__main__':
    main()
