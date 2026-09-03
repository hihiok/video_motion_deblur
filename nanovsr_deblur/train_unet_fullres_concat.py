import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.mixed_deblur import build_mixed_dataset
from models.nanovsr_unet_fullres_concat_deblur import NanoVSRFullResConcatUNetDeblur


RECIPE_ID = 'nanovsr_unet_fullres_concat_charbonnier_mix_v3'
ARCH = 'NanoVSRFullResConcatUNetDeblur'


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        d = x - y
        return torch.sqrt(d * d + self.eps).mean()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(args, device):
    return NanoVSRFullResConcatUNetDeblur(
        base_channels=args.base_channels,
        mid_channels=args.mid_channels,
        bottleneck_channels=args.bottleneck_channels,
        encoder_blocks=args.encoder_blocks,
        state_fusion_blocks=args.state_fusion_blocks,
        fullres_blocks=args.fullres_blocks,
        mid_blocks=args.mid_blocks,
        bottleneck_blocks=args.bottleneck_blocks,
        decoder_channels=args.decoder_channels,
        decoder_blocks=args.decoder_blocks,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)


def build_loader(roots, t, workers):
    ds, sampler, audit = build_mixed_dataset(
        roots, split='train', num_frames=t, patch_size=None, train=True
    )
    dl = DataLoader(
        ds, batch_size=1, sampler=sampler, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
        persistent_workers=workers > 0,
    )
    return ds, dl, audit


def print_audit(tag, audit):
    totals = {}
    print(f'[{tag}]', flush=True)
    for r in audit:
        totals[r['family']] = totals.get(r['family'], 0) + r['windows']
        print(f"family={r['family']} blur={r['blur_root']} gt={r['gt_root']} seq={r['sequences']} windows={r['windows']} T={r['frames_per_window']}", flush=True)
    print(f'FAMILY_WINDOWS={totals}', flush=True)


def save_ckpt(path, model, optimizer, scheduler, step, phase, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'recipe_id': RECIPE_ID,
        'architecture': ARCH,
        'model_config': model.config_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': int(step),
        'phase': phase,
        'args': vars(args),
    }, path)


def representative_components(ds):
    reps = {}
    for comp in ds.datasets:
        if not comp.samples:
            continue
        _, _, pairs = comp.samples[0]
        with Image.open(pairs[0][0]) as im:
            w, h = im.size
        reps.setdefault((comp.family, h, w), comp)
    return reps


def preflight(args, roots, device):
    ds, _, audit = build_loader(roots, args.long_frames, 0)
    print_audit('PREFLIGHT_T30', audit)
    failed = []
    for (family, h, w), comp in representative_components(ds).items():
        print(f'PREFLIGHT_BEGIN family={family} T={args.long_frames} H={h} W={w}', flush=True)
        torch.cuda.empty_cache()
        model = build_model(args, device).train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        criterion = CharbonnierLoss().to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        sample = comp[0]
        x = sample['blur'].unsqueeze(0).to(device)
        y = sample['sharp'].unsqueeze(0).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update(); torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f'PREFLIGHT_PASS family={family} H={h} W={w} peak_gpu_gib={peak:.3f} loss={loss.item():.6f}', flush=True)
        except torch.cuda.OutOfMemoryError:
            failed.append((family, h, w))
            print(f'PREFLIGHT_OOM family={family} H={h} W={w}', flush=True)
        finally:
            del model, opt, criterion, scaler, x, y
            torch.cuda.empty_cache()
    if failed:
        raise RuntimeError('Full-frame T30 concat-recurrent U-Net OOM: ' + json.dumps(failed))
    print('PREFLIGHT_STATUS=PASS', flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gopro-root', required=True); ap.add_argument('--dvd-root', required=True); ap.add_argument('--bsd-root', required=True)
    ap.add_argument('--output-dir', required=True); ap.add_argument('--resume', default=None)
    ap.add_argument('--base-channels', type=int, default=48); ap.add_argument('--mid-channels', type=int, default=64); ap.add_argument('--bottleneck-channels', type=int, default=96)
    ap.add_argument('--encoder-blocks', type=int, default=2); ap.add_argument('--state-fusion-blocks', type=int, default=1)
    ap.add_argument('--fullres-blocks', type=int, default=2); ap.add_argument('--mid-blocks', type=int, default=2); ap.add_argument('--bottleneck-blocks', type=int, default=4)
    ap.add_argument('--decoder-channels', type=int, default=64); ap.add_argument('--decoder-blocks', type=int, default=2)
    ap.add_argument('--short-frames', type=int, default=7); ap.add_argument('--long-frames', type=int, default=30)
    ap.add_argument('--switch-iter', type=int, default=50000); ap.add_argument('--total-iterations', type=int, default=150000)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--eta-min', type=float, default=1e-7)
    ap.add_argument('--save-every', type=int, default=5000); ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--amp', action='store_true'); ap.add_argument('--grad-checkpoint', action='store_true'); ap.add_argument('--preflight-only', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args(); seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device = torch.device('cuda')
    roots = {'GoPro': args.gopro_root, 'DVD': args.dvd_root, 'BSD': args.bsd_root}
    print(f'RECIPE_ID={RECIPE_ID}', flush=True)
    print('ARCHITECTURE=' + ARCH, flush=True)
    print('FULL_FRAME=YES RANDOM_CROP=NO RESIZE=NO BATCH=1 RECURRENT_STATE=FULL_RES CONCAT_STATE=YES', flush=True)
    print('LOSS=CharbonnierOnly', flush=True)
    if args.preflight_only:
        preflight(args, roots, device); return

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iterations, eta_min=args.eta_min)
    criterion = CharbonnierLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu')
        if ck.get('recipe_id') != RECIPE_ID or ck.get('architecture') != ARCH:
            raise RuntimeError('Refusing incompatible checkpoint resume')
        if ck.get('model_config') != model.config_dict():
            raise RuntimeError('Checkpoint model_config mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); start = int(ck['step'])

    is_long = start >= args.switch_iter
    t = args.long_frames if is_long else args.short_frames
    phase = 'long' if is_long else 'short'
    _, loader, audit = build_loader(roots, t, args.workers); print_audit('PHASE_' + phase.upper(), audit); it = iter(loader)
    model.train()
    for step in range(start + 1, args.total_iterations + 1):
        want_long = step > args.switch_iter
        if want_long != is_long:
            del it, loader; torch.cuda.empty_cache(); is_long = True; t = args.long_frames; phase = 'long'
            _, loader, audit = build_loader(roots, t, args.workers); print_audit('PHASE_LONG', audit); it = iter(loader)
            print(f'SWITCH_PHASE_AT_STEP={step}: T={args.short_frames}->{args.long_frames}', flush=True)
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        x = batch['blur'].to(device, non_blocking=True); y = batch['sharp'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(x); loss = criterion(pred, y)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if step == 1 or step % 100 == 0 or step in (args.switch_iter, args.switch_iter + 1):
            _, tt, _, h, w = x.shape
            print(f'step={step}/{args.total_iterations} phase={phase} T={tt} H={h} W={w} loss={loss.item():.6f} lr={optimizer.param_groups[0]["lr"]:.3e} grad_norm={float(grad):.4f}', flush=True)
        if step % args.save_every == 0 or step == args.total_iterations:
            save_ckpt(out / f'step_{step:07d}.pth', model, optimizer, scheduler, step, phase, args)
            save_ckpt(out / 'latest.pth', model, optimizer, scheduler, step, phase, args)


if __name__ == '__main__': main()
