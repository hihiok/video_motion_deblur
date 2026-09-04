"""Native-full-frame mixed-dataset training for NanoVNR WaveShift-PAGF."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import train_nanovnr_nafnet_rgb_fullframe as common
from models.network_nanovnr_waveshift_pagf import NanoVNRWaveShiftPAGF


ARCHITECTURE = 'NanoVNRWaveShiftPAGF'
RECIPE_IDS = {
    'haar_pagf': 'nanovnr_haar_pagf_native_fullframe_bsd_v1',
    'waveshift': 'nanovnr_waveshift_pagf_native_fullframe_bsd_v1',
    'waveshift_edge': 'nanovnr_waveshift_pagf_edge_native_fullframe_bsd_v1',
}


def build_model(variant, grad_checkpoint=False):
    if variant == 'haar_pagf':
        return NanoVNRWaveShiftPAGF(
            gsts_blocks=0,
            gsts_radii=(),
            use_edge_aware=False,
            grad_checkpoint=grad_checkpoint,
        )
    if variant == 'waveshift':
        return NanoVNRWaveShiftPAGF(
            gsts_blocks=2,
            gsts_radii=(2, 4),
            use_edge_aware=False,
            grad_checkpoint=grad_checkpoint,
        )
    if variant == 'waveshift_edge':
        return NanoVNRWaveShiftPAGF(
            gsts_blocks=2,
            gsts_radii=(2, 4),
            use_edge_aware=True,
            grad_checkpoint=grad_checkpoint,
        )
    raise ValueError(f'Unknown variant: {variant}')


def save_checkpoint(path, model, optimizer, scheduler, step, args, phase):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'recipe_id': RECIPE_IDS[args.variant],
            'architecture': ARCHITECTURE,
            'variant': args.variant,
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


def representatives_by_family_resolution(dataset):
    """Return an actual sample for every family/native-resolution pair."""
    representatives = {}
    for component in dataset.datasets:
        seen_sequences = set()
        for sample_index, (sequence, _, pairs) in enumerate(component.samples):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            with Image.open(pairs[0][0]) as image:
                width, height = image.size
            key = (component.family, int(height), int(width))
            representatives.setdefault(key, (component, sample_index))
    return representatives


def run_model(model, blur, use_checkpoint):
    if not use_checkpoint:
        return model(blur)
    from torch.utils.checkpoint import checkpoint

    def forward_video(inp):
        output, _ = model(inp, prev_forward_feat=None)
        return output

    return checkpoint(forward_video, blur, use_reentrant=False), None


def run_preflight(args, roots, device):
    print('PREFLIGHT_ONLY=YES', flush=True)
    dataset, _, audit = common.build_loader(roots, args.long_frames, workers=0)
    common.print_audit('PREFLIGHT_LONG', audit)
    representatives = representatives_by_family_resolution(dataset)
    keys = [list(key) for key in representatives]
    print('PREFLIGHT_RESOLUTION_KEYS=' + json.dumps(keys), flush=True)
    failures = []

    for (family, height, width), (component, sample_index) in representatives.items():
        print(
            f'PREFLIGHT_BEGIN family={family} T={args.long_frames} '
            f'H={height} W={width} variant={args.variant}',
            flush=True,
        )
        torch.cuda.empty_cache()
        model = build_model(args.variant, grad_checkpoint=False).to(device).train()
        criterion = common.CharbonnierLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        sample = component[sample_index]
        blur = sample['blur'].unsqueeze(0).to(device, non_blocking=True)
        sharp = sample['sharp'].unsqueeze(0).to(device, non_blocking=True)
        torch.cuda.reset_peak_memory_stats(device)
        pred = None
        loss = None
        try:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred, _ = run_model(model, blur, args.grad_checkpoint)
                loss = criterion(pred, sharp)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f'Non-finite preflight gradient: {grad_norm}')
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f'PREFLIGHT_PASS family={family} T={args.long_frames} H={height} '
                f'W={width} loss={loss.item():.6f} peak_gpu_gib={peak:.3f}',
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            failures.append((family, height, width, 'forward_or_backward'))
            print(
                f'PREFLIGHT_OOM family={family} T={args.long_frames} '
                f'H={height} W={width} stage=forward_or_backward',
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
        print('PREFLIGHT_FAILED=' + json.dumps([list(item) for item in failures]), flush=True)
        raise RuntimeError(
            'Native full-frame T=30 OOM. Do not crop, resize, shorten T, or alter the model.'
        )
    print('PREFLIGHT_STATUS=PASS', flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gopro-root', required=True)
    parser.add_argument('--dvd-root', required=True)
    parser.add_argument('--bsd-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument(
        '--variant',
        choices=tuple(RECIPE_IDS),
        default='waveshift_edge',
        help='Primary experiment is waveshift_edge; other choices are controlled ablations.',
    )
    parser.add_argument('--short-frames', type=int, default=7)
    parser.add_argument('--long-frames', type=int, default=30)
    parser.add_argument('--switch-iter', type=int, default=50000)
    parser.add_argument('--total-iterations', type=int, default=150000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eta-min', type=float, default=1e-7)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    common.set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required.')
    device = torch.device('cuda')
    roots = common.roots_from_args(args)
    recipe_id = RECIPE_IDS[args.variant]
    probe = build_model(args.variant, grad_checkpoint=False)

    print('RECIPE_ID=' + recipe_id, flush=True)
    print('ARCHITECTURE=' + ARCHITECTURE, flush=True)
    print('VARIANT=' + args.variant, flush=True)
    print('MODEL_CONFIG=' + json.dumps(probe.config_dict()), flush=True)
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
    del probe

    if args.preflight_only:
        run_preflight(args, roots, device)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args.variant, grad_checkpoint=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iterations, eta_min=args.eta_min
    )
    criterion = common.CharbonnierLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location='cpu')
        if checkpoint_data.get('recipe_id') != recipe_id:
            raise RuntimeError(
                f'Refusing incompatible recipe: {checkpoint_data.get("recipe_id")}'
            )
        if checkpoint_data.get('architecture') != ARCHITECTURE:
            raise RuntimeError(
                f'Refusing incompatible architecture: '
                f'{checkpoint_data.get("architecture")}'
            )
        if checkpoint_data.get('model_config') != model.config_dict():
            raise RuntimeError('Checkpoint model_config does not match current model.')
        model.load_state_dict(checkpoint_data['model'], strict=True)
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        scheduler.load_state_dict(checkpoint_data['scheduler'])
        start_step = int(checkpoint_data.get('step', 0))
        print(f'RESUMED_FROM={args.resume} STEP={start_step}', flush=True)

    current_long = start_step > args.switch_iter
    current_t = args.long_frames if current_long else args.short_frames
    phase = 'long' if current_long else 'short'
    _, loader, audit = common.build_loader(roots, current_t, args.workers)
    common.print_audit(f'PHASE_{phase.upper()}', audit)
    train_iterator = iter(loader)
    model.train()

    for step in range(start_step + 1, args.total_iterations + 1):
        should_long = step > args.switch_iter
        if should_long != current_long:
            del train_iterator, loader
            torch.cuda.empty_cache()
            current_long = True
            current_t = args.long_frames
            phase = 'long'
            _, loader, audit = common.build_loader(roots, current_t, args.workers)
            print(
                f'SWITCH_PHASE_AT_STEP={step}: '
                f'T={args.short_frames} -> T={args.long_frames}',
                flush=True,
            )
            common.print_audit('PHASE_LONG', audit)
            train_iterator = iter(loader)

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(loader)
            batch = next(train_iterator)

        blur = batch['blur'].to(device, non_blocking=True)
        sharp = batch['sharp'].to(device, non_blocking=True)
        if blur.shape != sharp.shape:
            raise RuntimeError(
                f'Blur/GT shape mismatch: {tuple(blur.shape)} vs {tuple(sharp.shape)}'
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            prediction, _ = run_model(model, blur, args.grad_checkpoint)
            loss = criterion(prediction, sharp)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}: {loss.item()}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f'Non-finite gradient at step {step}: {grad_norm}')
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 100 == 0 or step in (args.switch_iter, args.switch_iter + 1):
            lr = optimizer.param_groups[0]['lr']
            source = batch.get('source')
            source_text = source[0] if isinstance(source, (list, tuple)) else str(source)
            _, frames, _, height, width = blur.shape
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f'step={step}/{args.total_iterations} phase={phase} '
                f'source={source_text} T={frames} H={height} W={width} '
                f'loss={loss.item():.6f} lr={lr:.3e} '
                f'grad_norm={float(grad_norm):.4f} peak_gpu_gib={peak:.3f}',
                flush=True,
            )

        if step % args.save_every == 0 or step == args.total_iterations:
            path = output_dir / f'step_{step:07d}.pth'
            save_checkpoint(path, model, optimizer, scheduler, step, args, phase)
            save_checkpoint(
                output_dir / 'latest.pth', model, optimizer, scheduler, step, args, phase
            )
            print(f'SAVED={path}', flush=True)


if __name__ == '__main__':
    main()
