"""Native-full-frame mixed-dataset training for NanoVNR WaveShift-PAGF."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import train_nanovnr_nafnet_rgb_fullframe as common
from models.network_nanovnr_waveshift_pagf import NanoVNRWaveShiftPAGF
from amp_training import AMP_POLICY, make_scaler, restore_scaler, training_update


ARCHITECTURE = 'NanoVNRWaveShiftPAGF'
EXPECTED_BSD_ROOT = Path('/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms')
RECIPE_IDS = {
    'haar_pagf': 'nanovnr_haar_pagf_native_fullframe_t6_bsd3ms24ms_v3',
    'waveshift': 'nanovnr_waveshift_pagf_native_fullframe_t6_bsd3ms24ms_v3',
    'waveshift_edge': 'nanovnr_waveshift_pagf_edge_native_fullframe_t6_bsd3ms24ms_v3',
}
TRAIN_FRAMES = 6


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


def validate_bsd_root(root):
    """Lock this experiment to the single requested BSD 3ms-24ms tree."""
    actual = Path(root).expanduser().resolve()
    expected = EXPECTED_BSD_ROOT.resolve()
    if actual != expected:
        raise RuntimeError(
            f'BSD_ROOT_POLICY_VIOLATION: expected exactly {expected}, got {actual}'
        )
    for split in ('train', 'test'):
        split_root = actual / split
        if not split_root.is_dir():
            raise RuntimeError(f'Missing required BSD split: {split_root}')
    return str(actual)


def save_checkpoint(path, model, optimizer, scheduler, step, args, scaler, amp_stats):
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
            'scaler': scaler.state_dict(),
            'amp_policy': AMP_POLICY,
            'amp_stats': dict(amp_stats),
            'step': int(step),
            'phase': 'fixed_t6',
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
    dataset, _, audit = common.build_loader(roots, args.num_frames, workers=0)
    common.print_audit('PREFLIGHT_T6', audit)
    representatives = representatives_by_family_resolution(dataset)
    keys = [list(key) for key in representatives]
    print('PREFLIGHT_RESOLUTION_KEYS=' + json.dumps(keys), flush=True)
    failures = []

    for (family, height, width), (component, sample_index) in representatives.items():
        print(
            f'PREFLIGHT_BEGIN family={family} T={args.num_frames} '
            f'H={height} W={width} variant={args.variant}',
            flush=True,
        )
        torch.cuda.empty_cache()
        model = build_model(args.variant, grad_checkpoint=False).to(device).train()
        criterion = common.CharbonnierLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        scaler = make_scaler(args.amp)
        sample = component[sample_index]
        blur = sample['blur'].unsqueeze(0).to(device, non_blocking=True)
        sharp = sample['sharp'].unsqueeze(0).to(device, non_blocking=True)
        torch.cuda.reset_peak_memory_stats(device)
        def preflight_loss():
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred, _ = run_model(model, blur, args.grad_checkpoint)
                return criterion(pred, sharp)

        try:
            result = training_update(
                model, optimizer, scaler, preflight_loss,
                context={'stage': 'preflight', 'source': family,
                         'shape': list(blur.shape)},
            )
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f'PREFLIGHT_PASS family={family} T={args.num_frames} H={height} '
                f'W={width} loss={result["loss"]:.6f} peak_gpu_gib={peak:.3f}',
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            failures.append((family, height, width, 'forward_or_backward'))
            print(
                f'PREFLIGHT_OOM family={family} T={args.num_frames} '
                f'H={height} W={width} stage=forward_or_backward',
                flush=True,
            )
        finally:
            del model, criterion, optimizer, scaler, blur, sharp
            torch.cuda.empty_cache()

    if failures:
        print('PREFLIGHT_STATUS=FAIL', flush=True)
        print('PREFLIGHT_FAILED=' + json.dumps([list(item) for item in failures]), flush=True)
        raise RuntimeError(
            'Native full-frame T=6 OOM. Do not crop, resize, shorten T, or alter the model.'
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
    parser.add_argument('--num-frames', type=int, default=TRAIN_FRAMES)
    parser.add_argument('--total-iterations', type=int, default=150000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eta-min', type=float, default=1e-7)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--stop-after-step', type=int, default=None,
                        help='Save and exit after this successful step; cosine horizon unchanged.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_frames != TRAIN_FRAMES:
        raise RuntimeError(
            f'This deployment-matched recipe requires T={TRAIN_FRAMES}, '
            f'got T={args.num_frames}.'
        )
    end_step = args.stop_after_step or args.total_iterations
    if not 0 < end_step <= args.total_iterations:
        raise ValueError('--stop-after-step must be within the training horizon')
    common.set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required.')
    device = torch.device('cuda')
    args.bsd_root = validate_bsd_root(args.bsd_root)
    roots = common.roots_from_args(args)
    recipe_id = RECIPE_IDS[args.variant]
    probe = build_model(args.variant, grad_checkpoint=False)

    print('RECIPE_ID=' + recipe_id, flush=True)
    print('ARCHITECTURE=' + ARCHITECTURE, flush=True)
    print('VARIANT=' + args.variant, flush=True)
    print('MODEL_CONFIG=' + json.dumps(probe.config_dict()), flush=True)
    print('INPUT_CHANNELS=3 RGB', flush=True)
    print('FULL_FRAME=YES RANDOM_CROP=NO RESIZE=NO BATCH=1', flush=True)
    print('BSD_POLICY=STRICT_BSD_3MS24MS_DIRECT_TRAIN_TEST_ONLY', flush=True)
    print('BSD_ROOT=' + args.bsd_root, flush=True)
    print(
        f'TRAIN T={args.num_frames} fixed_for_all_steps total={args.total_iterations}',
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
    scaler = make_scaler(args.amp)

    amp_stats = {'overflow_attempts': 0, 'recovered_batches': 0}
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
        old_args = checkpoint_data.get('args', {})
        for key in ('total_iterations', 'lr', 'eta_min', 'num_frames', 'amp'):
            if key in old_args and old_args[key] != getattr(args, key):
                raise RuntimeError(f'CHECKPOINT_RECIPE_ARGUMENT_MISMATCH: {key}')
        model.load_state_dict(checkpoint_data['model'], strict=True)
        optimizer.load_state_dict(checkpoint_data['optimizer'])
        scheduler.load_state_dict(checkpoint_data['scheduler'])
        start_step = int(checkpoint_data.get('step', 0))
        scaler_status = restore_scaler(scaler, checkpoint_data)
        amp_stats.update(checkpoint_data.get('amp_stats', {}))
        print(f'{scaler_status} scale={scaler.get_scale()} AMP_POLICY={AMP_POLICY}', flush=True)
        print('DATA_ORDER_RESUME=RESEEDED_NOT_EXACT_SAMPLER_REPLAY', flush=True)
        if scheduler.last_epoch != start_step or scheduler.T_max != args.total_iterations:
            raise RuntimeError('CHECKPOINT_SCHEDULER_STEP_OR_HORIZON_MISMATCH')
        for name, value in model.state_dict().items():
            if not torch.isfinite(value).all().item():
                raise RuntimeError(f'NON_FINITE_CHECKPOINT_MODEL: {name}')
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                    raise RuntimeError(f'NON_FINITE_CHECKPOINT_OPTIMIZER: {name}')
        print(f'RESUMED_FROM={args.resume} STEP={start_step}', flush=True)

    if start_step >= end_step:
        raise RuntimeError('Resume checkpoint must precede requested end step')

    _, loader, audit = common.build_loader(roots, args.num_frames, args.workers)
    common.print_audit('TRAIN_T6', audit)
    train_iterator = iter(loader)
    model.train()

    for step in range(start_step + 1, end_step + 1):
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

        context = {'step': step, 'source': batch.get('source'),
                   'sequence': batch.get('seq'), 'shape': list(blur.shape)}

        def batch_loss():
            with torch.cuda.amp.autocast(enabled=args.amp):
                prediction, _ = run_model(model, blur, args.grad_checkpoint)
                return criterion(prediction, sharp)

        def emit_amp(record):
            if record['event'] == 'AMP_OVERFLOW':
                amp_stats['overflow_attempts'] += 1
            if record['event'] == 'AMP_RECOVERED':
                amp_stats['recovered_batches'] += 1
            text = json.dumps(record, allow_nan=False)
            print(text, flush=True)
            with (output_dir / 'amp_events.jsonl').open('a') as stream:
                stream.write(text + '\n')

        try:
            result = training_update(model, optimizer, scaler, batch_loss,
                                     context=context, emit=emit_amp)
        except RuntimeError as error:
            # Preserve the last good checkpoint; never label failed weights as valid.
            diagnostic = dict(context, error=str(error), last_successful_step=step - 1,
                              scaler=scaler.state_dict(), amp_stats=amp_stats)
            (output_dir / f'failure_step_{step:07d}.json').write_text(
                json.dumps(diagnostic, indent=2, allow_nan=False) + '\n'
            )
            raise
        scheduler.step()  # Exactly once per successful optimizer update.

        if step == 1 or step % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            source = batch.get('source')
            source_text = source[0] if isinstance(source, (list, tuple)) else str(source)
            _, frames, _, height, width = blur.shape
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f'step={step}/{args.total_iterations} phase=fixed_t6 '
                f'source={source_text} T={frames} H={height} W={width} '
                f'loss={result["loss"]:.6f} lr={lr:.3e} '
                f'grad_norm={result["grad_norm"]:.4f} scale={result["scale"]:.1f} '
                f'overflow_total={amp_stats["overflow_attempts"]} '
                f'peak_gpu_gib={peak:.3f}',
                flush=True,
            )

        if step % args.save_every == 0 or step == end_step:
            path = output_dir / f'step_{step:07d}.pth'
            save_checkpoint(path, model, optimizer, scheduler, step, args, scaler, amp_stats)
            save_checkpoint(
                output_dir / 'latest.pth', model, optimizer, scheduler, step, args, scaler, amp_stats
            )
            print(f'SAVED={path}', flush=True)


if __name__ == '__main__':
    main()
