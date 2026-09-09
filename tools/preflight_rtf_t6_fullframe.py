#!/usr/bin/env python3
"""Native T3/T6 CUDA preflight with accumulation, EMA and shape transitions."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtf_t6.checkpoint import load_rtfocuser_pretrained
from rtf_t6.datasets import build_domain_sequences, load_clip
from rtf_t6.losses import VideoDeblurLoss
from rtf_t6.model import RTFocuserT6
from rtf_t6.protocol import check_fullframe, clip_length, sha256_file
from tools.train_rtf_t6 import EMA


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rounds', type=int, default=2)
    return parser.parse_args()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def image_area(sequence) -> int:
    with Image.open(sequence.blur[0]) as image:
        width, height = image.size
    return width * height


# Preserve the helper name used by existing callers.
check_config = check_fullframe


def main() -> int:
    args = arguments()
    output = Path(args.output).expanduser().resolve()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    check_config(config)
    if args.rounds < 2:
        raise ValueError('At least two rounds are required to test shape transitions')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the full-frame memory preflight')
    torch.cuda.set_device(device)
    train_cfg = config['train']
    length = clip_length(config)
    accumulation = int(train_cfg['gradient_accumulation'])
    domains, roots = build_domain_sequences(config['datasets'], 'train', length)
    if set(domains) != {'bsd', 'dvd', 'gopro'}:
        raise ValueError('Preflight requires all three domains: bsd, dvd, gopro')
    model = RTFocuserT6(**config.get('model', {}))
    initialization = load_rtfocuser_pretrained(model, args.pretrained)
    model = model.to(device).train()
    # Formal training keeps EMA weights on the GPU too.
    ema = EMA(model, float(train_cfg.get('ema_decay', 0.999)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg['lr']),
        betas=tuple(train_cfg.get('betas', (0.9, 0.999))),
        weight_decay=float(train_cfg.get('weight_decay', 1e-4)))
    amp = bool(train_cfg.get('amp', True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = VideoDeblurLoss(**config.get('loss', {})).to(device)
    loss_cfg = config.get('loss', {})
    loss_iteration = int(loss_cfg.get('temporal_start_iter', 0)) + int(loss_cfg.get('temporal_ramp_iters', 0)) + 1
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    report = {
        'status': 'IN_PROGRESS', 'config': str(Path(args.config).resolve()),
        'config_sha256': sha256_file(args.config), 'git_commit': commit,
        'pretrained': str(Path(args.pretrained).resolve()),
        'pretrained_sha256': sha256_file(args.pretrained),
        'device': str(device), 'gpu': torch.cuda.get_device_name(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'roots': roots, 'target_coverage': initialization['target_coverage'],
        'clip_length': length, 'gradient_accumulation': accumulation,
        'rounds': args.rounds, 'spatial_mode': 'full_frame',
        'crop_applied': False, 'resize_applied': False,
        'activation_checkpointing': True, 'ema_on_gpu': True,
        'empty_cache_between_steps': False, 'domains': {}, 'optimizer_updates': 0,
    }
    write_report(output, report)
    try:
        for round_index in range(args.rounds):
            # Second round transposes native H/W, matching 90-degree augmentation.
            rotated = bool(round_index % 2)
            for domain, sequences in sorted(domains.items()):
                sequence = max(sequences, key=image_area)
                start = max((sequence.length - length) // 2, 0)
                indices = list(range(start, start + length))
                with Image.open(sequence.blur[start]) as image:
                    width, height = image.size
                shape = (length, 3, width, height) if rotated else (length, 3, height, width)
                report['active_step'] = {'round': round_index + 1, 'domain': domain,
                    'native_shape': [length, 3, height, width], 'training_shape': list(shape)}
                write_report(output, report)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                samples = []
                for micro in range(accumulation):
                    report['active_step']['microbatch'] = micro + 1
                    blur, gt = load_clip(sequence, indices, crop_size=0,
                        rng=random.Random(20260909 + micro), augment=False)
                    if rotated:
                        blur, gt = torch.rot90(blur, 1, (-2, -1)), torch.rot90(gt, 1, (-2, -1))
                    if tuple(blur.shape) != shape or tuple(gt.shape) != shape:
                        raise RuntimeError(f'Native-shape violation for {domain}/{sequence.name}')
                    blur = blur.contiguous().unsqueeze(0).to(device)
                    gt = gt.contiguous().unsqueeze(0).to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                        prediction = model(blur)
                        loss, components = criterion(prediction, gt, loss_iteration)
                    if prediction.shape != gt.shape or not bool(torch.isfinite(prediction).all()):
                        raise FloatingPointError(f'Invalid output for {domain}')
                    if not all(bool(torch.isfinite(value).all()) for value in components.values()):
                        raise FloatingPointError(f'Non-finite loss for {domain}')
                    scaler.scale(loss / accumulation).backward()
                    samples.append({key: float(value.detach()) for key, value in components.items()})
                    del blur, gt, prediction, loss, components
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                    float(train_cfg.get('clip_grad', 1.0)), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                report['optimizer_updates'] += 1
                torch.cuda.synchronize(device)
                train_seconds = time.perf_counter() - started
                train_peak = torch.cuda.max_memory_allocated(device) / 2**30
                # Exercise validation with the persistent optimizer and EMA resident.
                torch.cuda.reset_peak_memory_stats(device)
                blur, gt = load_clip(sequence, indices, 0, random.Random(0), False)
                blur, gt = blur.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                    prediction = ema.model(blur)
                if prediction.shape != gt.shape or not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError(f'Invalid EMA validation for {domain}')
                validation_peak = torch.cuda.max_memory_allocated(device) / 2**30
                del blur, gt, prediction
                torch.cuda.synchronize(device)
                step = {'round': round_index + 1, 'sequence': sequence.name, 'start': start,
                    'native_shape': [length, 3, height, width], 'training_shape': list(shape),
                    'output_shape': list(shape), 'rotation_degrees': 90 if rotated else 0,
                    'loss_components_by_microbatch': samples, 'gradient_norm': float(grad_norm),
                    'optimizer_step_pass': True, 'ema_validation_pass': True,
                    'train_seconds_per_update': train_seconds,
                    'train_seconds_per_microbatch': train_seconds / accumulation,
                    'elapsed_seconds': time.perf_counter() - started,
                    'train_peak_allocated_gib': train_peak,
                    'validation_peak_allocated_gib': validation_peak,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30}
                report['domains'].setdefault(domain, []).append(step)
                write_report(output, report)
                print(json.dumps({'event': 'preflight_step', 'domain': domain, **step}), flush=True)
    except Exception as error:
        report['status'] = 'FULLFRAME_PREFLIGHT_OOM' if isinstance(error, torch.cuda.OutOfMemoryError) else 'FULLFRAME_PREFLIGHT_FAILED'
        report['error'] = f'{type(error).__name__}: {error}'
        write_report(output, report)
        print(json.dumps(report, indent=2), flush=True)
        return 3
    report.pop('active_step', None)
    report['status'] = 'FULLFRAME_PREFLIGHT_PASS'
    write_report(output, report)
    print('FULLFRAME_PREFLIGHT_PASS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
