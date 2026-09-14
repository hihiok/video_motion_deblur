#!/usr/bin/env python3
"""Read-only checkpoint evaluation: official full GoPro test and business MP4s."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from rtf_t6.datasets import read_rgb
from rtf_temporal.posttrain import (Models, business_video, checkpoint_audit,
    full_gopro, gopro_test, probe_video, verify_holdout_no_exact_test_overlap, write_json)

BASE = Path('/data/pub/z00919662/motion_deblur')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'runs/rtfocuser_causal_temporal_finetune_v1')
    parser.add_argument('--official', type=Path, default=BASE / 'weights/GoPro_RT_Focuser_Standard_256.pth')
    parser.add_argument('--gopro-root', type=Path, required=True, help='Dataset root containing official test/, or test/ itself')
    parser.add_argument('--input-dir', type=Path, default=BASE / 'input')
    parser.add_argument('--output', type=Path, required=True, help='NEW directory, outside original training run')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tasks', nargs='+', choices=['gopro', 'business'], default=['gopro', 'business'])
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.run = args.run.expanduser().resolve()
    if args.output.exists():
        parser.error('Output already exists: use a NEW output directory; never overwrite an evaluation')
    if args.output == args.run or args.run in args.output.parents:
        parser.error('Output must be outside the training run')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(20260914)
    result = {'status': 'EVALUATION_RUNNING', 'no_ablation': True, 'errors': [],
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'torch_version': torch.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        result['evaluation_git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        result['evaluation_git_commit'] = 'UNVERSIONED'
    summary_path = args.output / 'summary.json'

    def failed(task, exc):
        result['errors'].append({'task': task, 'type': type(exc).__name__, 'message': str(exc)})
        traceback.print_exc()
        write_json(summary_path, result)

    try:
        audit, manifest = checkpoint_audit(args.run, args.official, args.output)
        models = Models(audit, args.device)
        result['best_update'] = audit['best_update']
        result['checkpoints'] = audit['checkpoints']
        # Uses an untrained, zero-initialized wrapper ONLY to verify official source equivalence.
        result['implementation_identity'] = models.verify_wrapper(torch.rand(1, 3, 65, 97, device=models.device), audit)
        write_json(summary_path, result)
    except Exception as exc:
        failed('checkpoint_or_model_integrity', exc)
        result['status'] = 'EVALUATION_FAILED'
        write_json(summary_path, result)
        return 1

    if 'gopro' in args.tasks:
        try:
            available, absent = gopro_test(args.gopro_root)
            records = next(iter(available.values()))
            verify_holdout_no_exact_test_overlap(manifest, records)
            result['native_frame_identity'] = models.verify_wrapper(read_rgb(records[0]['blur'][0])[None].to(models.device), audit)
            result['gopro_absent_variants'] = absent
            result['gopro'] = full_gopro(models, available, args.output)
            write_json(summary_path, result)
        except Exception as exc:
            failed('gopro', exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if 'business' in args.tasks:
        try:
            if not args.input_dir.is_dir():
                raise FileNotFoundError(f'Business input directory missing: {args.input_dir}')
            sources = sorted(p.resolve() for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == '.mp4')
            if not sources:
                raise FileNotFoundError(f'No MP4 files directly inside {args.input_dir}')
            result['business_sources'] = [str(p) for p in sources]
            result['business'] = []
            for source in sources:
                try:
                    info = probe_video(source)
                    # Conservative uncompressed-equivalent estimate: decoded + 2 restored PNG sequences + preview.
                    estimated_bytes = int(info['frames'] * (9 * info['width'] * info['height'] + 1920 * 392 * 3) * 1.3)
                    free = shutil.disk_usage(args.output).free
                    if free < estimated_bytes:
                        raise OSError(f'Insufficient free space: {free/2**30:.1f} GiB; conservative requirement {estimated_bytes/2**30:.1f} GiB')
                    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', source.stem).strip('_') or 'video'
                    key = stem + '_' + hashlib.sha256(str(source).encode()).hexdigest()[:8]
                    destination = args.output / 'business' / key
                    report = business_video(models, source, destination)
                    result['business'].append({'output': str(destination), **report})
                    write_json(summary_path, result)
                except Exception as exc:
                    failed('business:' + str(source), exc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception as exc:
            failed('business_discovery', exc)
    result['status'] = 'EVALUATION_COMPLETE' if not result['errors'] else 'EVALUATION_PARTIAL_FAILED'
    write_json(summary_path, result)
    print(json.dumps({'status': result['status'], 'summary': str(summary_path), 'errors': result['errors']}, ensure_ascii=False), flush=True)
    return int(bool(result['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
