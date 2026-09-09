#!/usr/bin/env python3
"""Reject missing/stale memory preflights before launching a full-frame run."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtf_t6.protocol import check_fullframe, clip_length, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    check_fullframe(config)
    report = json.loads(Path(args.report).read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=Path(__file__).resolve().parents[1], text=True).strip()
    expected = {
        'status': 'FULLFRAME_PREFLIGHT_PASS',
        'config_sha256': sha256_file(args.config),
        'pretrained_sha256': sha256_file(args.pretrained),
        'git_commit': commit, 'clip_length': clip_length(config),
        'gradient_accumulation': int(config['train']['gradient_accumulation']),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'ema_on_gpu': True, 'empty_cache_between_steps': False,
        'crop_applied': False, 'resize_applied': False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Preflight mismatch: {key}; rerun the preflight')
    rounds = int(report.get('rounds', 0))
    if rounds < 2 or set(report.get('domains', {})) != {'bsd', 'dvd', 'gopro'}:
        raise ValueError('Preflight must cover all three domains for at least two rounds')
    if report.get('optimizer_updates') != 3 * rounds:
        raise ValueError('Preflight has missing optimizer updates')
    for domain, steps in report['domains'].items():
        if len(steps) != rounds or not all(s.get('optimizer_step_pass') and s.get('ema_validation_pass') for s in steps):
            raise ValueError(f'Incomplete preflight for {domain}')
    print('FULLFRAME_GATE_VERIFIED')


if __name__ == '__main__':
    main()
