#!/usr/bin/env python3
"""Resolve new-server paths into an external YAML without editing tracked code."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtf_t6.protocol import a100_world_size_config, check_fullframe, sha256_file


ALIASES = {
    'gopro': ('GoPro', 'gopro', 'GOPRO_Large'),
    'bsd': ('BSD', 'bsd'),
    'dvd': ('DeepVideoDeblurring_Dataset', 'DVD', 'dvd'),
}
OFFICIAL_SHA = '6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb'


def dataset_roots(base: Path, domain: str, overrides: list[str] | None) -> list[str]:
    candidates = [Path(p).expanduser() for p in overrides] if overrides else [base / name for name in ALIASES[domain]]
    existing = list(dict.fromkeys(str(p.resolve()) for p in candidates if p.is_dir()))
    if overrides and any(not p.is_dir() for p in candidates):
        raise FileNotFoundError(f'One or more explicit {domain} roots do not exist')
    if not existing:
        raise FileNotFoundError(f'No {domain} root under {base}; inspect directory names and use --{domain}-root PATH')
    return existing


def official_checkpoint(root: Path, explicit: str | None) -> Path:
    name = 'GoPro_RT_Focuser_Standard_256.pth'
    candidates = [Path(explicit).expanduser()] if explicit else [
        root / 'weights' / name,
        root / 'envs/RT-Focuser/Pretrained_Weights' / name,
        root / 'envs/rt_focuser_repo/Pretrained_Weights' / name,
        root / 'benchmark/weights/rt_focuser' / name,
    ]
    for path in candidates:
        if path.is_file():
            if sha256_file(path) != OFFICIAL_SHA:
                raise ValueError(f'Official checkpoint SHA256 mismatch: {path}')
            return path.resolve()
    raise FileNotFoundError(f'Missing official checkpoint. Transfer {name} to {root / "weights" / name}, or use --pretrained PATH')


def write_unchanged_or_new(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise FileExistsError(f'Refusing to overwrite a different runtime configuration: {path}; select a new --run')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/pub/z00919662/motion_deblur')
    parser.add_argument('--dataset-base', default='/data/pub/z00919662/dataset')
    parser.add_argument('--run', default=None)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--gpus', type=int, choices=(1, 2), default=1)
    for domain in ALIASES:
        parser.add_argument(f'--{domain}-root', action='append', default=None,
                            help='Repeat to supply separate train/validation root candidates')
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    base = Path(args.dataset_base).expanduser().resolve()
    run = Path(args.run).expanduser().resolve() if args.run else root / f'runs/rtfocuser_shift_dst_t3_gopro_bsd_dvd_a100_v4_{args.gpus}gpu'
    repo = Path(__file__).resolve().parents[1]
    if run == repo or repo in run.parents:
        raise ValueError('--run must be outside the source repository')
    config = yaml.safe_load((repo / 'configs/rtf_t3_gopro_bsd_dvd_a100.yaml').read_text())
    config = a100_world_size_config(config, args.gpus)
    for domain in ALIASES:
        config['datasets'][domain]['roots'] = dataset_roots(base, domain, getattr(args, f'{domain}_root'))
    pretrained = official_checkpoint(root, args.pretrained)
    config['output'] = str(run)
    config['pretrained'] = str(pretrained)
    check_fullframe(config)
    config_path = run / 'runtime_config.yaml'
    metadata = {'config': str(config_path), 'run': str(run), 'dataset_base': str(base), 'gpus': args.gpus,
                'pretrained': str(pretrained), 'pretrained_sha256': OFFICIAL_SHA,
                'dataset_roots': config['datasets']}
    write_unchanged_or_new(config_path, yaml.safe_dump(config, sort_keys=False))
    write_unchanged_or_new(run / 'a100_setup.json', json.dumps(metadata, indent=2) + '\n')
    (run / 'audit').mkdir(exist_ok=True)
    print(json.dumps(metadata, indent=2))
    print('A100_SETUP_PASS')


if __name__ == '__main__':
    main()
