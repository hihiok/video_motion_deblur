#!/usr/bin/env python3
"""Choose one/two GPUs only from matching, successful training benchmarks."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtf_t6.protocol import a100_world_size_config, sha256_file


def load_benchmark(run: Path, world_size: int, commit: str):
    config_path = run / 'runtime_config.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    report = json.loads((run / 'benchmark/benchmark.json').read_text())
    expected = {'status': 'BENCHMARK_PASS', 'world_size': world_size,
                'frames_per_update': 12, 'weights_discarded': True,
                'config_sha256': sha256_file(config_path), 'git_commit': commit}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Benchmark mismatch: {key} in {run}')
    seconds = report['mean_seconds_per_update']
    if not math.isfinite(seconds) or seconds <= 0 or report.get('measured_updates', 0) < 12:
        raise ValueError('Benchmark must contain at least 12 measured, finite updates')
    return cfg, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-run', required=True)
    parser.add_argument('--dual-run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    single_run = Path(args.single_run).resolve()
    single_cfg, single = load_benchmark(single_run, 1, commit)
    selected_run, chosen = single_run, single
    speedup = None
    reason = 'Single GPU is the only completed valid benchmark'
    if args.dual_run:
        dual_run = Path(args.dual_run).resolve()
        dual_cfg, dual = load_benchmark(dual_run, 2, commit)
        converted = a100_world_size_config(single_cfg, 2)
        for key in ('datasets', 'model', 'train', 'loss', 'validation'):
            if converted[key] != dual_cfg[key]:
                raise ValueError(f'Single/dual benchmark protocol mismatch: {key}')
        for key in ('pretrained_sha256', 'torch_version', 'cuda_version'):
            if single[key] != dual[key]:
                raise ValueError(f'Single/dual benchmark mismatch: {key}')
        speedup = single['mean_seconds_per_update'] / dual['mean_seconds_per_update']
        if speedup >= 1.10:
            selected_run, chosen = dual_run, dual
            reason = 'Dual GPU measured at least 10% faster per 12-frame update'
        else:
            reason = 'Dual GPU gain is below 10%; use single GPU'
    result = {'status': 'A100_MODE_SELECTED', 'world_size': chosen['world_size'],
              'gpus': chosen['cuda_visible_devices'], 'run': str(selected_run),
              'config': str(selected_run / 'runtime_config.yaml'),
              'speedup_dual_over_single': speedup, 'reason': reason,
              'mean_seconds_per_update': chosen['mean_seconds_per_update'],
              'estimated_training_hours_excluding_validation': chosen['mean_seconds_per_update'] * 90000 / 3600,
              'git_commit': commit}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
