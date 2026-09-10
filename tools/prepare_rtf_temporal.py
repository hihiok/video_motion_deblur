#!/usr/bin/env python3
"""Prepare immutable runtime config/manifest outside the checkout."""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtf_temporal.data import DOMAINS, make_manifest, sha256
from tools.prepare_rtf_t3_a100 import ALIASES, official_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/data/pub/z00919662/motion_deblur')
    p.add_argument('--dataset-base', default='/data/pub/z00919662/dataset')
    p.add_argument('--run', required=True)
    p.add_argument('--pretrained')
    for d in DOMAINS:
        p.add_argument(f'--{d}-root')
    args = p.parse_args()
    run = Path(args.run).resolve()
    repo = Path(__file__).resolve().parents[1]
    if run == repo or repo in run.parents:
        raise ValueError('Output must be outside source checkout')
    if (run / 'runtime_config.yaml').exists():
        raise FileExistsError('Already prepared; reuse existing config, do not overwrite it')
    roots = {}
    for d in DOMAINS:
        explicit = getattr(args, f'{d}_root')
        if explicit:
            roots[d] = str(Path(explicit).resolve())
        else:
            candidates = [Path(args.dataset_base) / name for name in ALIASES[d]]
            candidates = list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir()))
            if len(candidates) != 1:
                raise ValueError(f'{d}: need one dataset root, found {candidates}; use --{d}-root')
            roots[d] = str(candidates[0])
    pretrained = official_checkpoint(Path(args.root), args.pretrained)
    manifest = make_manifest(roots)
    config = yaml.safe_load((repo / 'configs/rtf_temporal_finetune.yaml').read_text())
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    config.update(output=str(run), manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
                  pretrained=str(pretrained), pretrained_sha256=sha256(pretrained))
    (run / 'runtime_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    summary = {s: {d: {'sequences': len([r for r in manifest[s] if r['domain'] == d]),
                           'frames': sum(len(r['blur']) for r in manifest[s] if r['domain'] == d),
                           'resolutions': sorted({f'{r["height"]}x{r["width"]}' for r in manifest[s] if r['domain'] == d})}
                        for d in DOMAINS} for s in ('train', 'val')}
    (run / 'data_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('TEMPORAL_DATA_READY', run / 'runtime_config.yaml')


if __name__ == '__main__':
    main()
