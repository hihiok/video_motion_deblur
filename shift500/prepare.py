"""Resolve server inputs once; write configs and audit outside tracked checkout."""
import argparse
import json
from pathlib import Path
from .data import ALIASES, DOMAINS, make_manifest, sha256
from .model import ShiftModel, load_teacher
from .profile import profile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-base',default='/data/pub/z00919662/dataset')
    p.add_argument('--run',required=True);p.add_argument('--upstream',required=True)
    p.add_argument('--teacher-checkpoint',required=True)
    for d in DOMAINS:p.add_argument(f'--{d}-root')
    a=p.parse_args();run=Path(a.run).resolve();repo=Path(__file__).resolve().parents[1]
    if run==repo or repo in run.parents:raise ValueError('Run must be outside source checkout')
    if (run/'config.json').exists():raise FileExistsError('Reuse existing config; do not overwrite')
    roots={}
    for d in DOMAINS:
        override=getattr(a,f'{d}_root')
        candidates=[Path(override)] if override else [Path(a.dataset_base)/v for v in ALIASES[d]]
        candidates=list(dict.fromkeys(str(p.resolve()) for p in candidates if p.is_dir()))
        if len(candidates)!=1:raise ValueError(f'Need one {d} root, found {candidates}; use --{d}-root')
        roots[d]=candidates[0]
    teacher=ShiftModel(a.upstream,'teacher');load_teacher(teacher,a.teacher_checkpoint)
    reports={v:profile(a.upstream,v) for v in ('teacher','quality','compact')}
    for v in ('quality','compact'):
        if reports[v]['arithmetic_GFLOPs_per_output']>500:raise ValueError(f'{v}: budget exceeded')
    manifest=make_manifest(roots)
    run.mkdir(parents=True,exist_ok=True)
    mp=run/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    for v,r in reports.items():(run/f'profile_{v}.json').write_text(json.dumps(r,indent=2)+'\n')
    config=dict(upstream=str(Path(a.upstream).resolve()),teacher_checkpoint=str(Path(a.teacher_checkpoint).resolve()),
                teacher_sha256=sha256(a.teacher_checkpoint),manifest=str(mp),manifest_sha256=sha256(mp),
                output=str(run),seed=20260911,frames=16,clips_per_update=4,total_updates=180000,
                lr=0.0002,min_lr=0.000002,warmup_updates=2000,validate_every=5000,save_every=1000,
                workers=2,validation_windows_per_sequence=3,
                training_spatial_mode='native_full_frame_no_crop_no_resize',
                domain_sampling={'gopro':.5,'dvd':.25,'bsd':.25})
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    summary={s:{d:{'sequences':sum(r['domain']==d for r in manifest[s]),
                     'frames':sum(len(r['blur']) for r in manifest[s] if r['domain']==d),
                     'resolutions':sorted({f'{r["height"]}x{r["width"]}' for r in manifest[s] if r['domain']==d})}
                for d in DOMAINS} for s in ('train','val','test')}
    (run/'data_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    (run/'split_audit.json').write_text(json.dumps(manifest['split_audit'],indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('PREPARED',run/'config.json')

if __name__=='__main__':main()
