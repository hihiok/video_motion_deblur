"""Create a new crop run from the already-audited full-frame config; keep old run."""
import argparse
import copy
import json
from pathlib import Path
from .data import sha256
from .profile import profile
from .protocol import training_settings


def migrate(source_config, output):
    source = Path(source_config).resolve()
    run = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    c = json.loads(source.read_text())
    old_output = Path(c['output']).resolve()
    if run == repo or repo in run.parents or run == old_output or old_output in run.parents:
        raise ValueError('Use a separate sibling run outside source checkout and old run')
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('Destination must be new/empty; reuse completed new config rather than migrate again')
    if sha256(c['manifest']) != c['manifest_sha256']:
        raise ValueError('Source manifest changed')
    if sha256(c['teacher_checkpoint']) != c['teacher_sha256']:
        raise ValueError('Source teacher changed')
    old_manifest = json.loads(Path(c['manifest']).read_text())
    if old_manifest.get('split_audit', {}).get('gt_file_sha256_cross_split_check') != 'passed':
        raise ValueError('Need the successful audited manifest from the GoPro split fix')
    for r in old_manifest['train']:
        if len(r['blur']) < 13 or min(r['height'], r['width']) < 256:
            raise ValueError(f'Clip too short or smaller than crop: {r["name"]}')
    profiles = {v:profile(c['upstream'],v) for v in ('teacher','quality','compact')}
    for v in ('quality','compact'):
        if profiles[v]['arithmetic_GFLOPs_per_output'] > 500:
            raise ValueError(f'Deployment budget exceeded: {v}')
    manifest = copy.deepcopy(old_manifest)
    manifest.update(version=3, frames=13, parent_manifest_sha256=c['manifest_sha256'])
    mp = run / 'manifest.json'
    c.update(training_settings())
    c.update(output=str(run), manifest=str(mp), migrated_from_config=str(source),
             migrated_from_config_sha256=sha256(source))
    run.mkdir(parents=True,exist_ok=True)
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    c['manifest_sha256'] = sha256(mp)
    for v, report in profiles.items():
        (run/f'profile_{v}.json').write_text(json.dumps(report,indent=2)+'\n')
    (run/'split_audit.json').write_text(json.dumps(manifest['split_audit'],indent=2)+'\n')
    summary = {s:{d:{'sequences':sum(r['domain']==d for r in manifest[s]),
                      'frames':sum(len(r['blur']) for r in manifest[s] if r['domain']==d)}
                   for d in ('gopro','dvd','bsd')} for s in ('train','val','test')}
    (run/'data_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Config is written last so it marks a completed preparation.
    (run/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps({'config':str(run/'config.json'),'training':training_settings(),
                      'source_splits_preserved':True,'teacher_sha256':c['teacher_sha256']},indent=2))
    return c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--from-config',required=True)
    p.add_argument('--run',required=True)
    a=p.parse_args()
    migrate(a.from_config,a.run)

if __name__=='__main__':main()
