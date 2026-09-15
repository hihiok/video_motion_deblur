"""Derive a new run from existing audited data; never touch active run files."""
import argparse
import json
from pathlib import Path
from shift500.data import sha256, DOMAINS
from shift500.protocol import official_training_partition
from shift500.profile import profile as teacher_profile
from .profile import profile
from .protocol import training_settings


def prepare(source, output, upstream=None, continue_checkpoint=None):
    source=Path(source).resolve();run=Path(output).resolve()
    c=json.loads(source.read_text());source_config=dict(c);old=Path(c['output']).resolve()
    repo=Path(__file__).resolve().parents[1]
    if run==old or old in run.parents or run==repo or repo in run.parents:
        raise ValueError('Use a separate run outside old run and checkout')
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('New run must be empty; reuse an already complete config without preparing again')
    if sha256(c['manifest'])!=c['manifest_sha256'] or sha256(c['teacher_checkpoint'])!=c['teacher_sha256']:
        raise ValueError('Source manifest/teacher provenance mismatch')
    manifest=official_training_partition(json.loads(Path(c['manifest']).read_text()))
    if upstream:c['upstream']=str(Path(upstream).resolve())
    for r in manifest['train']:
        if len(r['blur'])<13 or min(r['height'],r['width'])<256:
            raise ValueError(f'Invalid training sequence {r["name"]}')
    counts={s:{d:dict(sequences=sum(r['domain']==d for r in manifest[s]),
                     frames=sum(len(r['blur']) for r in manifest[s] if r['domain']==d))
                 for d in DOMAINS} for s in ('train','val','test')}
    if counts['test']['gopro']!={'sequences':11,'frames':1111}:
        raise ValueError(f'GoPro must contain all 11 test sequences / 1111 frames: {counts["test"]["gopro"]}')
    if counts['train']['gopro']['sequences']!=22:
        raise ValueError('Need all 22 GoPro official train chunks')
    for d in DOMAINS:
        counts['train'][d]['eligible_windows']=sum(min(100,len(r['blur']))-12 for r in manifest['train'] if r['domain']==d)
    counts['test']['dvd']['complete_official_10_sequences']=counts['test']['dvd']['sequences']==10
    profiles={'student':profile(c['upstream']),'teacher':teacher_profile(c['upstream'],'teacher')}
    # Include fixed-window tail wastage, normalized to 1080p for every sequence.
    tail={}
    for d in DOMAINS:
        lengths=[len(r['blur']) for r in manifest['test'] if r['domain']==d]
        factor=sum(((n+11)//12)*12 for n in lengths)/sum(lengths)
        tail[d]=dict(factor=factor,gflops_per_useful_output=profiles['student']['arithmetic_GFLOPs_per_output']*factor)
    if profiles['student']['arithmetic_GFLOPs_per_output']>=500 or tail['gopro']['gflops_per_useful_output']>=500:
        raise ValueError(f'1080p FLOPs budget fail: {tail}')
    c.update(training_settings())
    c.update(output=str(run),manifest=str(run/'manifest.json'),migrated_from_config=str(source),
             migrated_from_config_sha256=sha256(source),model_id=training_settings()['training_recipe'])
    state=None
    if continue_checkpoint:
        import torch
        state=torch.load(continue_checkpoint,map_location='cpu',weights_only=False)
        if state.get('config')!=source_config or state.get('model_id')!=c['model_id'] or state.get('variant')!='student':
            raise ValueError('Continuation must use the source ShiftWave run, not old Shift500 quality/compact')
        if not 0 < state['update'] < c['total_updates']:
            raise ValueError('Only an unfinished ShiftWave checkpoint can continue')
    run.mkdir(parents=True,exist_ok=True)
    for name,value in [('manifest',manifest),('data_audit',counts),('profiles',profiles),('tail_budget',tail)]:
        (run/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    c['manifest_sha256']=sha256(c['manifest'])
    if state is not None:
        state['sampling_transition']=dict(source_checkpoint=str(Path(continue_checkpoint).resolve()),
            source_checkpoint_sha256=sha256(continue_checkpoint),source_config=source_config,
            update=state['update'],note='Preserve model/optimizer/scaler/update; subsequent updates use exactly equal sampling and global batch6. Earlier updates used source sampling.')
        state['config']=c
        (run/'student').mkdir()
        tmp=run/'student/latest.tmp';torch.save(state,tmp);tmp.replace(run/'student/latest.pth')
        (run/'sampling_transition.json').write_text(json.dumps(state['sampling_transition'],indent=2)+'\n')
    (run/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps(dict(config=str(run/'config.json'),student_gflops=profiles['student']['arithmetic_GFLOPs_per_output'],tail=tail),indent=2))
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--from-config',required=True)
    p.add_argument('--run',required=True);p.add_argument('--upstream');p.add_argument('--continue-checkpoint')
    a=p.parse_args();prepare(a.from_config,a.run,a.upstream,a.continue_checkpoint)

if __name__=='__main__':main()
