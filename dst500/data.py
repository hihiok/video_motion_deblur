"""Reuse strict paired RGB discovery; native full frames and train-only holdout."""
import argparse
import json
import random
from pathlib import Path
from shift500.data import discover, sha256, load_indices, load_training_clip

DOMAINS=('gopro','dvd','bsd')
CYCLE=('gopro','gopro','gopro','gopro','gopro','gopro','dvd','bsd')
GOPRO_TEST={'GOPR0384_11_00','GOPR0384_11_05','GOPR0385_11_01','GOPR0396_11_00',
            'GOPR0410_11_00','GOPR0854_11_00','GOPR0862_11_00','GOPR0868_11_00',
            'GOPR0869_11_00','GOPR0871_11_00','GOPR0881_11_01'}


def build_manifest(roots, frames=6):
    out={'version':1,'frames':frames,'roots':roots,'train':[],'val':[],'test':[], 'audit':{}}
    for d in DOMAINS:
        records=discover(roots[d],d,'train')
        if any(len(r['blur'])<frames for r in records): raise ValueError(f'Short {d} training clip')
        groups=sorted({r['group'] for r in records})
        if len(groups)<3: raise ValueError(f'Insufficient {d} training acquisitions')
        random.Random(20260915).shuffle(groups)
        holdout=set(groups[:max(1,round(len(groups)*.1))])
        for r in records: out['val' if r['group'] in holdout else 'train'].append(r)
        try:
            tests=discover(roots[d],d,'test')
        except ValueError as e:
            if d=='gopro' or not str(e).startswith('No explicit'): raise
            tests=[]
        if d=='gopro':
            if {r['name'] for r in tests}!=GOPRO_TEST or sum(len(r['blur']) for r in tests)!=1111:
                raise ValueError('GoPro test must have exact official 11 names / 1111 frames')
            if len(records)!=22: raise ValueError('GoPro official training split must have 22 clips')
        if set(r['name'] for r in records)&set(r['name'] for r in tests):
            raise ValueError(f'Entire clip overlaps train/test in {d}')
        out['test']+=tests
        out['audit'][d]={'train_clips':len(records),'test_clips':len(tests),
            'test_frames':sum(len(r['blur']) for r in tests),'holdout_groups':sorted(holdout),
            'train_test_shared_acquisitions':sorted({r['group'] for r in records}&{r['group'] for r in tests}),
            'note':'Official GoPro acquisition names may overlap; whole clips must not overlap. Auxiliary test may be incomplete.'}
    # Detect actual duplicate files across splits without confusing acquisition IDs.
    seen={}
    for split in ('train','val','test'):
        for r in out[split]:
            for p in r['gt']:
                digest=sha256(p)
                if digest in seen and seen[digest][0]!=split:
                    raise ValueError(f'GT byte-identical across splits: {seen[digest]} vs {split}:{p}')
                seen[digest]=(split,p)
    out['audit']['cross_split_gt_sha256']='passed'
    out['audit']['teacher_holdout_note']='Holdout excludes student sampling; official pretrained teacher may have seen it.'
    return out


def sample(manifest, update, micro, rank):
    index=update*8+micro*2+rank
    domain=CYCLE[index%8]
    records=[r for r in manifest['train'] if r['domain']==domain]
    rng=random.Random(20260915+index*9176)
    # Weight by valid windows, not number of sequences.
    counts=[len(r['blur'])-manifest['frames']+1 for r in records]
    r=rng.choices(records,weights=counts,k=1)[0]
    return load_training_clip(r,rng.randrange(len(r['blur'])-manifest['frames']+1),manifest['frames'],0,rng)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gopro-root',required=True)
    p.add_argument('--dvd-root',required=True);p.add_argument('--bsd-root',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    roots={d:str(Path(getattr(a,d+'_root')).resolve()) for d in DOMAINS}
    # Explicitly select 3ms-24ms data; mixed-exposure BSD roots are rejected.
    b=roots['bsd'].lower().replace('-','').replace('_','')
    if '3ms24ms' not in b:
        raise ValueError('Point --bsd-root to a dedicated BSD 3ms-24ms folder (a faithful symlink view is allowed)')
    out=build_manifest(roots)
    path=Path(a.output)
    if path.exists(): raise FileExistsError('Use an existing manifest unchanged or a new output path')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['audit'],indent=2))

if __name__=='__main__': main()
