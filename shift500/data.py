"""Strict split discovery, acquisition-disjoint holdout, native full-frame clips."""
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DOMAINS = ('gopro', 'dvd', 'bsd')
EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
BLUR = {'blur', 'input', 'lq', 'blurry'}
GT = {'gt', 'sharp', 'target', 'hq', 'clear'}
ALIASES = {'gopro': ('GoPro', 'gopro', 'GOPRO', 'GOPRO_Large'),
           'dvd': ('DVD', 'dvd', 'DeepVideoDeblurring_Dataset'), 'bsd': ('BSD', 'bsd')}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def images(folder):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXT]
    def frame_id(p):
        m = re.search(r'(\d+)$', p.stem)
        if not m:
            raise ValueError(f'No chronological frame id: {p}')
        return int(m.group(1))
    return sorted(files, key=frame_id)


def discover(root, domain, split):
    root = Path(root).resolve()
    found = []
    # Explicit split only. Supports split/scene/blur and split/blur/scene,
    # plus blur/split/scene. Never treats unsplit data as train or test.
    for br in sorted(p for p in root.rglob('*') if p.is_dir()):
        rel = br.relative_to(root)
        parts = rel.parts
        lower = [p.lower() for p in parts]
        split_aliases = {'train', 'training'} if split == 'train' else {'test', 'testing'}
        if not any(p in split_aliases for p in lower) or not any(p in BLUR for p in lower):
            continue
        if not any(p.is_file() and p.suffix.lower() in EXT for p in br.iterdir()):
            continue
        blur_positions = [i for i,p in enumerate(lower) if p in BLUR]
        if len(blur_positions) != 1:
            raise ValueError(f'Ambiguous blur layout: {br}')
        i = blur_positions[0]
        parent = root.joinpath(*parts[:i])
        gt_names = [p.name for p in parent.iterdir() if p.is_dir() and p.name.lower() in GT]
        candidates = [root.joinpath(*parts[:i], name, *parts[i+1:]) for name in gt_names]
        candidates = [p for p in candidates if p.is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'Need exactly one paired GT folder for {br}: {candidates}')
        gr = candidates[0]
        bs, gs = images(br), images(gr)
        if not bs or [p.stem for p in bs] != [p.stem for p in gs]:
            raise ValueError(f'Exact frame pairing mismatch: {br} / {gr}')
        ids = [int(re.search(r'(\d+)$', p.stem).group(1)) for p in bs]
        if any(b != a+1 for a,b in zip(ids,ids[1:])):
            raise ValueError(f'Duplicate/non-contiguous frames: {br}')
        shapes = set()
        for a,b in zip(bs,gs):
            with Image.open(a) as ia, Image.open(b) as ib:
                if ia.size != ib.size:
                    raise ValueError(f'Pair shape mismatch: {a}')
                shapes.add(ia.size)
        if len(shapes) != 1:
            raise ValueError(f'Changing frame size: {br}')
        w,h = next(iter(shapes))
        name_parts = [p for p in parts if p.lower() not in BLUR | split_aliases]
        name = '/'.join(name_parts)
        group = re.sub(r'(GOPR\d+_\d+)_\d+', r'\1', name, flags=re.I) if domain=='gopro' else name
        found.append(dict(domain=domain, name=name, group=group, height=h, width=w,
                          blur=[str(p.resolve()) for p in bs], gt=[str(p.resolve()) for p in gs]))
    paired_gt={str(Path(r['gt'][0]).parent) for r in found}
    for gr in (p for p in root.rglob('*') if p.is_dir()):
        lower=[p.lower() for p in gr.relative_to(root).parts]
        aliases={'train','training'} if split=='train' else {'test','testing'}
        if any(p in aliases for p in lower) and any(p in GT for p in lower):
            if any(p.is_file() and p.suffix.lower() in EXT for p in gr.iterdir()):
                if str(gr) not in paired_gt:
                    raise ValueError(f'GT sequence has no paired blur sequence: {gr}')
    if not found:
        raise ValueError(f'No explicit {domain}/{split} sequences in {root}')
    if len({r['name'] for r in found}) != len(found):
        raise ValueError(f'Duplicate sequence names in {domain}/{split}')
    return found


def make_manifest(roots, seed=20260911, frames=16):
    result = dict(version=1, seed=seed, frames=frames, roots=roots, train=[], val=[], test=[])
    for d in DOMAINS:
        records = discover(roots[d], d, 'train')
        if any(len(r['blur']) < frames for r in records):
            raise ValueError(f'{d}: training sequence shorter than {frames}; report, do not silently drop')
        groups = sorted({r['group'] for r in records})
        if len(groups) < 3:
            raise ValueError(f'{d}: need at least 3 training groups')
        random.Random(seed+sum(map(ord,d))).shuffle(groups)
        holdout = set(groups[:max(1,round(.1*len(groups)))])
        for r in records:
            result['val' if r['group'] in holdout else 'train'].append(r)
        tests = discover(roots[d], d, 'test')
        if set(r['group'] for r in records) & set(r['group'] for r in tests):
            raise ValueError(f'{d}: acquisition names overlap train/test; inspect layout')
        result['test'] += tests
    # Content-level leakage check on GT, streaming hashes; no decoded images kept.
    seen = {}
    for split in ('train','val','test'):
        for r in result[split]:
            for path in r['gt']:
                digest = sha256(path)
                if digest in seen and seen[digest][0] != split:
                    raise ValueError(f'GT content leakage: {seen[digest]} versus {split}:{path}')
                seen[digest] = (split, path)
    return result


def read_rgb(path):
    with Image.open(path) as im:
        a = np.array(im.convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(a).permute(2,0,1)


def load_indices(record, indices):
    return (torch.stack([read_rgb(record['blur'][i]) for i in indices]),
            torch.stack([read_rgb(record['gt'][i]) for i in indices]))


class Clips(Dataset):
    # Exactly 2 GoPro, 1 DVD, 1 BSD clips per optimizer update across ranks.
    cycle = ('gopro','dvd','gopro','bsd')
    def __init__(self, manifest, total, frames=16, seed=20260911):
        self.records = {d:[r for r in manifest['train'] if r['domain']==d] for d in DOMAINS}
        self.total, self.frames, self.seed = total, frames, seed
    def __len__(self):
        return self.total
    def __getitem__(self, index):
        rng = random.Random(self.seed+int(index)*9176)
        domain = self.cycle[index%4]
        r = rng.choice(self.records[domain])
        start = rng.randrange(len(r['blur'])-self.frames+1)
        x,y = load_indices(r,range(start,start+self.frames))
        if rng.random()<.5: x,y=x.flip(-1),y.flip(-1)
        if rng.random()<.5: x,y=x.flip(-2),y.flip(-2)
        if rng.random()<.5: x,y=x.flip(0),y.flip(0)
        return {'blur':x.contiguous(), 'gt':y.contiguous(), 'domain':domain}
