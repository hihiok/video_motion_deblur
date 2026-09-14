"""Strict split discovery, acquisition-disjoint holdout, native full-frame clips."""
from bisect import bisect_right
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
    result = dict(version=2, seed=seed, frames=frames, roots=roots,
                  split_policy='official clip train/test; acquisition-group holdout inside official train only',
                  split_audit={}, train=[], val=[], test=[])
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
        shared_clips = sorted(set(r['name'] for r in records) & set(r['name'] for r in tests))
        if shared_clips:
            raise ValueError(f'{d}: complete clip names overlap train/test: {shared_clips}')
        shared_groups = sorted(set(r['group'] for r in records) & set(r['group'] for r in tests))
        # Official GoPro splits different chunks of some acquisitions across
        # train/test. Keep that benchmark partition; a shared acquisition label
        # does not establish shared frames. Never move/drop official test clips.
        # The holdout above still groups ONLY the official training clips.
        if shared_groups and d != 'gopro':
            raise ValueError(f'{d}: acquisition names overlap train/test; inspect layout')
        result['split_audit'][d] = {
            'official_train_clips': len(records),
            'official_test_clips': len(tests),
            'official_train_test_shared_acquisitions': shared_groups,
            'acquisition_overlap_policy': 'record_only_for_gopro' if d == 'gopro' else 'reject',
            'train_val_holdout_groups': sorted(holdout),
            'complete_clip_overlap_check': 'passed',
        }
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
    result['split_audit']['gt_file_sha256_cross_split_check'] = 'passed'
    return result


def read_rgb(path):
    with Image.open(path) as im:
        a = np.array(im.convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(a).permute(2,0,1)


def load_indices(record, indices):
    return (torch.stack([read_rgb(record['blur'][i]) for i in indices]),
            torch.stack([read_rgb(record['gt'][i]) for i in indices]))


def load_training_clip(record, start, frames, crop_size, rng, augment=True):
    """One shared crop for the entire temporal clip and all paired GT frames.

    Crop before tensor construction: never stage full-frame clips on the GPU.
    Spatial flips/90-degree rotation match upstream video_image_dataset.py.
    """
    if start < 0 or start + frames > len(record['blur']):
        raise ValueError('Training clip crosses sequence boundary')
    if crop_size:
        h, w = record['height'], record['width']
        if min(h, w) < crop_size:
            raise ValueError(f'Image smaller than training crop: {record["name"]}')
        left, top = rng.randrange(w - crop_size + 1), rng.randrange(h - crop_size + 1)
        box = (left, top, left + crop_size, top + crop_size)
        def read_crop(path):
            with Image.open(path) as image:
                if image.size != (w, h):
                    raise ValueError(f'Image dimensions changed since data audit: {path}')
                array = np.array(image.crop(box).convert('RGB'), dtype=np.float32) / 255.
            return torch.from_numpy(array).permute(2, 0, 1)
        x, y = [torch.stack([read_crop(record[key][i]) for i in range(start, start + frames)])
                for key in ('blur', 'gt')]
    else:
        x, y = load_indices(record, range(start, start + frames))
        box = (0, 0, x.shape[-1], x.shape[-2])
    if augment:
        if rng.random() < .5: x, y = x.flip(-1), y.flip(-1)
        if rng.random() < .5: x, y = x.flip(-2), y.flip(-2)
        if crop_size:
            if rng.random() < .5: x, y = x.rot90(1, (-2, -1)), y.rot90(1, (-2, -1))
        elif rng.random() < .5:
            x, y = x.flip(0), y.flip(0)
    return {'blur': x.contiguous(), 'gt': y.contiguous(), 'domain': record['domain'],
            'crop_box': box, 'sequence': record['name']}


class Clips(Dataset):
    # Repeat twice per global batch of eight: GoPro4/DVD2/BSD2.
    cycle = ('gopro','dvd','gopro','bsd')
    def __init__(self, manifest, total, frames=16, seed=20260911, crop_size=0,
                 n_frames_per_video=None):
        self.records = {d:[r for r in manifest['train'] if r['domain']==d] for d in DOMAINS}
        self.total, self.frames, self.seed = total, frames, seed
        self.crop_size = crop_size
        self.ends={};self.orders={}
        for d,records in self.records.items():
            ends=[];count=0
            for r in records:
                n=len(r['blur']) if n_frames_per_video is None else min(n_frames_per_video,len(r['blur']))
                if n<frames:raise ValueError('Training sequence too short')
                count+=n-frames+1;ends.append(count)
            if not count:raise ValueError(f'Empty training domain: {d}')
            self.ends[d]=ends

    def __len__(self):
        return self.total

    def locate(self,index):
        domain=self.cycle[index%4]
        occurrence=index//2 if domain=='gopro' else index//4
        count=self.ends[domain][-1]
        epoch,position=divmod(occurrence,count)
        if self.orders.get(domain,(-1,None))[0]!=epoch:
            order=list(range(count))
            random.Random(self.seed+epoch*9176+sum(map(ord,domain))).shuffle(order)
            self.orders[domain]=(epoch,order)
        window=self.orders[domain][1][position]
        record=bisect_right(self.ends[domain],window)
        start=window-(self.ends[domain][record-1] if record else 0)
        return domain,self.records[domain][record],start

    def __getitem__(self, index):
        rng = random.Random(self.seed+int(index)*9176)
        domain,r,start=self.locate(index)
        return load_training_clip(r, start, self.frames, self.crop_size, rng)
