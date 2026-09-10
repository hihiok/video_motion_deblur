"""Audited, sequence-disjoint train/holdout manifest and native-frame clips."""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rtf_t6.datasets import discover_sequences, PairedSequence, load_clip
from .flow import is_cut, pair_flow


DOMAINS = ('gopro', 'bsd', 'dvd')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sequence_group(domain, name):
    # GoPro chunks from one acquisition must remain in the same split.
    if domain == 'gopro' and re.match(r'^GOPR\d+_\d+_\d+$', name, re.I):
        return name.rsplit('_', 1)[0]
    return name


def audit_sequence(seq):
    stems = [p.stem for p in seq.blur]
    ids = []
    for stem in stems:
        match = re.search(r'(\d+)$', stem)
        if match is None:
            raise ValueError(f'Cannot establish chronological IDs: {seq.name}/{stem}')
        ids.append(int(match.group(1)))
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids = [ids[i] for i in order]
    if any(b != a + 1 for a, b in zip(ids, ids[1:])):
        raise ValueError(f'Non-contiguous or duplicate frame IDs: {seq.domain}/{seq.name}')
    blur, gt = [seq.blur[i] for i in order], [seq.gt[i] for i in order]
    shapes, signature = set(), hashlib.sha256()
    for a, b in zip(blur, gt):
        with Image.open(a) as img:
            sa = img.size
        with Image.open(b) as img:
            sb = img.size
        if sa != sb:
            raise ValueError(f'Blur/GT size mismatch: {a}, {b}')
        shapes.add(sa)
        for p in (a, b):
            stat = p.stat()
            signature.update(f'{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    if len(shapes) != 1:
        raise ValueError(f'Changing resolution within sequence: {seq.name}')
    w, h = next(iter(shapes))
    return dict(domain=seq.domain, name=seq.name, group=sequence_group(seq.domain, seq.name),
                blur=[str(p.resolve()) for p in blur], gt=[str(p.resolve()) for p in gt],
                height=h, width=w, source_signature=signature.hexdigest())


def make_manifest(roots, clip_length=4, seed=20260910):
    result = dict(version=1, clip_length=clip_length, seed=seed,
                  validation_protocol='10% acquisition-group holdout from official TRAIN only; test untouched',
                  train=[], val=[], roots={})
    all_paths = set()
    for domain in DOMAINS:
        root, seqs = discover_sequences(domain, roots[domain], 'train', clip_length)
        # Require explicit train layout: never treat an unsplit root as training.
        has_train = (root.name.lower() in ('train', 'training') or
                     any(p.name.lower() in ('train', 'training') for p in root.iterdir() if p.is_dir()) or
                     any(p.is_dir() and p.name.lower() in ('train', 'training') for p in root.glob('*/*')))
        if not has_train:
            raise ValueError(f'Explicit training split required: {root}')
        records = [audit_sequence(s) for s in seqs]
        groups = sorted({r['group'] for r in records})
        if len(groups) < 3:
            raise ValueError(f'{domain}: fewer than 3 independent training groups')
        random.Random(seed + sum(map(ord, domain))).shuffle(groups)
        holdout = set(groups[:max(1, round(len(groups) * .1))])
        result['roots'][domain] = str(root)
        for record in records:
            for path in record['blur'] + record['gt']:
                if path in all_paths:
                    raise ValueError(f'Reused image across domains/sequences: {path}')
                all_paths.add(path)
            split = 'val' if record['group'] in holdout else 'train'
            result[split].append(record)
        # Actual image-content overlap check on all GT frames across train/holdout.
        train_hashes = {sha256(p) for r in result['train'] if r['domain'] == domain for p in r['gt']}
        if any(sha256(p) in train_hashes for r in result['val'] if r['domain'] == domain for p in r['gt']):
            raise ValueError(f'{domain}: identical GT content crosses train/holdout split')
        print(json.dumps({'event': 'domain_audit', 'domain': domain,
                          'sequences': len(records), 'holdout_groups': sorted(holdout)}), flush=True)
    return result


def load_record(record, start, length, augment=False, seed=0, cut_threshold=.30):
    seq = PairedSequence(record['domain'], '', record['name'],
                         tuple(map(Path, record['blur'])), tuple(map(Path, record['gt'])))
    blur, gt = load_clip(seq, list(range(start, start + length)), 0, random.Random(seed), augment)
    frames = gt.permute(0, 2, 3, 1).numpy()
    inputs = blur.permute(0, 2, 3, 1).numpy()
    flows, masks, resets = [], [], [True]
    for t in range(1, length):
        cut = is_cut(inputs[t - 1], inputs[t], cut_threshold)
        resets.append(cut)
        if cut:
            h, w = frames[t].shape[:2]
            flow, mask = torch.zeros(2, h, w), torch.zeros(1, h, w)
        else:
            flow, mask = pair_flow(frames[t - 1], frames[t])
        flows.append(flow)
        masks.append(mask)
    return {'blur': blur, 'gt': gt, 'flow': torch.stack(flows), 'mask': torch.stack(masks),
            'resets': torch.tensor(resets), 'domain': record['domain'],
            'sequence': record['name'], 'start': start}


class TrainingClips(Dataset):
    """Deterministic index->clip; resume skips no data and DDP needs no sampler."""
    def __init__(self, manifest, samples, seed, length=4, cut_threshold=.30):
        self.records = {d: [r for r in manifest['train'] if r['domain'] == d] for d in DOMAINS}
        self.samples, self.seed, self.length = samples, seed, length
        self.cut_threshold = cut_threshold

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = random.Random(self.seed + int(index) * 9176)
        domain = DOMAINS[int(index) % 3]
        record = rng.choice(self.records[domain])
        start = rng.randrange(len(record['blur']) - self.length + 1)
        return load_record(record, start, self.length, True, rng.randrange(2**31), self.cut_threshold)


class RankSampleIndices:
    def __init__(self, start_update, end_update, rank, world, clips_per_update=2):
        if clips_per_update % world:
            raise ValueError('Global clips per update must be divisible by world size')
        self.args = start_update, end_update, rank, world, clips_per_update

    def __iter__(self):
        start, end, rank, world, clips = self.args
        for update in range(start, end):
            for offset in range(rank, clips, world):
                yield update * clips + offset

    def __len__(self):
        start, end, _, world, clips = self.args
        return (end - start) * clips // world
