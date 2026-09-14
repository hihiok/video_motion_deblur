"""Explicit test splits, immutable RGB8 frames, and train/selection overlap audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
from rtf_temporal.data import sha256
from rtf_temporal.posttrain import numeric_frames, write_json

META = Path(__file__).parent / 'meta'
EXPECTED = {'gopro': (11, 1111), 'bsd': (20, 3000), 'dvd': (10, 1000)}


def rgb(path):
    with Image.open(path) as im:
        if im.mode != 'RGB':
            raise ValueError(f'Expected RGB8 source; refusing implicit RAW/grayscale/alpha conversion: {path} ({im.mode})')
        return np.array(im, dtype=np.uint8)


def pixel_hash(array):
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def child(parent, name):
    matches = [p for p in Path(parent).iterdir() if p.is_dir() and p.name.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f'Expected one directory {parent}/{name}; found {len(matches)}')
    return matches[0]


def image_dir(path):
    # BSD official distribution has Blur/RGB and Sharp/RGB.
    nested = [p for p in path.iterdir() if p.is_dir() and p.name.lower() == 'rgb']
    return nested[0] if len(nested) == 1 else path


def pair_directories(spec):
    root = Path(spec['root']).expanduser().resolve()
    test = root if root.name.lower() == 'test' else child(root, 'test')
    lq_name, gt_name = spec['lq_dir'], spec['gt_dir']
    if spec['layout'] == 'split_modalities':
        lq, gt = child(test, lq_name), child(test, gt_name)
        left = {p.name: p for p in lq.iterdir() if p.is_dir()}
        right = {p.name: p for p in gt.iterdir() if p.is_dir()}
        if left.keys() != right.keys():
            raise ValueError('Blur/GT sequence name sets differ')
        return [(name, image_dir(left[name]), image_dir(right[name])) for name in sorted(left)]
    if spec['layout'] != 'sequence_modalities':
        raise ValueError('layout must be split_modalities or sequence_modalities')
    return [(p.name, image_dir(child(p, lq_name)), image_dir(child(p, gt_name)))
            for p in sorted(test.iterdir()) if p.is_dir()]


def official_meta(domain):
    if domain == 'bsd':
        return None  # Original RGB test: 20 scenes, each 150 frames, 640x480.
    path = META / ('GoPro_test.txt' if domain == 'gopro' else 'DVD_test.txt')
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            name, count, dimensions = line.split()
            rows[name] = (int(count), tuple(map(int, dimensions.strip('()').split(','))))
    return rows


def prepare_dataset(domain, spec, out, *, expected=None, check_official_names=True):
    """Test-only expected override exists for synthetic unit tests, never exposed by CLI."""
    if domain not in EXPECTED:
        raise ValueError(domain)
    if domain == 'bsd' and spec.get('variant') != '3ms24ms':
        raise ValueError('This benchmark is explicitly BSD 3ms24ms only')
    if domain == 'bsd' and not spec.get('variant_evidence'):
        raise ValueError('BSD exposure setting requires source/path evidence; folder BSD alone is insufficient')
    if domain == 'gopro' and spec['lq_dir'].lower() not in ('blur', 'blur_gamma'):
        raise ValueError('GoPro variant must be explicit blur or blur_gamma')
    if domain == 'gopro' and spec.get('variant') != spec['lq_dir'].lower():
        raise ValueError('GoPro variant label must match the explicitly selected input directory')
    pairs = pair_directories(spec)
    nseq, nframe = expected or EXPECTED[domain]
    if len(pairs) != nseq:
        raise ValueError(f'{domain}: expected {nseq} official test sequences, found {len(pairs)}')
    official = official_meta(domain) if check_official_names else None
    if official is not None and {x[0] for x in pairs} != set(official):
        raise ValueError(f'{domain}: sequence IDs differ from published test metadata; do not rename to make them pass')
    records = []
    out = Path(out)
    for name, lq, gt in pairs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Unsupported sequence name: {name}')
        a, b = numeric_frames(lq), numeric_frames(gt)
        if [p.stem for p in a] != [p.stem for p in b]:
            raise ValueError(f'{domain}/{name}: strict input/GT frame pairing failed')
        if official is not None and len(a) != official[name][0]:
            raise ValueError(f'{domain}/{name}: wrong frame count')
        if domain == 'bsd' and expected is None and len(a) != 150:
            raise ValueError(f'BSD test must have 150 frames per scene: {name}')
        for folder in ('blur', 'gt'):
            (out / folder / name).mkdir(parents=True, exist_ok=False)
        frames = []
        shape = None
        for i, (bp, gp) in enumerate(zip(a, b)):
            ba, ga = rgb(bp), rgb(gp)
            if ba.shape != ga.shape or (shape is not None and shape != ba.shape):
                raise ValueError(f'Spatial mismatch or changing sequence geometry: {bp}')
            shape = ba.shape
            if official is not None and shape != official[name][1]:
                raise ValueError(f'{domain}/{name}: native {shape} differs from published {official[name][1]}; no silent resizing')
            if domain == 'bsd' and expected is None and shape != (480, 640, 3):
                raise ValueError('BSD 3ms24ms RGB test must be 480x640; RAW/other variants are separate benchmarks')
            dests = {}
            for label, source, array in [('blur', bp, ba), ('gt', gp, ga)]:
                dest = out / label / name / f'{i:08d}.png'
                # Canonical PNG names preserve exact decoded RGB values even for JPEG source datasets.
                with Image.open(source) as im:
                    linkable = source.suffix.lower() == '.png' and im.mode == 'RGB'
                if linkable:
                    dest.symlink_to(source.resolve())
                else:
                    Image.fromarray(array).save(dest)
                dests[label] = str(dest.resolve() if not linkable else dest.absolute())
            frames.append({'index': i, 'original_frame_id': bp.stem,
                'source_blur': str(bp.resolve()), 'source_gt': str(gp.resolve()),
                'blur': dests['blur'], 'gt': dests['gt'],
                'blur_sha256': sha256(dests['blur']), 'gt_sha256': sha256(dests['gt']),
                'gt_rgb_sha256': pixel_hash(ga), 'shape': list(shape)})
        records.append({'name': name, 'frames': frames})
        print(f'DATA_PREPARED {domain}/{name} {len(frames)}', flush=True)
    count = sum(len(r['frames']) for r in records)
    if count != nframe:
        raise ValueError(f'{domain}: expected {nframe} frames, found {count}')
    return {'domain': domain, 'spec': spec, 'sequences': records, 'frames': count,
            'input_root': str((out / 'blur').absolute()), 'gt_root': str((out / 'gt').absolute())}


def verify_files(dataset):
    for seq in dataset['sequences']:
        for f in seq['frames']:
            for key in ('blur', 'gt'):
                if sha256(f[key]) != f[key + '_sha256']:
                    raise ValueError(f'Frozen dataset modified: {f[key]}')


def overlap_audit(training_manifest, datasets):
    """Includes validation because it selected best_stable; checks ALL domains."""
    records = training_manifest['train'] + training_manifest['val']
    identities = {(r['domain'], r['name'], Path(p).stem) for r in records for p in r['gt']}
    content = {}
    unreadable = []
    for i, r in enumerate(records):
        for path in r['gt']:
            try:
                digest = pixel_hash(rgb(path))
                content.setdefault(digest, []).append(path)
            except (OSError, ValueError) as exc:
                unreadable.append({'path': path, 'error': str(exc)})
        if (i + 1) % 10 == 0:
            print(f'TRAIN_SELECTION_HASHED {i+1}/{len(records)}', flush=True)
    result = {'scope': 'All fine-tuning train AND checkpoint-selection val GT; decoded RGB hashes',
              'unreadable': unreadable, 'datasets': {}}
    for domain, ds in datasets.items():
        hits = []
        for seq in ds['sequences']:
            for f in seq['frames']:
                identity = (domain, seq['name'], f['original_frame_id']) in identities
                matching = content.get(f['gt_rgb_sha256'], [])
                if identity or matching:
                    hits.append({'sequence': seq['name'], 'frame': f['original_frame_id'],
                                 'identity_overlap': identity, 'matching_train_or_val_paths': matching[:3]})
        result['datasets'][domain] = {'status': 'OVERLAP' if hits else ('UNVERIFIABLE' if unreadable else 'CLEAN'),
                                     'overlap_frames': len(hits), 'examples': hits[:100]}
    return result
