"""Read-only sequence discovery and immutable full-test manifests."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
BLUR = ('blur', 'blurry', 'input', 'blur_gamma', 'test_gt_blurred')
GT = ('gt', 'sharp', 'target', 'label', 'test_gt')
SKIP = {'raw', 'mp4', 'gt_mp4', 'blur_mp4', 'annotations', 'train', 'training', 'val', 'validation'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def blob(path):
    b = Path(path).read_bytes()
    return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, obj, exclusive=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def natural(s):
    return tuple((1, int(x)) if x.isdigit() else (0, x.casefold())
                 for x in re.split(r'(\d+)', str(s)))


def within(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    try:
        p.relative_to(r)
    except ValueError as exc:
        raise ValueError(f'Path escapes allowed root: {p}; root={r}') from exc
    return p


def rgb(path):
    with Image.open(path) as im:
        # Do not silently drop alpha, reduce bit depth, or recolor grayscale GT.
        if im.mode != 'RGB':
            raise ValueError(f'Expected native 8-bit RGB, got {im.mode}: {path}')
        a = np.array(im, dtype=np.uint8, copy=True)
    return a


def pixel_sha(a):
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def image_paths(folder):
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTS),
                  key=lambda p: (natural(p.name), p.name))


def dirs(folder, root):
    result = {}
    for p in sorted(Path(folder).iterdir(), key=lambda q: natural(q.name)):
        if not p.is_dir() or p.name.startswith('.'):
            continue
        key = p.name.casefold()
        if key in result:
            raise ValueError(f'Ambiguous case-insensitive directories: {folder}/{p.name}')
        within(p, root)
        result[key] = p
    return result


def one_alias(d, names, where):
    found = [d[k] for k in names if k in d]
    if len(found) > 1:
        raise ValueError(f'Ambiguous data variants at {where}: {found}; configure an explicit pair root')
    return found[0] if found else None


def discover(split_root):
    """Every complete sequence below one explicit split root, never train/config fallback."""
    root = Path(split_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = []

    def expand(b, g, name, depth):
        bf, gf = image_paths(b), image_paths(g)
        if bf or gf:
            bm, gm = {}, {}
            for files, mapping in ((bf, bm), (gf, gm)):
                for p in files:
                    within(p, root)
                    if p.stem in mapping:
                        raise ValueError(f'Duplicate frame stem: {p}')
                    mapping[p.stem] = p
            if not bm or bm.keys() != gm.keys():
                raise ValueError(f'Incomplete pairs {b} vs {g}: '
                                 f'blur_only={sorted(bm.keys()-gm.keys())[:8]}, '
                                 f'gt_only={sorted(gm.keys()-bm.keys())[:8]}')
            records.append((name or '__root__', [(p, gm[p.stem]) for p in bf]))
            return
        if depth > 8:
            raise ValueError(f'Dataset nesting exceeds supported depth: {b}')
        bd, gd = dirs(b, root), dirs(g, root)
        keys = [k for k in bd if k not in SKIP]
        if not keys:
            raise ValueError(f'No RGB frames in pair: {b}, {g}')
        if set(keys) != {k for k in gd if k not in SKIP}:
            raise ValueError(f'Mismatched sequence folders: {b} vs {g}')
        for k in keys:
            new_name = name if k == 'rgb' else '/'.join(filter(None, (name, bd[k].name)))
            expand(bd[k], gd[k], new_name, depth + 1)

    def scan(base, depth):
        d = dirs(base, root)
        b, g = one_alias(d, BLUR, base), one_alias(d, GT, base)
        if b is not None or g is not None:
            if b is None or g is None:
                raise ValueError(f'Missing blur or GT at {base}; do not silently skip this sequence')
            name = base.relative_to(root).as_posix()
            expand(b, g, '' if name == '.' else name, 0)
            return
        if depth > 8:
            raise ValueError(f'Unsupported nesting at {base}')
        for k, p in d.items():
            if k in SKIP or 'mp4' in k or 'purple' in k:
                continue
            scan(p, depth + 1)

    scan(root, 0)
    records.sort(key=lambda q: (natural(q[0]), q[0]))
    if not records or len({r[0] for r in records}) != len(records):
        raise ValueError(f'No sequences or duplicate sequence identifiers: {root}')
    return records


def manifest_id(manifest):
    return digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})


def load_manifest(path):
    m = read_json(path)
    if m.get('manifest_sha256') != manifest_id(m):
        raise ValueError('Manifest was modified after it was frozen')
    return m


def checked_frame(row, role):
    p = row[role]
    if sha(p) != row[role + '_sha256']:
        raise ValueError(f'Input/GT changed after manifest creation: {p}')
    a = rgb(p)
    if list(a.shape) != row['shape']:
        raise ValueError(f'Shape changed: {p}')
    return a


def build_manifest(config, out):
    cfg = read_json(config)
    if set(cfg['datasets']) != {'GoPro', 'BSD', 'DVD'}:
        raise ValueError('Exactly GoPro, BSD and DVD are required; no SKIP_BSD')
    result = {'schema': 1, 'scope': 'ALL_FRAMES_OF_ALL_LOCAL_TEST_SEQUENCES',
              'paper_comparable': False, 'coverage_note': 'Counts alone do not certify an official release.',
              'configuration': cfg, 'datasets': {}, 'sequences': []}
    for family, spec in cfg['datasets'].items():
        root = Path(spec['test_root']).resolve()
        if root.name != 'test':
            raise ValueError(f'Use an explicit direct test root: {root}')
        exposure = spec.get('exposure', '')
        if family == 'BSD' and exposure not in ('1ms8ms', '2ms16ms', '3ms24ms'):
            raise ValueError('BSD exposure must be verified from original source; no mixed/unknown aggregate')
        found = discover(root)
        frame_count = 0
        test_ids, test_pixels = set(), set()
        for seq, pairs in found:
            rows = []
            shape = None
            for i, (lq, gt) in enumerate(pairs):
                x, y = rgb(lq), rgb(gt)
                if x.shape != y.shape or (shape is not None and list(x.shape) != shape):
                    raise ValueError(f'Variable or mismatched sequence shape: {lq}, {gt}')
                shape = list(x.shape)
                r = {'index': i, 'frame': lq.stem, 'lq': str(lq.resolve()), 'gt': str(gt.resolve()),
                     'lq_sha256': sha(lq), 'gt_sha256': sha(gt), 'shape': shape,
                     'lq_pixel_sha256': pixel_sha(x), 'gt_pixel_sha256': pixel_sha(y)}
                rows.append(r)
                test_pixels.add(r['gt_pixel_sha256'])
            test_ids.add(seq)
            result['sequences'].append({'dataset': family, 'sequence': seq,
                                        'exposure': exposure, 'frames': rows})
            frame_count += len(rows)
        overlap = {'status': 'NOT_AUDITED', 'same_sequence_ids': [], 'exact_gt_pixel_matches': []}
        if spec.get('train_root'):
            train = discover(spec['train_root'])
            overlap['same_sequence_ids'] = sorted(test_ids & {q[0] for q in train})
            for seq, pairs in train:
                for _, p in pairs:
                    if pixel_sha(rgb(p)) in test_pixels:
                        overlap['exact_gt_pixel_matches'].append(str(p))
            overlap['status'] = 'REVIEW_REQUIRED' if (overlap['same_sequence_ids'] or
                                    overlap['exact_gt_pixel_matches']) else 'NO_EXACT_OVERLAP_FOUND'
            # Do not collapse GoPro GOPRxxxx_chunk to a shared acquisition prefix.
            if overlap['status'] == 'REVIEW_REQUIRED':
                raise ValueError(f'{family}: train/test overlap evidence: {overlap}')
        expected = spec.get('expected_counts', {})
        counts_match = (not expected or (len(found) == expected.get('videos', len(found)) and
                                        frame_count == expected.get('frames', frame_count)))
        result['datasets'][family] = {'videos': len(found), 'frames': frame_count,
            'expected_counts': expected, 'counts_match': counts_match, 'exposure': exposure,
            'train_overlap_audit': overlap,
            'image_extensions': dict(Counter(Path(r['lq']).suffix.lower() for s in result['sequences']
                                        if s['dataset'] == family for r in s['frames']))}
        print(f'{family}: {len(found)} videos / {frame_count} frames; reference counts match={counts_match}', flush=True)
    result['manifest_sha256'] = manifest_id(result)
    write_json(out, result, exclusive=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    build_manifest(a.config, a.out)

if __name__ == '__main__':
    main()
