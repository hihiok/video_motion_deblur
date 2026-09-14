"""Plan and verify a byte-preserving BSD/test GT copy from original Sharp/RGB.

MP4 is provenance evidence, NEVER an image source for benchmark metrics.
Automatic alignment requires exact decoded blur-pixel equality for every frame.
"""
from __future__ import annotations
import argparse
import os
import shutil
import uuid
from pathlib import Path
from .data import (EXTS, digest, image_paths, pixel_sha, read_json, rgb, sha, within, write_json)


def rgb_dir(path):
    p = Path(path)
    for name in ('RGB', 'rgb'):
        if (p / name).is_dir():
            return p / name
    return p


def inventory(bsd_root, raw_roots, mp4_root):
    root = Path(bsd_root).resolve()
    mp4root = Path(mp4_root).resolve()
    if not mp4root.is_dir():
        raise FileNotFoundError(mp4root)
    videos = sorted(str(p.resolve()) for p in mp4root.rglob('*.mp4'))
    if not videos:
        raise ValueError(f'No MP4 provenance files: {mp4root}')
    sources, seen = [], set()
    for raw in raw_roots:
        raw = Path(raw).resolve()
        if not raw.is_dir():
            raise FileNotFoundError(raw)
        for parent, ds, _ in os.walk(raw, followlinks=False):
            parent = Path(parent)
            # Original TEST source only. No use of train frames as test GT.
            ds[:] = sorted(d for d in ds if not d.startswith('.') and
                           d.casefold() not in ('train', 'training', 'val', 'validation', 'raw') and
                           'mp4' not in d.casefold() and 'purple' not in d.casefold())
            for name in list(ds):
                if name.casefold() != 'sharp':
                    continue
                sharp_dir = rgb_dir(parent / name)
                if not image_paths(sharp_dir):
                    continue
                within(sharp_dir, raw)
                # Must be in an explicit test split of the original release.
                if 'test' not in {part.casefold() for part in sharp_dir.parts}:
                    continue
                options = [rgb_dir(parent / b) for b in ('Blur', 'blur') if (parent / b).is_dir()]
                if len(options) != 1 or not image_paths(options[0]):
                    continue
                blur_dir = options[0]
                within(blur_dir, raw)
                key = str(sharp_dir.resolve())
                if key in seen:
                    continue
                seen.add(key)
                exposure = [e for e in ('1ms8ms', '2ms16ms', '3ms24ms')
                            if e in key.lower().replace('-', '').replace('_', '')]
                sources.append({'sharp_dir': key, 'blur_dir': str(blur_dir.resolve()),
                                'exposure': exposure[0] if len(exposure) == 1 else 'UNKNOWN'})
    if not sources:
        raise ValueError('No original test/<sequence>/Sharp[/RGB] + Blur[/RGB] pairs found')
    return {'schema': 1, 'bsd_root': str(root), 'mp4_root': str(mp4root),
            'mp4_files': videos, 'original_test_sources': sources}


def local_blur_sequences(root):
    test = Path(root).resolve() / 'test'
    options = [test / n for n in ('blur', 'Blur') if (test / n).is_dir()]
    if len(options) != 1:
        raise ValueError('Expected exactly one BSD/test/blur directory; do not auto-move datasets')
    found = []
    for parent, ds, _ in os.walk(options[0], followlinks=False):
        ds[:] = sorted(d for d in ds if not d.startswith('.') and d.lower() != 'raw')
        p = Path(parent)
        frames = image_paths(p)
        if frames:
            name = p.relative_to(options[0]).as_posix()
            if name.endswith('/RGB') or name.endswith('/rgb'):
                name = name.rsplit('/', 1)[0]
            if name == '.':
                raise ValueError('A video subdirectory is required in BSD/test/blur')
            for f in frames:
                within(f, test)
            found.append((name, frames))
    if not found:
        raise ValueError('No local BSD/test blur frames found')
    return sorted(found)


def align_exact(local, candidate, cache=None):
    """No similarity ranking, no PSNR-based offset search, no sorted-index pairing."""
    bf = image_paths(candidate['blur_dir'])
    gf = image_paths(candidate['sharp_dir'])
    gm = {p.stem: p for p in gf}
    if len(gm) != len(gf) or not bf or any(p.stem not in gm for p in bf):
        return None
    cache = {} if cache is None else cache
    def pixels(p):
        key = str(p.resolve())
        if key not in cache:
            cache[key] = pixel_sha(rgb(p))
        return cache[key]
    for q in bf:
        within(q, candidate['blur_dir'])
    for q in gf:
        within(q, candidate['sharp_dir'])
    local_hashes = [pixels(p) for p in local]
    raw_hashes = [pixels(p) for p in bf]
    n = len(local)
    # A unique contiguous exact pixel identity establishes offsets even after renaming.
    starts = [s for s in range(len(bf) - n + 1) if raw_hashes[s:s+n] == local_hashes]
    if len(starts) != 1:
        return None
    start = starts[0]
    rows = []
    for p, src_blur in zip(local, bf[start:start+n]):
        src_gt = gm[src_blur.stem]
        x, y = rgb(p), rgb(src_gt)
        if x.shape != y.shape:
            raise ValueError(f'Original blur/Sharp shape mismatch: {p}, {src_gt}')
        rows.append({'local_blur': str(p.resolve()), 'local_blur_sha256': sha(p),
                     'source_blur': str(src_blur.resolve()), 'source_blur_sha256': sha(src_blur),
                     'source_sharp': str(src_gt.resolve()), 'source_sharp_sha256': sha(src_gt),
                     'destination_name': p.stem + src_gt.suffix.lower(), 'shape': list(x.shape)})
    return {'alignment': 'EXACT_RGB_PIXELS_EVERY_BLUR_FRAME_UNIQUE_CONTIGUOUS_MATCH',
            'raw_start_index': start, 'source_total_frames': len(bf),
            'local_frames': n, 'is_complete_original_video': start == 0 and n == len(bf), 'frames': rows}


def plan(inv_file, out, mapping_file=None):
    inv = read_json(inv_file)
    mapping = read_json(mapping_file) if mapping_file else {}
    videos = inv['mp4_files']
    records, errors, cache = [], [], {}
    for name, local in local_blur_sequences(inv['bsd_root']):
        spec = mapping.get(name, {})
        mp4s = ([str(Path(spec['mp4']).resolve())] if spec.get('mp4') else
                [p for p in videos if Path(p).stem in (name, name.replace('/', '_'), name.split('/')[-1])])
        if len(mp4s) != 1 or mp4s[0] not in videos:
            errors.append(f'{name}: need unique mp4 provenance; add explicit mapping JSON, not code edits')
            continue
        candidates = inv['original_test_sources']
        if spec.get('source_sharp_dir'):
            candidates = [s for s in candidates if s['sharp_dir'] == str(Path(spec['source_sharp_dir']).resolve())]
        matches = []
        for source in candidates:
            match = align_exact(local, source, cache)
            if match is not None:
                exposure = source['exposure']
                # Explicit exposure label must come from a recorded source document, not color/PSNR.
                if exposure == 'UNKNOWN' and spec.get('exposure_evidence'):
                    evidence = Path(spec['exposure_evidence'])
                    if not evidence.is_file():
                        raise FileNotFoundError(evidence)
                    exposure = spec.get('exposure', 'UNKNOWN')
                    match['exposure_evidence'] = {'path': str(evidence.resolve()), 'sha256': sha(evidence)}
                matches.append({**source, **match, 'exposure': exposure})
        if len(matches) != 1:
            errors.append(f'{name}: {len(matches)} exact source matches; do not guess an exposure, offset or color swap')
            continue
        rec = matches[0]
        rec.update(sequence=name, mp4=mp4s[0], mp4_sha256=sha(mp4s[0]))
        records.append(rec)
        print(f'MATCH {name}: {rec["sharp_dir"]} frames={len(local)}', flush=True)
    exposures = {s['exposure'] for s in records}
    if len(exposures) != 1 or not exposures <= {'1ms8ms', '2ms16ms', '3ms24ms'}:
        errors.append(f'BSD exposure ambiguous/mixed: {sorted(exposures)}; separate benchmark cohorts required')
    result = {'schema': 1, 'bsd_root': inv['bsd_root'], 'status': 'BLOCKED' if errors else 'READY_TO_COPY',
              'errors': errors, 'sequences': records,
              'note': 'MP4 is a source-name reference only; GT bytes come only from original Sharp images.'}
    result['plan_sha256'] = digest(result)
    write_json(out, result, exclusive=True)
    if errors:
        raise ValueError('GT restore plan BLOCKED; see errors in ' + str(out))
    return result


def apply(plan_file):
    p = read_json(plan_file)
    if p.get('plan_sha256') != digest({k: v for k, v in p.items() if k != 'plan_sha256'}):
        raise ValueError('Restore plan was changed')
    if p['status'] != 'READY_TO_COPY':
        raise ValueError('Restore plan is not ready')
    test = Path(p['bsd_root']).resolve() / 'test'
    dest = test / 'gt'
    # Validate the whole operation before creating/copying any image.
    for s in p['sequences']:
        if sha(s['mp4']) != s['mp4_sha256']:
            raise ValueError('Provenance video changed')
        for r in s['frames']:
            for role in ('local_blur', 'source_blur', 'source_sharp'):
                if sha(r[role]) != r[role + '_sha256']:
                    raise ValueError(f'Source changed after plan: {r[role]}')
            if pixel_sha(rgb(r['local_blur'])) != pixel_sha(rgb(r['source_blur'])):
                raise ValueError('Alignment is no longer exact')
            target = dest / s['sequence'] / r['destination_name']
            within(target, dest)
            if target.exists() and sha(target) != r['source_sharp_sha256']:
                raise ValueError(f'Existing GT differs; will NOT overwrite: {target}')
    if dest.exists():
        # Existing complete result may be verified idempotently; partial folders are not overwritten.
        for s in p['sequences']:
            for r in s['frames']:
                target = dest / s['sequence'] / r['destination_name']
                if not target.is_file() or sha(target) != r['source_sharp_sha256']:
                    raise ValueError('Existing gt is incomplete; retain it and report instead of overwriting')
        print('EXISTING_GT_VERIFIED; no writes')
        return
    pending = test / ('.gt_restore_pending_' + uuid.uuid4().hex)
    pending.mkdir()
    try:
        for s in p['sequences']:
            for r in s['frames']:
                q = pending / s['sequence'] / r['destination_name']
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(r['source_sharp'], q)
                if sha(q) != r['source_sharp_sha256']:
                    raise ValueError(f'Copy verification failed: {q}')
        write_json(pending / 'restore_provenance.json', p)
        if dest.exists():
            raise ValueError('Destination appeared concurrently; no overwrite')
        pending.rename(dest)
    except Exception:
        print(f'Incomplete staging retained for inspection: {pending}')
        raise
    print(f'GT_RESTORED={dest}; byte-identical copy; no decoding or recoloring')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest='command', required=True)
    p = subs.add_parser('inventory')
    p.add_argument('--bsd-root', required=True)
    p.add_argument('--raw-root', action='append', required=True)
    p.add_argument('--mp4-root', required=True)
    p.add_argument('--out', required=True)
    p = subs.add_parser('plan')
    p.add_argument('--inventory', required=True)
    p.add_argument('--mapping')
    p.add_argument('--out', required=True)
    p = subs.add_parser('apply')
    p.add_argument('--plan', required=True)
    a = ap.parse_args()
    if a.command == 'inventory':
        write_json(a.out, inventory(a.bsd_root, a.raw_root, a.mp4_root), exclusive=True)
    elif a.command == 'plan':
        plan(a.inventory, a.out, a.mapping)
    else:
        apply(a.plan)

if __name__ == '__main__':
    main()
