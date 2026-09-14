"""One RGB8 PSNR implementation for every producer; no crop, rescale, alignment or MP4."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from .data import rgb
from rtf_temporal.posttrain import write_json

PROTOCOL = 'rgb8-png-full-frame-v1: clip then round once to uint8; RGB MAX=255; crop_border=0; frame-mean dB'


def metric(pred, gt):
    if pred.dtype != np.uint8 or gt.dtype != np.uint8 or pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError('Expected same-shape RGB uint8 arrays')
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    psnr = float('inf') if mse == 0 else 10 * math.log10(255.0**2 / mse)
    return psnr, float(structural_similarity(pred, gt, channel_axis=2, data_range=255, win_size=7))


def safe_number(value):
    return 'inf' if math.isinf(value) and value > 0 else float(value)


def collect_sequence(seq, directory, *, input_baseline=False):
    expected = {f'{f["index"]:08d}.png' for f in seq['frames']}
    if not input_baseline:
        actual = {p.name for p in Path(directory).iterdir() if p.is_file()}
        if actual != expected:
            raise ValueError(f'Prediction frame set mismatch: missing={sorted(expected-actual)[:5]} extra={sorted(actual-expected)[:5]}')
    rows = []
    for f in seq['frames']:
        path = Path(f['blur']) if input_baseline else Path(directory) / f'{f["index"]:08d}.png'
        with Image.open(path) as im:
            if im.format != 'PNG' or im.mode != 'RGB':
                raise ValueError(f'Expected lossless RGB8 PNG, not JPEG/grayscale/palette/RGBA: {path}')
        pred, gt = rgb(path), rgb(f['gt'])
        if list(pred.shape) != f['shape']:
            raise ValueError(f'Wrong output size: {path}')
        psnr, ssim = metric(pred, gt)
        inp = rgb(f['blur'])
        rows.append({'sequence': seq['name'], 'index': f['index'], 'original_frame_id': f['original_frame_id'],
                     'psnr': psnr, 'ssim': ssim, 'mean_abs_change_from_input': float(np.abs(pred.astype(float)-inp).mean())})
    return rows


def score(dataset, predictions, out, *, input_baseline=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if not input_baseline:
        expected = {s['name'] for s in dataset['sequences']}
        actual = {p.name for p in Path(predictions).iterdir() if p.is_dir()}
        if actual != expected:
            raise ValueError('Prediction sequence set differs from frozen full test set')
    rows, sequences = [], []
    for seq in dataset['sequences']:
        part = collect_sequence(seq, Path(predictions) / seq['name'], input_baseline=input_baseline)
        rows.extend(part)
        sequences.append({'sequence': seq['name'], 'frames': len(part),
            'psnr': safe_number(float(np.mean([r['psnr'] for r in part]))),
            'ssim': float(np.mean([r['ssim'] for r in part]))})
    if len(rows) != dataset['frames']:
        raise ValueError('Cannot publish incomplete test results')
    with (out / 'per_frame.csv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {'protocol': PROTOCOL, 'dataset': dataset['domain'], 'variant': dataset['spec']['variant'],
        'frames': len(rows), 'sequences': len(sequences),
        'psnr_frame_mean': safe_number(float(np.mean([r['psnr'] for r in rows]))),
        'psnr_sequence_mean': safe_number(float(np.mean([float(s['psnr']) for s in sequences]))),
        'ssim_frame_mean': float(np.mean([r['ssim'] for r in rows])),
        'mean_abs_change_from_input': float(np.mean([r['mean_abs_change_from_input'] for r in rows]))}
    write_json(out / 'per_sequence.json', sequences); write_json(out / 'metrics.json', summary)
    return summary


def reference_check(dataset, predictions, mapping):
    """Optional official-output anchor with EXPLICIT original frame IDs, never positional zip."""
    index = {(s['name'], f['original_frame_id']): f for s in dataset['sequences'] for f in s['frames']}
    seen, errors, deltas = set(), [], []
    for row in mapping:
        key = row['sequence'], row['original_frame_id']
        if key in seen or key not in index:
            raise ValueError('Invalid/duplicate reference frame mapping')
        seen.add(key); f = index[key]
        pred = rgb(Path(predictions) / key[0] / f'{f["index"]:08d}.png')
        ref, gt = rgb(row['official_prediction']), rgb(f['gt'])
        if pred.shape != ref.shape:
            raise ValueError('Official reference geometry differs; no resizing to hide protocol mismatch')
        errors.append(float(np.abs(pred.astype(float)-ref).max()))
        a, _ = metric(pred, gt); b, _ = metric(ref, gt)
        deltas.append(0.0 if a == b else abs(a-b))
    if len(seen) < 32 or len({k[0] for k in seen}) < 2:
        raise ValueError('Reference anchor needs >=32 unique mapped frames across >=2 sequences')
    return {'frames': len(seen), 'sequences': len({k[0] for k in seen}),
            'max_pixel_difference_255': max(errors), 'mean_abs_psnr_delta': float(np.mean(deltas)),
            'passes': max(errors) <= 2 and float(np.mean(deltas)) <= .05}
