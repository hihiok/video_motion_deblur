"""One scorer for every method; all native test frames, FP32 inference by default.

Official/paper protocols are not inferred from a method name. This is a declared
local all-frame benchmark; paper results belong in a separate reference table.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .data import (blob, checked_frame, digest, load_manifest, read_json, rgb, sha,
                   within, write_json)

MODEL_BLOB = 'de3b8032940bd96413fc685ce15389de3e899a90'
METRIC = {'space': 'RGB', 'range': [0, 1], 'spatial_crop': 0,
          'alignment': 'NONE', 'brightness_fit': False, 'color_swap': False,
          'primary': 'PSNR_RGB8_per_frame_arithmetic_mean',
          'secondary': 'PSNR_RGB_float_per_frame_mean_when_available',
          'quantization': 'clip_0_1_then_numpy_rint_times_255',
          'boundaries': 'ALL original frames exactly once'}


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quantize(a):
    if not np.isfinite(a).all():
        raise ValueError('NaN/Inf output; not a valid benchmark result')
    return np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)


def psnr(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid metric input')
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float('inf') if mse == 0 else -10 * math.log10(mse)


def number(n):
    return 'inf' if math.isinf(n) else float(n)


def average(vals):
    vals = list(vals)
    return number(sum(map(float, vals)) / len(vals))


def reflect(i, n):
    if n == 1:
        return 0
    i %= 2 * n - 2
    return i if i < n else 2 * n - 2 - i


def model_info(args):
    import torch
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.method == 'nano':
        if blob(args.model_file) != MODEL_BLOB:
            raise ValueError('Nano model source differs from reviewed source; synchronize, do not override hash')
        if ck.get('architecture') != 'NanoVNRNAFNetRGB':
            raise ValueError('Wrong checkpoint architecture')
        model = import_file(args.model_file, 'bench_nano').NanoVNRNAFNetRGB(12, grad_checkpoint=False)
        model.load_state_dict(ck['model'], strict=True)
        if sum(p.numel() for p in model.parameters()) != 414923:
            raise ValueError('Wrong Nano parameter count')
        if ck.get('step') != 125000:
            raise ValueError('This benchmark freezes the selected 125k checkpoint; no test-set retuning')
    else:
        arch = Path(args.shift_repo) / 'basicsr/models/archs/gshift_deblur2.py'
        if not arch.is_file():
            raise FileNotFoundError(arch)
        # Same constructor as official inference/test_deblur_small.py; no network rewriting.
        args.model_file = str(arch)
        model = import_file(arch, 'bench_shift_small').GShiftNet(future_frames=2, past_frames=2)
        if not isinstance(ck, dict) or 'params' not in ck:
            raise ValueError('Require official Shift-Net Ours-s params checkpoint; no silent key conversion')
        model.load_state_dict(ck['params'], strict=True)
        count = sum(p.numel() for p in model.parameters())
        if not 3_000_000 < count < 6_000_000:
            raise ValueError(f'Unexpected Ours-s size: {count}; stop rather than load Ours+')
    info = {'architecture_file': str(Path(args.model_file).resolve()),
            'architecture_sha256': sha(args.model_file), 'architecture_git_blob': blob(args.model_file),
            'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': sha(args.checkpoint),
            'checkpoint_step': ck.get('step') if isinstance(ck, dict) else None,
            'params': sum(p.numel() for p in model.parameters()), 'strict_load': True}
    return model, info


def predict_sequence(args, frames, model, device):
    import torch
    import torch.nn.functional as F
    n = len(frames)
    state = None
    if args.context == 16:
        # Same 16 input frames and same target for every method; reset per target.
        for target in range(n):
            ids = [reflect(i, n) for i in range(target-8, target+8)]
            arr = np.stack([checked_frame(frames[i], 'lq') for i in ids]).astype(np.float32)/255
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            h, w = arr.shape[1:3]
            if args.method == 'shift-small':
                ph, pw = (-h) % 4, (-w) % 4
                if ph or pw:
                    x = F.pad(x.reshape(-1, 3, h, w), (0, pw, 0, ph), mode='replicate').reshape(1, 16, 3, h+ph, w+pw)
            if args.precision == 'fp16':
                x = x.half()
            with torch.inference_mode():
                if args.method == 'nano':
                    all_y, _ = model(x, prev_forward_feat=None)
                    y = all_y[0, 8]
                else:
                    all_y = model(x.contiguous())
                    if all_y.ndim == 5 and all_y.shape[0] == 1:
                        all_y = all_y[0]
                    if all_y.ndim != 4 or all_y.shape[0] != 12:
                        raise ValueError('Shift-Net fixed16 must return 12 outputs')
                    y = all_y[6]  # original input index 8 minus two leading halo frames
            pred = y[:, :h, :w].float().permute(1, 2, 0).cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError('NaN/Inf model output')
            yield target, np.clip(pred, 0, 1)
            if (target+1) % 25 == 0 or target+1 == n:
                print(f'fixed16 targets={target+1}/{n}', flush=True)
        return
    chunk = args.chunk if args.method == 'nano' else args.one_len
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        ids = list(range(start, end)) if args.method == 'nano' else [reflect(i, n) for i in range(start-2, end+2)]
        arr = np.stack([checked_frame(frames[i], 'lq') for i in ids]).astype(np.float32) / 255
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        h, w = arr.shape[1:3]
        if args.method == 'shift-small':
            ph, pw = (-h) % 4, (-w) % 4
            if ph or pw:
                x = F.pad(x.reshape(-1, 3, h, w), (0, pw, 0, ph), mode='replicate').reshape(1, len(ids), 3, h+ph, w+pw)
        if args.precision == 'fp16':
            x = x.half()
        with torch.inference_mode():
            if args.method == 'nano':
                y, state = model(x, prev_forward_feat=state)
                state = state.detach()
                y = y[0]
            else:
                y = model(x.contiguous())
                if y.ndim == 5 and y.shape[0] == 1:
                    y = y[0]
        if y.ndim != 4 or tuple(y.shape[:2]) != (end-start, 3):
            raise ValueError(f'Unexpected output shape {tuple(y.shape)} for original frames {start}:{end}')
        result = y[:, :, :h, :w].float().permute(0, 2, 3, 1).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('NaN/Inf model output')
        for j in range(end-start):
            yield start+j, np.clip(result[j], 0, 1)
        print(f'frames={end}/{n}', flush=True)


def seq_key(seq):
    # Preserve readable paths and reject unsafe IDs in an edited/untrusted manifest.
    key = Path(seq['dataset']) / seq['sequence']
    if key.is_absolute() or '..' in key.parts:
        raise ValueError('Unsafe sequence ID')
    return key


def run(args):
    import torch
    m = load_manifest(args.manifest)
    meta = read_json(args.metadata)
    for key in ('training_data', 'checkpoint_origin', 'checkpoint_selection'):
        if not meta.get(key):
            raise ValueError(f'Metadata required: {key}')
    if args.precision == 'fp16' and args.device == 'cpu':
        raise ValueError('FP16 CPU benchmark is not supported')
    if args.chunk < 1 or args.one_len < 1:
        raise ValueError('Invalid temporal chunk')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    model, info = model_info(args)
    model = model.to(device).eval()
    if args.precision == 'fp16':
        model = model.half()
    info.update(torch=torch.__version__, precision=args.precision, tf32=False,
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU_TEST_ONLY')
    protocol = ({'chunk': args.chunk, 'state_carry': 'forward within sequence only',
                 'backward': 'reset each nonoverlap chunk'} if args.method == 'nano' else
                {'one_len': args.one_len, 'past_halo': 2, 'future_halo': 2,
                 'temporal_padding': 'reflect at video boundaries', 'tail': 'process all',
                 'note': 'ALL-frame wrapper differs from official script head/tail dropping'})
    if args.context == 16:
        protocol = {'context': 16, 'target_input_index': 8, 'past': 8, 'future': 7,
                    'temporal_padding': 'reflect', 'state': 'reset for every target',
                    'targets': 'all original frames exactly once'}
    signature = {'schema': 1, 'method': args.method, 'manifest_sha256': m['manifest_sha256'],
                 'metadata': meta, 'model': info, 'protocol': protocol, 'metric': METRIC,
                 'track': ('MATCHED_CONTEXT16_SAME_TARGETS_NOT_EQUAL_TRAINING' if args.context else
                           'ALL_FRAME_CHECKPOINT_TRACK_NOT_EQUAL_CONTEXT_OR_EQUAL_TRAINING'),
                 'paper_comparable': False,
                 'evaluation_code_sha256': sha(__file__),
                 'data_code_sha256': sha(Path(__file__).with_name('data.py'))}
    sig_hash = digest(signature)
    out = Path(args.out)
    if out.exists():
        if not args.resume or not (out / 'run.json').is_file() or read_json(out / 'run.json') != signature:
            raise ValueError('Existing run differs or --resume missing; never overwrite another run')
    else:
        out.mkdir(parents=True)
        write_json(out / 'run.json', signature, exclusive=True)
        shutil.copy2(args.model_file, out / 'model_source_snapshot.py')
    completed = []
    for seq in m['sequences']:
        final = out / 'sequences' / seq_key(seq)
        if final.exists():
            result = read_json(final / 'sequence.json')
            if result.get('signature') != sig_hash:
                raise ValueError('Sequence signature mismatch')
            for row in result['frames']:
                if sha(final / row['output_file']) != row['output_sha256']:
                    raise ValueError('Saved prediction modified')
            # Resume never carries hidden across sequences. Input/GT mutation is also checked.
            for row in seq['frames']:
                checked_frame(row, 'lq'); checked_frame(row, 'gt')
            completed.append(result)
            continue
        pending = out / ('.pending_' + uuid.uuid4().hex)
        (pending / 'pred').mkdir(parents=True)
        print(f'BEGIN {seq["dataset"]}/{seq["sequence"]} native={seq["frames"][0]["shape"]}', flush=True)
        rows = []
        start_time = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with (pending / 'per_frame.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'frame', 'input_psnr', 'output_psnr_rgb8', 'output_psnr_float'])
            writer.writeheader()
            for i, pred in predict_sequence(args, seq['frames'], model, device):
                r = seq['frames'][i]
                lq = checked_frame(r, 'lq').astype(np.float64) / 255
                gt = checked_frame(r, 'gt').astype(np.float64) / 255
                if pred.shape != gt.shape:
                    raise ValueError('Prediction/GT shape mismatch; no metric-time resize')
                u8 = quantize(pred)
                filename = f'pred/{i:08d}.png'
                Image.fromarray(u8).save(pending / filename)
                metric = {'index': i, 'frame': r['frame'], 'input_psnr': number(psnr(lq, gt)),
                          'output_psnr_rgb8': number(psnr(u8.astype(np.float64)/255, gt)),
                          'output_psnr_float': number(psnr(pred, gt))}
                writer.writerow(metric); f.flush()
                rows.append({**metric, 'output_file': filename, 'output_sha256': sha(pending / filename)})
                if i in {0, len(seq['frames'])//2, len(seq['frames'])-1}:
                    preview = np.concatenate((np.rint(lq*255).astype(np.uint8), u8, np.rint(gt*255).astype(np.uint8)), axis=1)
                    Image.fromarray(preview).save(pending / f'preview_{i:08d}.png')
        if len(rows) != len(seq['frames']):
            raise ValueError('Incomplete output frame coverage')
        result = {'dataset': seq['dataset'], 'sequence': seq['sequence'], 'signature': sig_hash,
                  'frames': rows, 'seconds': time.time()-start_time,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None}
        write_json(pending / 'sequence.json', result)
        final.parent.mkdir(parents=True, exist_ok=True)
        pending.rename(final)
        completed.append(result)
        summarize(completed, out / 'summary.partial.json', signature, complete=False)
    summarize(completed, out / 'summary.json', signature, complete=True)


def summarize(results, path, signature, complete):
    groups = defaultdict(list)
    per_video = []
    for s in results:
        rows = s['frames']
        summary = {'dataset': s['dataset'], 'sequence': s['sequence'], 'frames': len(rows)}
        for key in ('input_psnr', 'output_psnr_rgb8', 'output_psnr_float'):
            values = [r[key] for r in rows if r.get(key) is not None]
            summary[key] = average(values) if len(values) == len(rows) else None
        per_video.append(summary)
        groups[s['dataset']].append(summary)
    datasets = {}
    for family, videos in groups.items():
        entry = {'videos': len(videos), 'frames': sum(v['frames'] for v in videos)}
        for key in ('input_psnr', 'output_psnr_rgb8', 'output_psnr_float'):
            if all(v[key] is not None for v in videos):
                entry[key + '_frame_mean'] = number(sum(float(v[key])*v['frames'] for v in videos)/entry['frames'])
                entry[key + '_video_mean'] = average(v[key] for v in videos)
        datasets[family] = entry
    report = {'status': 'COMPLETE' if complete else 'PARTIAL', 'run': signature,
              'datasets': datasets, 'videos': per_video}
    write_json(path, report)
    if complete:
        with Path(path).with_suffix('.csv').open('w', newline='') as f:
            fields = ['dataset', 'sequence', 'frames', 'input_psnr', 'output_psnr_rgb8', 'output_psnr_float']
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(per_video)
        print('COMPLETE: ' + str(path), flush=True)
    return report


def score_external(args):
    """Score any method's native RGB PNG outputs using an explicit frame-index file."""
    m = load_manifest(args.manifest)
    index = read_json(args.index)
    if index.get('manifest_sha256') != m['manifest_sha256']:
        raise ValueError('External output manifest mismatch')
    for k in ('method', 'training_data', 'checkpoint_sha256', 'protocol', 'precision', 'source_code_commit'):
        if not index.get(k):
            raise ValueError(f'External provenance field missing: {k}')
    keyed = {}
    for r in index['outputs']:
        key = (r['dataset'], r['sequence'], r['frame'])
        if key in keyed:
            raise ValueError('Duplicate external output')
        keyed[key] = r
    results = []
    for s in m['sequences']:
        rows = []
        for r in s['frames']:
            key = (s['dataset'], s['sequence'], r['frame'])
            if key not in keyed:
                raise ValueError(f'Missing output: {key}; no partial leaderboard')
            pred_row = keyed.pop(key)
            path = Path(pred_row['path'])
            if path.suffix.lower() != '.png' or sha(path) != pred_row['sha256']:
                raise ValueError('Require hashed lossless PNG predictions, never MP4/JPEG')
            pred = rgb(path).astype(np.float64)/255
            x, gt = checked_frame(r, 'lq')/255.0, checked_frame(r, 'gt')/255.0
            rows.append({'index': r['index'], 'frame': r['frame'], 'input_psnr': number(psnr(x, gt)),
                         'output_psnr_rgb8': number(psnr(pred, gt)), 'output_psnr_float': None})
        results.append({'dataset': s['dataset'], 'sequence': s['sequence'], 'frames': rows})
    if keyed:
        raise ValueError('Extra unknown outputs in external index')
    signature = {'method': index['method'], 'manifest_sha256': m['manifest_sha256'],
                 'metric': METRIC, 'external_provenance': {k: v for k, v in index.items() if k != 'outputs'},
                 'paper_comparable': False}
    if Path(args.out).exists():
        raise ValueError('Refusing to overwrite score')
    summarize(results, args.out, signature, complete=True)
    write_json(Path(args.out).with_name(Path(args.out).stem + '_per_frame.json'), results, exclusive=True)


def compare(args):
    reports = [read_json(p) for p in args.reports]
    first = reports[0]
    def coverage(r):
        return sorted((v['dataset'], v['sequence'], v['frames']) for v in r['videos'])
    for r in reports:
        if r['status'] != 'COMPLETE' or r['run']['manifest_sha256'] != first['run']['manifest_sha256']:
            raise ValueError('Incomplete or different test manifests; refuse a misleading leaderboard')
        if r['run']['metric'] != first['run']['metric'] or coverage(r) != coverage(first):
            raise ValueError('Different metric or frame coverage')
    tracks = [r['run'].get('track', 'EXTERNAL_DECLARED') for r in reports]
    matched = ['MATCHED_CONTEXT16' in t for t in tracks]
    if any(matched):
        if not all(matched) or any(r['run']['protocol'] != first['run']['protocol'] for r in reports):
            raise ValueError('Cannot mix fixed-context16 and unmatched temporal protocols')
    precisions = {r['run'].get('model', {}).get('precision') or
                  r['run'].get('external_provenance', {}).get('precision') for r in reports}
    if len(precisions) != 1:
        raise ValueError('Mixed inference precision; keep these in separate tables')
    # Temporal protocols/training data are retained; not an architecture-only fairness claim.
    rows = [{'method': r['run']['method'], 'run': r['run'], 'results': r['datasets']} for r in reports]
    write_json(args.out, {'track': ('MATCHED_CONTEXT16_SAME_TEST_SAME_RGB8_METRIC' if all(matched) else
                                    'DECLARED_PROTOCOL_SAME_TEST_SAME_RGB8_METRIC'),
                         'equal_training': False, 'equal_temporal_context': all(matched),
                         'methods': rows}, exclusive=True)
    for r in reports:
        for family, d in r['datasets'].items():
            print(r['run']['method'], family, d['output_psnr_rgb8_frame_mean'])


def make_external_index(args):
    m = load_manifest(args.manifest)
    metadata = read_json(args.metadata)
    output = {**metadata, 'manifest_sha256': m['manifest_sha256'], 'outputs': []}
    root = Path(args.prediction_root).resolve()
    for s in m['sequences']:
        for r in s['frames']:
            p = within(root / seq_key(s) / (r['frame'] + '.png'), root)
            if not p.is_file():
                raise FileNotFoundError(f'Missing exact named output: {p}; no index-only guessing')
            output['outputs'].append({'dataset': s['dataset'], 'sequence': s['sequence'],
                'frame': r['frame'], 'path': str(p), 'sha256': sha(p)})
    write_json(args.out, output, exclusive=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('run')
    q.add_argument('--manifest', required=True); q.add_argument('--metadata', required=True)
    q.add_argument('--method', choices=('nano', 'shift-small'), required=True)
    q.add_argument('--checkpoint', required=True); q.add_argument('--out', required=True)
    q.add_argument('--model-file', default=str(Path(__file__).resolve().parents[1]/'nanovsr_deblur/models/network_nanovnr_nafnet_rgb.py'))
    q.add_argument('--shift-repo'); q.add_argument('--device', default='cuda:0')
    q.add_argument('--precision', choices=('fp32', 'fp16'), default='fp32')
    q.add_argument('--chunk', type=int, default=15); q.add_argument('--one-len', type=int, default=16)
    q.add_argument('--context', type=int, choices=(0, 16), default=0,
                   help='0: declared deployment protocol; 16: strict same input window per target')
    q.add_argument('--resume', action='store_true')
    q = sub.add_parser('score')
    q.add_argument('--manifest', required=True); q.add_argument('--index', required=True); q.add_argument('--out', required=True)
    q = sub.add_parser('index')
    q.add_argument('--manifest', required=True); q.add_argument('--metadata', required=True)
    q.add_argument('--prediction-root', required=True); q.add_argument('--out', required=True)
    q = sub.add_parser('compare')
    q.add_argument('--reports', nargs='+', required=True); q.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'run':
        if a.method == 'shift-small' and not a.shift_repo:
            p.error('--shift-repo required for shift-small')
        run(a)
    elif a.command == 'score':
        score_external(a)
    elif a.command == 'index':
        make_external_index(a)
    else:
        compare(a)

if __name__ == '__main__':
    main()
