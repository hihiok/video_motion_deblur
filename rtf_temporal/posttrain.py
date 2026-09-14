"""Post-training audit and evaluation only; never changes a training run.

Official model source is vendored in upstream_reference.py from Git blob
c314f62633ebdbc5c5614cfc42f04e495d69fa57 (ReaganWu/RT-Focuser).
Its MIT license is in third_party/RT_FOCUSER_LICENSE.
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from rtf_t6.checkpoint import unwrap_state_dict
from rtf_t6.datasets import read_rgb, IMAGE_EXTENSIONS
from .data import sha256
from .flow import is_cut, pair_flow, warp
from .model import TemporalRTFocuser
from .upstream_reference import RT_Focuser_Standard


OFFICIAL_SHA = '6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def run_command(command):
    return subprocess.check_output(command, text=True)


def numeric_frames(directory):
    paths = [p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    def number(path):
        match = re.search(r'(\d+)$', path.stem)
        if not match:
            raise ValueError(f'No chronological numeric suffix: {path}')
        return int(match.group(1))
    paths.sort(key=number)
    if not paths or any(number(b) != number(a) + 1 for a, b in zip(paths, paths[1:])):
        raise ValueError(f'Empty, duplicate or nonconsecutive frame IDs: {directory}')
    return paths


def gopro_test(root, variants=('blur_gamma', 'blur'), expected_frames=1111, expected_sequences=11):
    """Never falls back to train/val; never groups acquisition IDs across official splits."""
    root = Path(root)
    test = root if root.name.lower() == 'test' else root / 'test'
    if not test.is_dir():
        raise FileNotFoundError(f'Official GoPro test directory missing: {test}')
    sequences = sorted(p for p in test.iterdir() if p.is_dir())
    if len(sequences) != expected_sequences:
        raise ValueError(f'Expected {expected_sequences} test sequences, found {len(sequences)}')
    available, absent = {}, []
    for variant in variants:
        exists = [(s / variant).is_dir() for s in sequences]
        if not any(exists):
            absent.append(variant)
            continue
        if not all(exists):
            raise ValueError(f'Partially missing {variant}: cannot compare on unequal image sets')
        records = []
        for seq in sequences:
            blur, gt = numeric_frames(seq / variant), numeric_frames(seq / 'sharp')
            if [p.stem for p in blur] != [p.stem for p in gt]:
                raise ValueError(f'Pairing mismatch in {seq}/{variant}')
            for a, b in zip(blur, gt):
                with Image.open(a) as im:
                    sa = im.size
                with Image.open(b) as im:
                    sb = im.size
                if sa != sb:
                    raise ValueError(f'Paired dimensions differ: {a}, {b}')
            records.append({'name': seq.name, 'blur': blur, 'gt': gt})
        count = sum(len(r['blur']) for r in records)
        if count != expected_frames:
            raise ValueError(f'Expected {expected_frames} test images, found {count} in {variant}')
        available[variant] = records
    if not available:
        raise FileNotFoundError('Neither requested GoPro blur variant is present')
    return available, absent


def verify_holdout_no_exact_test_overlap(manifest, records):
    """GoPro acquisition names may legitimately occur in both official splits.

    Only exact frame identity/path/content overlap with fine-tuning data is blocked.
    """
    train = [r for r in manifest['train'] if r['domain'] == 'gopro']
    train_ids = {(r['name'], Path(p).stem) for r in train for p in r['gt']}
    test_ids = {(r['name'], p.stem) for r in records for p in r['gt']}
    if train_ids & test_ids:
        raise ValueError('Exact sequence/frame IDs overlap with fine-tuning train')
    hashes = {sha256(p) for r in train for p in r['gt']}
    overlap = [str(p) for r in records for p in r['gt'] if sha256(p) in hashes]
    if overlap:
        raise ValueError(f'Exact GT file contents overlap train/test, examples: {overlap[:5]}')


def checkpoint_audit(run, official, output):
    run, official, output = Path(run).resolve(), Path(official).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError('Audit output must be outside the training run')
    if sha256(official) != OFFICIAL_SHA:
        raise ValueError('Official checkpoint SHA256 differs from the approved checkpoint')
    snapshots = output / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    report = {'training_run': str(run), 'checkpoints': {}}
    for label, source in [('official', official), ('best_stable', run / 'checkpoints/best_stable.pth')]:
        digest = sha256(source)
        dest = snapshots / f'{label}_{digest[:12]}.pth'
        if not dest.exists():
            shutil.copy2(source, dest)
        if sha256(dest) != digest or sha256(source) != digest:
            raise RuntimeError('Checkpoint changed during snapshot; retry after writer completes')
        report['checkpoints'][label] = {'source': str(source), 'snapshot': str(dest), 'sha256': digest}
    best = torch.load(report['checkpoints']['best_stable']['snapshot'], map_location='cpu', weights_only=False)
    if best['config']['protocol'] != 'rtfocuser_causal_temporal_finetune_v1':
        raise ValueError('Not a temporal fine-tuning v1 checkpoint')
    report['best_update'] = int(best['update'])
    report['training_git_commit'] = best['git_commit']
    report['baseline'] = best['baseline']
    vp = run / f'validation_{report["best_update"]:06d}.json'
    report['best_validation'] = json.loads(vp.read_text()) if vp.exists() else None
    manifest = json.loads((run / 'manifest.json').read_text())
    if sha256(run / 'manifest.json') != best['config']['manifest_sha256']:
        raise ValueError('Manifest differs from checkpoint training provenance')
    samples = [r for r in manifest['train'] + manifest['val'] if r['domain'] == 'gopro']
    report['gopro_training_input_parents'] = sorted({str(Path(r['blur'][0]).parent) for r in samples})
    report['selection_note'] = 'Only best_stable is evaluated; no latest substitution or temporal-module ablation'
    write_json(output / 'checkpoint_audit.json', report)
    return report, manifest


class Models:
    def __init__(self, audit, device):
        self.device = torch.device(device)
        # FP32 reference: TF32 is disabled as well as autocast.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.official = RT_Focuser_Standard().to(self.device).eval()
        path = audit['checkpoints']['official']['snapshot']
        self.official.load_state_dict(unwrap_state_dict(torch.load(path, map_location='cpu', weights_only=False)), strict=True)
        payload = torch.load(audit['checkpoints']['best_stable']['snapshot'], map_location='cpu', weights_only=False)
        self.temporal = TemporalRTFocuser(**{**payload['config']['model'], 'activation_checkpointing': False}).to(self.device).eval()
        self.temporal.load_state_dict(payload['model'], strict=True)
        self.cut_threshold = payload['config']['train']['cut_threshold']

    @torch.inference_mode()
    def original(self, inp):
        h, w = inp.shape[-2:]
        padded = torch.nn.functional.pad(inp, (0, (-w) % 16, 0, (-h) % 16), mode='replicate')
        return self.official(padded)[..., :h, :w]

    @torch.inference_mode()
    def verify_wrapper(self, inp, audit):
        wrapper = TemporalRTFocuser(activation_checkpointing=False).to(self.device).eval()
        wrapper.load_official(audit['checkpoints']['official']['snapshot'])
        ref = self.original(inp)
        actual, _ = wrapper.step(inp)
        error = float((ref - actual).abs().max())
        del wrapper
        if error > 1e-5:
            raise RuntimeError(f'Official/zero-initialized wrapper mismatch: max_abs={error}')
        return {'max_abs_error': error, 'tolerance': 1e-5, 'fp32': True,
                'purpose': 'Implementation identity check, not a trained-model ablation'}


def image_metrics(pred, gt):
    p, g = pred.float().cpu().numpy(), gt.float().cpu().numpy()
    mse = float(np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2))
    value = -10 * math.log10(max(mse, 1e-12))
    ss = structural_similarity(g.transpose(1, 2, 0), p.transpose(1, 2, 0), channel_axis=2, data_range=1.0)
    return {'psnr': value, 'ssim': float(ss), 'mse': mse}


def full_gopro(models, available, output):
    output = Path(output)
    result = {'protocol': 'Full official GoPro test; native RGB [0,1]; no crop/resize; FP32; TF32 off; per-frame PSNR mean',
              'paper_comparison': '30.67 dB is a published reference, not a pass/fail target; public official validation uses random 256 crops',
              'variants': {}}
    for variant, records in available.items():
        rows = []
        temporal_sums = {m: [0., 0., 0.] for m in ('input', 'official', 'best_stable')}
        for record in records:
            state, prev_inp, previous = None, None, None
            for index, (bp, gp) in enumerate(zip(record['blur'], record['gt'])):
                inp = read_rgb(bp)[None].to(models.device)
                gt = read_rgb(gp)[None].to(models.device)
                inp_np = inp[0].permute(1, 2, 0).cpu().numpy()
                gt_np = gt[0].permute(1, 2, 0).cpu().numpy()
                reset = prev_inp is None or prev_inp.shape != inp_np.shape or is_cut(prev_inp, inp_np, models.cut_threshold)
                if reset:
                    state = None
                with torch.inference_mode():
                    orig = models.original(inp)
                    fine, state = models.temporal.step(inp, state)
                preds = {'input': inp, 'official': orig, 'best_stable': fine}
                for name, pred in preds.items():
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError(f'Nonfinite output: {variant}/{record["name"]}/{bp.name}/{name}')
                    rows.append({'sequence': record['name'], 'frame': bp.stem, 'model': name,
                                 'state_reset': reset, **image_metrics(pred[0], gt[0])})
                if previous is not None and not reset:
                    flow, mask = pair_flow(previous['gt_np'], gt_np)
                    flow, mask = flow[None].to(models.device), mask[None].to(models.device)
                    for name, pred in preds.items():
                        delta = pred - warp(previous[name], flow) - gt + warp(previous['gt'], flow)
                        sums = temporal_sums[name]
                        sums[0] += float((delta.abs() * mask).sum())
                        sums[1] += float(mask.sum()) * 3
                        sums[2] += mask.numel() * 3
                previous = {**preds, 'gt': gt, 'gt_np': gt_np}
                prev_inp = inp_np
                if (index + 1) % 50 == 0:
                    print(f'GOPRO {variant}/{record["name"]} {index+1}/{len(record["blur"])}', flush=True)
            print(f'GOPRO_SEQUENCE_COMPLETE {variant}/{record["name"]}', flush=True)
        summary = {}
        for name in temporal_sums:
            selected = [r for r in rows if r['model'] == name]
            n, d, possible = temporal_sums[name]
            summary[name] = {'frames': len(selected), 'psnr': float(np.mean([r['psnr'] for r in selected])),
                             'ssim': float(np.mean([r['ssim'] for r in selected])),
                             'aligned_temporal_l1': n / d if d else None,
                             'valid_flow_fraction': d / possible if possible else None}
        result['variants'][variant] = summary
        with (output / f'gopro_{variant}_per_frame.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        seqrows = []
        for record in records:
            for name in temporal_sums:
                selected = [r for r in rows if r['model'] == name and r['sequence'] == record['name']]
                seqrows.append({'sequence': record['name'], 'model': name, 'frames': len(selected),
                                'psnr': float(np.mean([r['psnr'] for r in selected])),
                                'ssim': float(np.mean([r['ssim'] for r in selected]))})
        write_json(output / f'gopro_{variant}_per_sequence.json', seqrows)
        write_json(output / 'gopro_full_test.json', result)
    return result


def probe_video(path):
    raw = json.loads(run_command(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_streams', '-show_frames', '-show_entries',
        'stream=width,height,avg_frame_rate,r_frame_rate,start_time,duration,pix_fmt,color_transfer:frame=best_effort_timestamp_time,pkt_duration_time',
        '-of', 'json', str(path)]))
    frames, streams = raw.get('frames', []), raw.get('streams', [])
    if len(streams) != 1 or not frames:
        raise ValueError(f'No decoded video frames: {path}')
    pts = [float(f['best_effort_timestamp_time']) for f in frames]
    differences = np.diff(pts)
    if len(differences) and np.any(differences <= 0):
        raise ValueError('Non-increasing source timestamps; needs explicit timeline handling')
    fps = Fraction(streams[0]['avg_frame_rate'])
    if fps <= 0:
        raise ValueError('Invalid source average frame rate')
    duration_last = float(frames[-1].get('pkt_duration_time', 0))
    if duration_last <= 0 and streams[0].get('duration') not in (None, 'N/A'):
        duration_last = float(streams[0].get('start_time', pts[0])) + float(streams[0]['duration']) - pts[-1]
    if duration_last <= 0:
        duration_last = float(np.median(differences)) if len(differences) else 1 / float(fps)
    return {'frames': len(frames), 'width': streams[0]['width'], 'height': streams[0]['height'],
            'avg_frame_rate': str(fps), 'pts': pts,
            'pixel_format': streams[0].get('pix_fmt'), 'color_transfer': streams[0].get('color_transfer'),
            'durations': differences.tolist() + [duration_last],
            'variable_frame_rate': bool(len(differences) and np.max(np.abs(differences - 1 / float(fps))) > .001)}


def safe_concat_path(path):
    return "'" + str(Path(path).resolve()).replace("'", "'\\''") + "'"


def encode_frames(directory, target, info):
    directory, target = Path(directory), Path(target)
    frames = sorted(directory.glob('*.png'))
    if len(frames) != info['frames']:
        raise ValueError(f'Frame count mismatch before encode: {directory}')
    common = ['ffmpeg', '-nostdin', '-v', 'error', '-n']
    if info['variable_frame_rate']:
        listing = target.with_suffix('.ffconcat')
        lines = ['ffconcat version 1.0']
        for frame, duration in zip(frames, info['durations']):
            lines.extend([f'file {safe_concat_path(frame)}', 'option framerate 1000000', f'duration {duration:.9f}'])
        # The repeated tail establishes the last duration; frame limit excludes its image.
        lines.extend([f'file {safe_concat_path(frames[-1])}', 'option framerate 1000000'])
        listing.write_text('\n'.join(lines) + '\n')
        inputs = ['-safe', '0', '-f', 'concat', '-i', str(listing), '-fps_mode', 'vfr', '-enc_time_base', '1:1000000']
    else:
        inputs = ['-framerate', info['avg_frame_rate'], '-i', str(directory / '%08d.png')]
    # With B frames disabled, packet N is also display frame N. Preserve the last
    # hold duration explicitly; a high-rate PNG demuxer otherwise makes it 1 us.
    tail_duration = f"setts=duration='if(eq(N,{info['frames']-1}),{info['durations'][-1]:.9f}/TB,DURATION)'"
    subprocess.run(common + inputs + ['-frames:v', str(info['frames']), '-an', '-c:v', 'libx264',
        '-bf', '0', '-bsf:v', tail_duration,
        '-crf', '16', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-threads', '2', '-video_track_timescale', '1000000', str(target)], check=True)
    check = probe_video(target)
    if check['frames'] != info['frames']:
        raise RuntimeError('Encoded output frame count mismatch')
    timing_error = max(abs((a - check['pts'][0]) - (b - info['pts'][0])) for a, b in zip(check['pts'], info['pts']))
    if timing_error > .002:
        raise RuntimeError(f'Output timing drift {timing_error}s; preserve PNGs and report')
    duration_error = abs(sum(check['durations']) - sum(info['durations']))
    if duration_error > .002:
        raise RuntimeError(f'Output duration drift {duration_error}s; preserve PNGs and report')
    return {'frames': check['frames'], 'max_timestamp_error_seconds': timing_error,
            'duration_error_seconds': duration_error, 'duration_seconds': sum(check['durations'])}


def comparison_frames(inp, original, fine):
    from PIL import ImageDraw
    images = []
    for tensor, label in zip((inp, original, fine), ('INPUT', 'OFFICIAL', 'TEMPORAL BEST')):
        image = Image.fromarray((tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8))
        image.thumbnail((640, 360))
        panel = Image.new('RGB', (640, 392), 'black')
        panel.paste(image, ((640 - image.width)//2, 32 + (360-image.height)//2))
        ImageDraw.Draw(panel).text((12, 9), label, fill='white')
        images.append(panel)
    combined = Image.new('RGB', (1920, 392))
    for index, panel in enumerate(images):
        combined.paste(panel, (640 * index, 0))
    return combined


def business_video(models, source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    info = probe_video(source)
    if info.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('HDR/PQ/HLG source needs an agreed tone-mapping policy for this SDR model; no silent conversion')
    input_frames = output / 'input_frames'; input_frames.mkdir()
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-threads', '2', '-i', str(source),
                    '-map', '0:v:0', '-vsync', '0', '-start_number', '0', '-threads', '2', '-pix_fmt', 'rgb24',
                    str(input_frames / '%08d.png')], check=True)
    frames = sorted(input_frames.glob('*.png'))
    if len(frames) != info['frames']:
        raise RuntimeError('Extracted/source decoded frame counts differ')
    destinations = {name: output / (name + '_frames') for name in ('official', 'best_stable', 'comparison')}
    for path in destinations.values():
        path.mkdir()
    state, previous, reset_indices = None, None, []
    for index, path in enumerate(frames):
        inp = read_rgb(path)[None].to(models.device)
        array = inp[0].permute(1, 2, 0).cpu().numpy()
        cut = previous is None or previous.shape != array.shape or is_cut(previous, array, models.cut_threshold)
        if cut:
            state = None; reset_indices.append(index)
        with torch.inference_mode():
            original = models.original(inp)
            fine, state = models.temporal.step(inp, state)
        for label, tensor in [('official', original), ('best_stable', fine)]:
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f'Nonfinite business output {index}/{label}')
            im = Image.fromarray((tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8))
            im.save(destinations[label] / path.name)
        comparison_frames(inp, original, fine).save(destinations['comparison'] / path.name)
        previous = array
        if (index + 1) % 50 == 0:
            print(f'BUSINESS {source.name} {index+1}/{len(frames)}', flush=True)
    encoded = {}
    for label, directory in destinations.items():
        encoded[label] = encode_frames(directory, output / f'{label}.mp4', info)
    report = {'source': str(source), 'source_sha256': sha256(source), 'source_probe': info,
              'reset_indices': reset_indices, 'encoded': encoded, 'audio': 'comparison outputs are silent',
              'pixel_processing': 'FFmpeg standard decode to RGB PNG; no manual gamma/exposure normalization',
              'output_geometry': 'model PNGs preserve decoded dimensions; 3-panel preview scales each panel to fit 640x360',
              'quality_metrics': 'No PSNR/SSIM claims on business video without sharp GT'}
    write_json(output / 'business_report.json', report)
    return report
