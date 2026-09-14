"""Full GoPro RGB8 PSNR, bounded T6 center inference; no training or videos."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_nanovnr_waveshift_t6_center import (
    load_model, match_frame_pairs, natural_key, psnr, reflect_index,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rgb8(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected 3-channel RGB8 image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def discover(root):
    test = Path(root).resolve() / 'test'
    if not test.is_dir():
        raise FileNotFoundError(f'Missing explicit test split: {test}')
    sequences = []
    for sequence in sorted((p for p in test.iterdir() if p.is_dir()), key=natural_key):
        blur = sequence / 'blur'
        targets = [sequence / n for n in ('sharp', 'gt', 'GT') if (sequence / n).is_dir()]
        if not blur.is_dir() or len(targets) != 1:
            raise ValueError(f'Need blur and exactly one sharp/gt/GT directory: {sequence}')
        pairs = match_frame_pairs(blur, targets[0])
        if not pairs:
            raise ValueError(f'Empty sequence: {sequence}')
        sequences.append((sequence.name, pairs))
    if len(sequences) != 11 or sum(len(p) for _, p in sequences) != 1111:
        raise ValueError('Incomplete/nonstandard GoPro test: expected 11 sequences / 1111 pairs; '
                         f'found {len(sequences)} / {sum(len(p) for _, p in sequences)}')
    return sequences


def run(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Refusing existing output directory: {out}')
    out.mkdir(parents=True)
    metadata = {'status': 'RUNNING', 'dataset_root': str(Path(args.dataset_root).resolve()),
                'checkpoint': str(Path(args.checkpoint).resolve()),
                'policy': 'T6_CENTER_RESET_PER_TARGET', 'precision': 'FP32_TF32_OFF',
                'temporal_padding': 'reflect_without_repeating_edge',
                'selected_position': 3, 'past': 3, 'future': 2,
                'metric': 'RGB8 clamp[0,1], round-half-up; full-frame per-frame PSNR arithmetic mean',
                'input_directory': 'blur (not blur_gamma)', 'border_crop': 0,
                'deploy_reparam': True, 'completed_frames': 0}
    summary_path = out / 'summary.json'
    summary_path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        sequences = discover(args.dataset_root)
        manifest = [{'sequence': name, 'frames': len(pairs),
                     'pairs': [[str(a), str(b)] for a, b in pairs]} for name, pairs in sequences]
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        metadata['checkpoint_sha256'] = sha256(args.checkpoint)
        metadata['torch'] = torch.__version__
        metadata['cuda'] = torch.version.cuda
        try:
            metadata['git_head'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            metadata['git_head'] = 'UNKNOWN'
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required')
        metadata['gpu'] = torch.cuda.get_device_name(device)
        model, checkpoint = load_model(args.checkpoint, device, deploy_reparam=True)
        if checkpoint.get('step') != 150000 or checkpoint.get('variant') != 'waveshift_edge':
            raise ValueError('Expected step=150000, variant=waveshift_edge')
        if checkpoint['model_config'] != model.config_dict():
            raise ValueError('Checkpoint model_config does not exactly match instantiated model')
        metadata['architecture'] = checkpoint['architecture']
        metadata['model_config'] = checkpoint['model_config']
        metadata['checkpoint_step'] = checkpoint['step']
        probe = read_rgb8(sequences[0][1][0][1])
        if not np.isinf(psnr(probe, probe)):
            raise RuntimeError('GT-vs-GT PSNR sanity failed')
        del probe
        totals, sequence_rows = [], []
        fields = ['sequence', 'frame', 'name', 'window_indices', 'input_psnr_rgb8',
                  'output_psnr_rgb8', 'gain_db', 'input_mean', 'output_mean',
                  'input_dark_rate', 'output_dark_rate', 'raw_min', 'raw_max', 'severe_dark']
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with (out / 'per_frame.csv').open('w', newline='') as stream, torch.inference_mode():
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, pairs in sequences:
                @lru_cache(maxsize=8)
                def frame(index):
                    return read_rgb8(pairs[index][0])
                scores = []
                for index, (blur_path, gt_path) in enumerate(pairs):
                    indices = [reflect_index(index + offset - 3, len(pairs)) for offset in range(6)]
                    array = np.stack([frame(j) for j in indices])
                    target = read_rgb8(gt_path)
                    original = frame(index)
                    if target.shape != original.shape:
                        raise ValueError(f'Input/GT shape mismatch: {gt_path}')
                    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)
                    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
                    prediction, state = model(tensor, prev_forward_feat=None)
                    if tuple(prediction.shape) != tuple(tensor.shape):
                        raise RuntimeError('Unexpected model output shape')
                    if not torch.isfinite(prediction).all().item():
                        raise RuntimeError(f'NON_FINITE_OUTPUT: {name}/{blur_path.name}')
                    raw = prediction[0, 3].float()
                    result = (raw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
                    input_score, output_score = psnr(original, target), psnr(result, target)
                    im, om = float(original.mean()), float(result.mean())
                    idark, odark = float((original <= 1).mean()), float((result <= 1).mean())
                    dark = im > 15 and om < 0.5 * im and odark > max(0.25, idark + 0.20)
                    writer.writerow(dict(zip(fields, [name, index, blur_path.name, ','.join(map(str, indices)),
                        input_score, output_score, output_score-input_score, im, om, idark, odark,
                        float(raw.min()), float(raw.max()), dark])))
                    stream.flush()
                    if dark:
                        raise RuntimeError(f'SEVERE_DARK_OUTPUT: {name}/{blur_path.name}; inspect per_frame.csv')
                    scores.append((input_score, output_score))
                    totals.append((input_score, output_score))
                    metadata['completed_frames'] = len(totals)
                    if index in {0, len(pairs)//2, len(pairs)-1}:
                        preview = cv2.cvtColor(np.hstack([original, result, target]), cv2.COLOR_RGB2BGR)
                        # Exact-resolution, lossless inspection panels; never used for metrics.
                        if not cv2.imwrite(str(out / f'{name}_{index:06d}_input_output_gt.png'), preview):
                            raise RuntimeError('Preview write failed')
                    del prediction, state, raw, tensor, array
                    if index == 0 or (index + 1) % 20 == 0:
                        print(f'{name}: {index+1}/{len(pairs)} total={len(totals)}/1111 '
                              f'input={input_score:.4f} output={output_score:.4f}', flush=True)
                frame.cache_clear()
                avg = np.mean(scores, axis=0)
                sequence_rows.append({'sequence': name, 'frames': len(scores),
                                      'input_psnr_rgb8': float(avg[0]), 'output_psnr_rgb8': float(avg[1]),
                                      'gain_db': float(avg[1]-avg[0])})
        with (out / 'per_sequence.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sequence_rows[0].keys())
            writer.writeheader()
            writer.writerows(sequence_rows)
        if len(totals) != 1111:
            raise RuntimeError('Coverage mismatch')
        avg = np.mean(totals, axis=0)
        metadata.update(status='PASS', sequences=len(sequence_rows), frames=len(totals),
                        input_psnr_rgb8=float(avg[0]), output_psnr_rgb8=float(avg[1]),
                        gain_db=float(avg[1]-avg[0]), severe_dark_frames=[],
                        elapsed_seconds=time.perf_counter()-started,
                        peak_gpu_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        print(json.dumps(metadata, indent=2), flush=True)
    except Exception as exc:
        metadata.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='/data/pub/z00919662/dataset/GoPro')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())
