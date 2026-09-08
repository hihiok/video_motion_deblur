"""Bounded T=6 center-frame inference for videos or paired test sequences.

Every output frame is produced from an independent six-frame window and the
recurrent state is reset for that window. This matches the training horizon and
prevents an unbounded forward state from drifting over a long business video.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from models.network_nanovnr_waveshift_pagf import NanoVNRWaveShiftPAGF


WINDOW = 6
CENTER = WINDOW // 2
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
BLUR_NAMES = ('blur', 'Blur', 'blurry', 'input')
GT_NAMES = ('sharp', 'Sharp', 'gt', 'GT', 'target', 'label')
ARCHITECTURE = 'NanoVNRWaveShiftPAGF'


def load_model(checkpoint, device, deploy_reparam):
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if data.get('architecture') != ARCHITECTURE:
        raise RuntimeError(f'Unexpected architecture: {data.get("architecture")}')
    config = data.get('model_config')
    if not isinstance(config, dict):
        raise RuntimeError('Checkpoint is missing model_config.')
    model = NanoVNRWaveShiftPAGF.from_config(config).to(device).eval()
    model.load_state_dict(data['model'], strict=True)
    if deploy_reparam:
        model.switch_to_deploy()
    return model, data


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(path))]


def image_files(path):
    return sorted(
        [item for item in Path(path).iterdir()
         if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def candidate_bases(root, split, strict_root_split):
    root = Path(root)
    if strict_root_split:
        direct = root / split
        return [direct] if direct.is_dir() else []
    aliases = [split]
    if split == 'train':
        aliases.append('training')
    elif split == 'test':
        aliases.append('testing')
    bases = [root / name for name in aliases if (root / name).is_dir()]
    for child in sorted([p for p in root.iterdir() if p.is_dir()], key=natural_key):
        bases.extend(child / name for name in aliases if (child / name).is_dir())
    if not bases:
        bases.append(root)
    result, seen = [], set()
    for base in bases:
        resolved = base.resolve()
        if resolved not in seen:
            result.append(base)
            seen.add(resolved)
    return result


def discover_pair_roots(root, split, strict_root_split):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    pairs, seen = [], set()

    def add_pair(blur, gt):
        key = (blur.resolve(), gt.resolve())
        if blur.is_dir() and gt.is_dir() and key not in seen:
            pairs.append((blur, gt))
            seen.add(key)

    for base in candidate_bases(root, split, strict_root_split):
        for blur_name in BLUR_NAMES:
            blur = base / blur_name
            for gt_name in GT_NAMES:
                gt = base / gt_name
                if blur.is_dir() and gt.is_dir():
                    add_pair(blur, gt)
                    break
        for sequence in sorted(
            [p for p in base.iterdir() if p.is_dir()], key=natural_key
        ):
            for blur_name in ('Blur', 'blur'):
                blur_container = sequence / blur_name
                blur = blur_container / 'RGB'
                if not blur.is_dir():
                    blur = blur_container
                for gt_name in ('Sharp', 'sharp', 'GT', 'gt'):
                    gt_container = sequence / gt_name
                    gt = gt_container / 'RGB'
                    if not gt.is_dir():
                        gt = gt_container
                    if blur.is_dir() and gt.is_dir() and image_files(blur) and image_files(gt):
                        add_pair(blur, gt)
                        break
    return pairs


def sequence_dirs(root):
    directories = sorted([p for p in Path(root).iterdir() if p.is_dir()], key=natural_key)
    return directories if directories else [Path(root)]


def match_frame_pairs(blur_dir, gt_dir):
    blur = image_files(blur_dir)
    gt = image_files(gt_dir)
    gt_by_name = {path.name: path for path in gt}
    pairs = [(path, gt_by_name[path.name]) for path in blur if path.name in gt_by_name]
    if pairs and len(pairs) == len(blur) == len(gt):
        return pairs
    raise RuntimeError(
        f'Exact filename alignment failed: blur={blur_dir} ({len(blur)}), '
        f'gt={gt_dir} ({len(gt)}), exact={len(pairs)}'
    )


def reflect_index(index, length):
    if length < 1:
        raise ValueError('Cannot reflect an empty sequence.')
    if length == 1:
        return 0
    while index < 0 or index >= length:
        index = -index if index < 0 else 2 * length - index - 2
    return index


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'Cannot decode image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_video(path, max_frames=0):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while not max_frames or len(frames) < max_frames:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f'No frames decoded: {path}')
    return frames, None, [f'frame_{i:06d}' for i in range(len(frames))], fps, str(path)


def dataset_sequences(root, family, split):
    strict_bsd = family.lower() == 'bsd'
    candidates = []
    for blur_root, gt_root in discover_pair_roots(root, split, strict_bsd):
        for blur_dir in sequence_dirs(blur_root):
            name = blur_dir.name if blur_dir != blur_root else '__root__'
            gt_dir = gt_root / name if name != '__root__' else gt_root
            if not gt_dir.is_dir():
                raise RuntimeError(f'Missing GT sequence: {gt_dir}')
            pairs = match_frame_pairs(blur_dir, gt_dir)
            if pairs:
                candidates.append((blur_dir, gt_dir, pairs))
    if not candidates:
        raise RuntimeError(f'No paired {family}/{split} sequence found under {root}')
    return candidates


def load_dataset(root, family, split, sequence_index, max_frames, fps):
    candidates = dataset_sequences(root, family, split)
    if not 0 <= sequence_index < len(candidates):
        raise IndexError(
            f'sequence-index={sequence_index}; available sequences={len(candidates)}'
        )
    blur_dir, gt_dir, pairs = candidates[sequence_index]
    if max_frames:
        pairs = pairs[:max_frames]
    frames = [read_rgb(blur) for blur, _ in pairs]
    targets = [read_rgb(gt) for _, gt in pairs]
    names = [blur.name for blur, _ in pairs]
    source = f'{family}:{split}:{blur_dir}'
    return frames, targets, names, fps, source


def writer(path, fps, size):
    value = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size
    )
    if not value.isOpened():
        raise RuntimeError(f'Cannot create video: {path}')
    return value


def label_frame(bgr, text):
    result = bgr.copy()
    cv2.rectangle(result, (0, 0), (340, 42), (0, 0, 0), -1)
    cv2.putText(result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-video')
    source.add_argument('--dataset-root')
    parser.add_argument('--family', choices=('GoPro', 'DVD', 'BSD'))
    parser.add_argument('--split', default='test')
    parser.add_argument('--sequence-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--dataset-fps', type=float, default=25.0)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-deploy-reparam', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.max_frames < 0:
        raise ValueError('--max-frames must be nonnegative')
    if args.dataset_root and not args.family:
        parser.error('--family is required with --dataset-root')

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f'Refusing non-empty output directory: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_video:
        frames, targets, names, fps, source_name = load_video(
            args.input_video, args.max_frames
        )
    else:
        frames, targets, names, fps, source_name = load_dataset(
            args.dataset_root, args.family, args.split, args.sequence_index,
            args.max_frames, args.dataset_fps,
        )
    count = len(frames)
    height, width = frames[0].shape[:2]
    for index, frame in enumerate(frames):
        if frame.shape != frames[0].shape:
            raise RuntimeError(f'Variable input shape at frame {index}: {frame.shape}')
    if targets is not None:
        for index, target in enumerate(targets):
            if target.shape != frames[0].shape:
                raise RuntimeError(f'Input/GT shape mismatch at frame {index}')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    if args.fp16 and device.type != 'cuda':
        raise RuntimeError('--fp16 is supported only on CUDA.')
    model, checkpoint_data = load_model(
        args.checkpoint, device, deploy_reparam=not args.no_deploy_reparam
    )
    if checkpoint_data.get('step') != 150000:
        raise RuntimeError(f'Expected step 150000, got {checkpoint_data.get("step")}')
    if int(model.temporal_radius) != 2:
        raise RuntimeError(f'Unexpected temporal radius: {model.temporal_radius}')

    restored_path = output_dir / 'restored.mp4'
    comparison_path = output_dir / (
        'input_vs_output_vs_gt.mp4' if targets is not None else 'input_vs_output.mp4'
    )
    restored_writer = writer(restored_path, fps, (width, height))
    columns = 3 if targets is not None else 2
    comparison_scale = min(1.0, 1920.0 / (width * columns))
    comparison_size = (
        int(round(width * columns * comparison_scale)),
        int(round(height * comparison_scale)),
    )
    comparison_writer = writer(comparison_path, fps, comparison_size)
    preview_indices = set(
        np.linspace(0, count - 1, min(5, count)).round().astype(int).tolist()
    )
    preview_rows = []
    rows = []
    input_sum = np.zeros(3, dtype=np.float64)
    output_sum = np.zeros(3, dtype=np.float64)
    input_dark = output_dark = pixels = 0
    input_laplacian, output_laplacian = [], []
    black_flags = []

    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for target_index in range(count):
            window_indices = [
                reflect_index(target_index + offset - CENTER, count)
                for offset in range(WINDOW)
            ]
            array = np.stack([frames[index] for index in window_indices])
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.fp16
            ):
                prediction, _ = model(tensor, prev_forward_feat=None)
            raw = prediction[0, CENTER].float()
            if not torch.isfinite(raw).all().item():
                raise RuntimeError(f'NON_FINITE_OUTPUT at frame {target_index}')
            raw_min = float(raw.min())
            raw_max = float(raw.max())
            output_rgb = (
                raw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5
            ).astype(np.uint8)
            input_rgb = frames[target_index]
            input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            restored_writer.write(output_bgr)
            panels = [label_frame(input_bgr, f'Input {target_index}'),
                      label_frame(output_bgr, 'WaveShift T6 center')]
            frame_psnr = None
            if targets is not None:
                target_rgb = targets[target_index]
                target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
                panels.append(label_frame(target_bgr, 'GT'))
                frame_psnr = psnr(output_rgb, target_rgb)
            combined = np.hstack(panels)
            if comparison_scale < 1.0:
                combined = cv2.resize(
                    combined, comparison_size, interpolation=cv2.INTER_AREA
                )
            comparison_writer.write(combined)
            if target_index in preview_indices:
                preview_rows.append(combined)

            input_mean = float(input_rgb.mean())
            output_mean = float(output_rgb.mean())
            input_dark_rate = float((input_rgb <= 1).mean())
            output_dark_rate = float((output_rgb <= 1).mean())
            severe_dark = (
                input_mean > 15.0
                and output_mean < 0.50 * input_mean
                and output_dark_rate > max(0.25, input_dark_rate + 0.20)
            )
            if severe_dark:
                black_flags.append(target_index)
            input_sum += input_rgb.reshape(-1, 3).sum(axis=0)
            output_sum += output_rgb.reshape(-1, 3).sum(axis=0)
            input_dark += int((input_rgb <= 1).sum())
            output_dark += int((output_rgb <= 1).sum())
            pixels += input_rgb.size
            input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
            output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY)
            input_laplacian.append(float(cv2.Laplacian(input_gray, cv2.CV_32F).var()))
            output_laplacian.append(float(cv2.Laplacian(output_gray, cv2.CV_32F).var()))
            rows.append({
                'frame': target_index, 'name': names[target_index],
                'window_indices': ','.join(map(str, window_indices)),
                'input_mean': input_mean, 'output_mean': output_mean,
                'output_to_input_mean_ratio': output_mean / max(input_mean, 1e-12),
                'input_dark_clip_rate': input_dark_rate,
                'output_dark_clip_rate': output_dark_rate,
                'raw_output_min': raw_min, 'raw_output_max': raw_max,
                'output_psnr_rgb': frame_psnr,
                'severe_dark_flag': severe_dark,
            })
            if target_index == 0 or (target_index + 1) % 20 == 0:
                print(
                    f'frame={target_index + 1}/{count} window={window_indices} '
                    f'input_mean={input_mean:.3f} output_mean={output_mean:.3f} '
                    f'dark={output_dark_rate:.5f} severe_dark={severe_dark}',
                    flush=True,
                )

    restored_writer.release()
    comparison_writer.release()
    per_channel_pixels = pixels // 3
    input_mean_rgb = input_sum / per_channel_pixels
    output_mean_rgb = output_sum / per_channel_pixels
    input_lap = float(np.mean(input_laplacian))
    output_lap = float(np.mean(output_laplacian))
    psnr_values = [row['output_psnr_rgb'] for row in rows
                   if row['output_psnr_rgb'] is not None]
    summary = {
        'status': 'PASS' if not black_flags else 'SEVERE_DARK_OUTPUT',
        'source': source_name,
        'checkpoint': str(args.checkpoint),
        'checkpoint_step': int(checkpoint_data['step']),
        'architecture': checkpoint_data.get('architecture'),
        'variant': checkpoint_data.get('variant'),
        'frames': count, 'fps': fps, 'width': width, 'height': height,
        'temporal_window': WINDOW, 'selected_position': CENTER,
        'future_frames': WINDOW - CENTER - 1,
        'state_reset_per_target': True,
        'precision': 'fp16' if args.fp16 else 'fp32',
        'deploy_reparam': not args.no_deploy_reparam,
        'input_mean_rgb': input_mean_rgb.tolist(),
        'output_mean_rgb': output_mean_rgb.tolist(),
        'mean_shift_rgb': (output_mean_rgb - input_mean_rgb).tolist(),
        'input_dark_clip_rate': input_dark / pixels,
        'output_dark_clip_rate': output_dark / pixels,
        'laplacian_variance_ratio': output_lap / max(input_lap, 1e-12),
        'severe_dark_frames': black_flags,
        'mean_psnr_rgb': float(np.mean(psnr_values)) if psnr_values else None,
        'peak_gpu_gib': (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == 'cuda' else 0.0
        ),
        'restored_video': str(restored_path),
        'comparison_video': str(comparison_path),
    }
    with (output_dir / 'per_frame.csv').open('w', newline='') as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    preview = np.vstack(preview_rows)
    preview_path = output_dir / 'preview.jpg'
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f'Cannot write preview: {preview_path}')

    print('INFERENCE_SUMMARY=' + json.dumps(summary))
    print(f'PREVIEW={preview_path}')


if __name__ == '__main__':
    main()
