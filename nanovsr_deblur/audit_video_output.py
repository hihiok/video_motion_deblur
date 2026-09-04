"""No-reference safety audit for a restored business video."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def open_video(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    return capture


def label(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (260, 42), (0, 0, 0), -1)
    cv2.putText(
        result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunk', type=int, default=15)
    parser.add_argument('--preview', required=True)
    args = parser.parse_args()
    if args.chunk < 1:
        raise ValueError('--chunk must be >= 1')

    input_capture = open_video(args.input)
    output_capture = open_video(args.output)
    input_count_hint = int(input_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_count_hint = int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = set(
        np.linspace(0, max(0, min(input_count_hint, output_count_hint) - 1), 5)
        .round().astype(int).tolist()
    )

    frame_count = 0
    shape = None
    absolute_change_sum = 0.0
    channel_shift_sum = np.zeros(3, dtype=np.float64)
    input_dark = input_bright = output_dark = output_bright = 0
    pixel_count = 0
    input_laplacian = []
    output_laplacian = []
    input_temporal = []
    output_temporal = []
    previous_input_gray = None
    previous_output_gray = None
    preview_rows = []

    while True:
        input_ok, input_bgr = input_capture.read()
        output_ok, output_bgr = output_capture.read()
        if input_ok != output_ok:
            raise RuntimeError('Input/output decoded frame counts differ.')
        if not input_ok:
            break
        if input_bgr.shape != output_bgr.shape:
            raise RuntimeError(
                f'Frame shape mismatch at {frame_count}: '
                f'{input_bgr.shape} vs {output_bgr.shape}'
            )
        if shape is None:
            shape = input_bgr.shape
        elif input_bgr.shape != shape:
            raise RuntimeError(f'Variable input frame shape at {frame_count}.')

        input_float = input_bgr.astype(np.float32)
        output_float = output_bgr.astype(np.float32)
        difference = output_float - input_float
        absolute_change_sum += float(np.abs(difference).sum())
        channel_shift_sum += difference.reshape(-1, 3).sum(axis=0)
        input_dark += int((input_bgr <= 1).sum())
        input_bright += int((input_bgr >= 254).sum())
        output_dark += int((output_bgr <= 1).sum())
        output_bright += int((output_bgr >= 254).sum())
        pixel_count += input_bgr.size

        input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        output_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        input_laplacian.append(float(cv2.Laplacian(input_gray, cv2.CV_32F).var()))
        output_laplacian.append(float(cv2.Laplacian(output_gray, cv2.CV_32F).var()))
        if previous_input_gray is not None:
            input_temporal.append(float(np.abs(input_gray - previous_input_gray).mean()))
            output_temporal.append(float(np.abs(output_gray - previous_output_gray).mean()))
        previous_input_gray = input_gray
        previous_output_gray = output_gray

        if frame_count in selected:
            preview_rows.append(
                np.hstack([
                    label(input_bgr, f'Input frame {frame_count}'),
                    label(output_bgr, f'Output frame {frame_count}'),
                ])
            )
        frame_count += 1

    input_capture.release()
    output_capture.release()
    if frame_count == 0:
        raise RuntimeError('No paired frames decoded.')

    preview = np.vstack(preview_rows)
    if preview.shape[1] > 1920:
        scale = 1920.0 / preview.shape[1]
        preview = cv2.resize(
            preview,
            (1920, int(round(preview.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    preview_path = Path(args.preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f'Cannot write preview: {preview_path}')

    transitions = np.arange(1, frame_count)
    boundary_mask = transitions % args.chunk == 0
    output_temporal_array = np.asarray(output_temporal, dtype=np.float64)
    input_temporal_array = np.asarray(input_temporal, dtype=np.float64)
    boundary_output = (
        float(output_temporal_array[boundary_mask].mean())
        if boundary_mask.any() else 0.0
    )
    nonboundary_output = (
        float(output_temporal_array[~boundary_mask].mean())
        if (~boundary_mask).any() else 0.0
    )
    boundary_ratio = boundary_output / max(nonboundary_output, 1e-12)

    channel_shift_bgr = channel_shift_sum / max(1, pixel_count // 3)
    input_lap = float(np.mean(input_laplacian))
    output_lap = float(np.mean(output_laplacian))
    print('VIDEO_OUTPUT_AUDIT=PASS')
    print(f'FRAMES={frame_count}')
    print(f'SIZE={shape[1]}x{shape[0]}')
    print(f'MEAN_ABS_OUTPUT_CHANGE={absolute_change_sum / pixel_count:.6f}')
    print(
        'MEAN_CHANNEL_SHIFT_BGR='
        + ','.join(f'{value:+.6f}' for value in channel_shift_bgr)
    )
    print(f'INPUT_DARK_CLIP_RATE={input_dark / pixel_count:.8f}')
    print(f'OUTPUT_DARK_CLIP_RATE={output_dark / pixel_count:.8f}')
    print(f'INPUT_BRIGHT_CLIP_RATE={input_bright / pixel_count:.8f}')
    print(f'OUTPUT_BRIGHT_CLIP_RATE={output_bright / pixel_count:.8f}')
    print(f'INPUT_LAPLACIAN_VARIANCE={input_lap:.6f}')
    print(f'OUTPUT_LAPLACIAN_VARIANCE={output_lap:.6f}')
    print(f'LAPLACIAN_VARIANCE_RATIO={output_lap / max(input_lap, 1e-12):.6f}')
    print(f'INPUT_TEMPORAL_MAD={input_temporal_array.mean():.6f}')
    print(f'OUTPUT_TEMPORAL_MAD={output_temporal_array.mean():.6f}')
    print(f'CHUNK_BOUNDARY_OUTPUT_MAD={boundary_output:.6f}')
    print(f'NONBOUNDARY_OUTPUT_MAD={nonboundary_output:.6f}')
    print(f'CHUNK_BOUNDARY_DISCONTINUITY_RATIO={boundary_ratio:.6f}')
    print(f'PREVIEW={preview_path}')


if __name__ == '__main__':
    main()
