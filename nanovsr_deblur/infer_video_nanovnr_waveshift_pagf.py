import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.network_nanovnr_waveshift_pagf import NanoVNRWaveShiftPAGF


ARCHITECTURE = 'NanoVNRWaveShiftPAGF'


def load_model(checkpoint, device, deploy_reparam=True):
    data = torch.load(checkpoint, map_location='cpu')
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


def load_video(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f'No frames decoded: {path}')
    return frames, fps


def create_writer(path, fps, size):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create output: {output}')
    return writer, output


def label_frame(bgr, text):
    result = bgr.copy()
    cv2.rectangle(result, (0, 0), (230, 42), (0, 0, 0), -1)
    cv2.putText(
        result, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--side-by-side-output', default=None)
    parser.add_argument('--chunk', type=int, default=15)
    parser.add_argument(
        '--halo', type=int, default=-1,
        help='-1 uses the exact GSTS temporal radius stored by the model.',
    )
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-deploy-reparam', action='store_true')
    args = parser.parse_args()
    if args.chunk < 1:
        raise ValueError('--chunk must be >= 1')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint_data = load_model(
        args.checkpoint,
        device,
        deploy_reparam=not args.no_deploy_reparam,
    )
    halo = model.temporal_radius if args.halo < 0 else args.halo
    if halo < model.temporal_radius:
        raise ValueError(
            f'halo={halo} is smaller than GSTS temporal_radius={model.temporal_radius}'
        )

    frames, fps = load_video(args.input)
    count = len(frames)
    height, width = frames[0].shape[:2]
    for index, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            raise RuntimeError(
                f'Variable frame size at {index}: {frame.shape[:2]} '
                f'vs {(height, width)}'
            )

    outputs = []
    previous_forward = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for core_start_global in range(0, count, args.chunk):
            core_end_global = min(count, core_start_global + args.chunk)
            halo_start = max(0, core_start_global - halo)
            halo_end = min(count, core_end_global + halo)
            core_start_local = core_start_global - halo_start
            core_end_local = core_start_local + (core_end_global - core_start_global)
            array = np.stack(frames[halo_start:halo_end]).astype(np.float32) / 255.0
            tensor = (
                torch.from_numpy(array)
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(device)
            )
            with torch.cuda.amp.autocast(enabled=args.fp16):
                restored, previous_forward = model(
                    tensor,
                    prev_forward_feat=previous_forward,
                    core_start=core_start_local,
                    core_end=core_end_local,
                )
            previous_forward = previous_forward.detach()
            restored = (
                restored[0]
                .float()
                .clamp(0, 1)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            outputs.extend(restored)
            print(
                f'core={core_start_global}:{core_end_global} '
                f'halo={halo_start}:{halo_end} carry_forward_state=YES',
                flush=True,
            )

    if len(outputs) != count:
        raise RuntimeError(f'Output frame count mismatch: {len(outputs)} vs {count}')
    output_writer, output_path = create_writer(args.output, fps, (width, height))
    side_writer = None
    side_path = None
    if args.side_by_side_output:
        side_writer, side_path = create_writer(
            args.side_by_side_output, fps, (width * 2, height)
        )

    for input_rgb, output_rgb in zip(frames, outputs):
        output_u8 = np.clip(output_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
        output_bgr = cv2.cvtColor(output_u8, cv2.COLOR_RGB2BGR)
        output_writer.write(output_bgr)
        if side_writer is not None:
            side_writer.write(
                np.hstack([
                    label_frame(input_bgr, 'Input'),
                    label_frame(output_bgr, 'WaveShift-PAGF'),
                ])
            )
    output_writer.release()
    if side_writer is not None:
        side_writer.release()

    peak = 0.0
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f'ARCHITECTURE={checkpoint_data.get("architecture")}')
    print(f'VARIANT={checkpoint_data.get("variant")}')
    print(f'CHECKPOINT_STEP={checkpoint_data.get("step")}')
    print(f'OUTPUT={output_path}')
    print(f'SIDE_BY_SIDE_OUTPUT={side_path}')
    print(f'FRAMES={count} FPS={fps:.3f} SIZE={width}x{height}')
    print(
        f'CHUNK={args.chunk} HALO={halo} FORWARD_LL_STATE_CARRY=YES '
        f'BACKWARD_STATE_PER_CHUNK=RESET'
    )
    print(f'DEPLOY_REPARAM={not args.no_deploy_reparam}')
    print(f'PEAK_GPU_GIB={peak:.3f}')


if __name__ == '__main__':
    main()
