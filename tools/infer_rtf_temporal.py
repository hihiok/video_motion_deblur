#!/usr/bin/env python3
"""Stream a frame directory through official/spatial/temporal weights; save PNG/MP4."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtf_t6.datasets import read_rgb, IMAGE_EXTENSIONS
from rtf_temporal.model import TemporalRTFocuser
from rtf_temporal.flow import is_cut
from rtf_temporal.data import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Directory of consecutive RGB frames')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--official', action='store_true')
    p.add_argument('--spatial-only', action='store_true', help='Ablation using trained backbone, temporal disabled')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--fps', type=float, help='Known source FPS; omitted means PNG-only, never guess')
    p.add_argument('--cut-threshold', type=float, default=.30)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    frames_dir = output / 'frames'; frames_dir.mkdir(parents=True, exist_ok=True)
    frames = [x for x in Path(args.input).iterdir() if x.suffix.lower() in IMAGE_EXTENSIONS]
    def frame_id(path):
        match = re.search(r'(\d+)$', path.stem)
        if match is None:
            raise ValueError(f'Cannot identify chronological frame number: {path}')
        return int(match.group(1))
    frames.sort(key=frame_id)
    if not frames or any(frame_id(b) != frame_id(a) + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError('Input must contain nonempty consecutive unique frame IDs')
    model = TemporalRTFocuser(activation_checkpointing=False)
    if args.official:
        model.load_official(args.checkpoint)
    else:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if ckpt['config']['protocol'] != 'rtfocuser_causal_temporal_finetune_v1':
            raise ValueError('Wrong checkpoint protocol')
        model = TemporalRTFocuser(**{**ckpt['config']['model'], 'activation_checkpointing': False})
        model.load_state_dict(ckpt['model'], strict=True)
        del ckpt
    model.to(device).eval()
    previous, state, resets, mapping, sizes = None, None, [], [], set()
    started = time.perf_counter()
    with torch.inference_mode():
        for i, path in enumerate(frames):
            inp = read_rgb(path)[None].to(device)
            array = inp[0].permute(1, 2, 0).cpu().numpy()
            size = tuple(inp.shape[-2:]); sizes.add(size)
            reset = previous is None or previous.shape != array.shape or is_cut(previous, array, args.cut_threshold)
            if reset:
                state = None; resets.append(i)
            # FP32 inference is the reference; no framewise auto-exposure/normalization.
            pred, state = model.step(inp, state, spatial_only=args.official or args.spatial_only)
            arr = pred[0].permute(1, 2, 0).cpu().numpy()
            name = f'{i:08d}.png'
            Image.fromarray((arr.clip(0, 1) * 255).round().astype(np.uint8)).save(frames_dir / name)
            mapping.append({'source': str(path), 'output': name, 'state_reset': reset})
            previous = array
            if (i + 1) % 50 == 0:
                print(f'INFERENCE {i+1}/{len(frames)}', flush=True)
    report = dict(frames=len(frames), seconds=time.perf_counter() - started, reset_indices=resets,
                  checkpoint_sha256=sha256(args.checkpoint), mapping=mapping, fp32=True,
                  spatial_only=args.official or args.spatial_only)
    (output / 'inference.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.fps is not None:
        if args.fps <= 0 or len(sizes) != 1:
            raise ValueError('MP4 requires positive source FPS and fixed frame size; PNGs are complete')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-framerate', str(args.fps),
                        '-i', str(frames_dir / '%08d.png'), '-c:v', 'libx264', '-crf', '16',
                        '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-threads', '2',
                        str(output / 'output.mp4')], check=True)
    print('INFERENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
