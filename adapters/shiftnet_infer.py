#!/usr/bin/env python3
"""Run official Shift-Net+ deblurring checkpoints on an arbitrary frame folder."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import import_repo, inspect_frames, list_frames, load_rgb_float, reflection_indices, save_rgb_float, unwrap_state_dict, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--one-len", type=int, default=48)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def load_model_class():
    try:
        from basicsr.models.archs.gshift_deblur1 import GShiftNet
    except ImportError:
        from basicsr.archs.gshift_deblur1 import GShiftNet
    return GShiftNet


def pad_video_to_multiple(x: torch.Tensor, multiple: int = 4):
    """Pad B,T,C,H,W without calling 2D reflect-pad on a 5D tensor."""
    if x.ndim != 5:
        raise ValueError(f"Expected B,T,C,H,W tensor, got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x
    mode = "reflect" if h > 1 and w > 1 and ph < h and pw < w else "replicate"
    flat = x.reshape(b * t, c, h, w)
    flat = F.pad(flat, (0, pw, 0, ph), mode=mode)
    return flat.reshape(b, t, c, h + ph, w + pw)


def main():
    args = parse_args()
    repo = import_repo(args.repo)
    GShiftNet = load_model_class()
    frames = list_frames(args.input)
    height, width = inspect_frames(frames)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = GShiftNet(future_frames=2, past_frames=2).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = unwrap_state_dict(checkpoint)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")
    if args.fp16:
        model.half()

    started = time.time()
    n = len(frames)
    with torch.inference_mode():
        for start in range(0, n, args.one_len):
            end = min(start + args.one_len, n)
            ids = reflection_indices(start - 2, end + 2, n)
            arrays = [load_rgb_float(frames[i]) for i in ids]
            tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            tensor = pad_video_to_multiple(tensor, 4)
            if args.fp16:
                tensor = tensor.half()
            pred = model(tensor)
            # Official GShiftNet returns the valid central T-4 frames, normally without batch dim.
            if pred.ndim == 5:
                pred = pred[0]
            if pred.shape[0] == len(ids):
                pred = pred[2:-2]
            if pred.shape[0] != end - start:
                raise RuntimeError(
                    f"Shift-Net returned {tuple(pred.shape)} for {len(ids)} input frames; expected {end-start} outputs"
                )
            pred = pred.float().cpu()[:, :, :height, :width].permute(0, 2, 3, 1).numpy()
            for local, idx in enumerate(range(start, end)):
                save_rgb_float(pred[local], out_dir / frames[idx].name)
            print(f"Shift-Net chunk [{start}, {end})")

    write_json(out_dir.parent / "run_metadata.json", {
        "model": "Shift-Net+",
        "repo": str(repo),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frame_count": n,
        "width": width,
        "height": height,
        "one_len": args.one_len,
        "fp16": args.fp16,
        "runtime_seconds": time.time() - started,
        "torch": torch.__version__,
    })


if __name__ == "__main__":
    main()
