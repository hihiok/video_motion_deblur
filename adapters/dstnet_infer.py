#!/usr/bin/env python3
"""Run official DSTNet checkpoints on an arbitrary image sequence without GT."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    blend_weights,
    inspect_frames,
    list_frames,
    load_rgb_float,
    save_rgb_float,
    temporal_chunks,
    unwrap_state_dict,
    write_json,
)
from dstnet_compat import load_dstnet_deblur


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clip-len", type=int, default=30)
    p.add_argument("--overlap", type=int, default=10)
    p.add_argument("--num-feat", type=int, default=64)
    p.add_argument("--num-block", type=int, default=15)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def pad_video_tensor(x: torch.Tensor, multiple: int = 4):
    """Pad B,T,C,H,W by reshaping to 4D first.

    PyTorch reflect padding does not support applying 2D padding directly to a
    5D tensor in every supported version.
    """
    if x.ndim != 5:
        raise ValueError(f"Expected B,T,C,H,W tensor, got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, ph, pw
    mode = "reflect" if h > 1 and w > 1 and ph < h and pw < w else "replicate"
    flat = x.reshape(b * t, c, h, w)
    flat = F.pad(flat, (0, pw, 0, ph), mode=mode)
    return flat.reshape(b, t, c, h + ph, w + pw), ph, pw


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    Deblur, dynamic_backend = load_dstnet_deblur(repo)

    frames = list_frames(args.input)
    height, width = inspect_frames(frames)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = Deblur(num_feat=args.num_feat, num_block=args.num_block).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = unwrap_state_dict(ckpt)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")

    sums = [np.zeros((height, width, 3), np.float32) for _ in frames]
    weights = np.zeros(len(frames), np.float32)
    chunks = temporal_chunks(len(frames), args.clip_len, args.overlap)
    started = time.time()

    with torch.inference_mode():
        for chunk_id, (start, end) in enumerate(chunks):
            arrays = [load_rgb_float(p) for p in frames[start:end]]
            tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            tensor, _, _ = pad_video_tensor(tensor, 4)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(tensor)
            pred = pred[0, :, :, :height, :width].float().cpu().permute(0, 2, 3, 1).numpy()
            if pred.shape[0] != end - start:
                raise RuntimeError(f"DSTNet returned {pred.shape[0]} frames for chunk length {end-start}")
            local_w = blend_weights(end - start, start, end, len(frames), args.overlap)
            for j, idx in enumerate(range(start, end)):
                sums[idx] += pred[j] * local_w[j]
                weights[idx] += local_w[j]
            print(
                f"DSTNet chunk {chunk_id+1}/{len(chunks)}: [{start}, {end}); "
                f"dynamic_backend={dynamic_backend}"
            )

    for idx, src in enumerate(frames):
        if weights[idx] <= 0:
            raise RuntimeError(f"No output weight accumulated for frame {idx}")
        save_rgb_float(sums[idx] / weights[idx], output / src.name)

    write_json(output.parent / "run_metadata.json", {
        "model": "DSTNet",
        "repo": str(repo),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input": str(Path(args.input).resolve()),
        "output": str(output.resolve()),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "clip_len": args.clip_len,
        "overlap": args.overlap,
        "amp": args.amp,
        "dynamic_conv_backend": dynamic_backend,
        "runtime_seconds": time.time() - started,
        "torch": torch.__version__,
    })


if __name__ == "__main__":
    main()
