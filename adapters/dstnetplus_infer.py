#!/usr/bin/env python3
"""Run official DSTNet+ Base (TPAMI 2025) on an arbitrary frame folder."""
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
from dstnetplus_compat import load_dstnetplus_base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--overlap", type=int, default=2)
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--tile-overlap", type=int, default=64)
    p.add_argument("--min-tile-size", type=int, default=192)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def pad_video_tensor(x: torch.Tensor, multiple: int = 2):
    if x.ndim != 5:
        raise ValueError(f"Expected B,T,C,H,W, got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, 0, 0
    mode = "reflect" if h > 1 and w > 1 and ph < h and pw < w else "replicate"
    flat = x.reshape(b * t, c, h, w)
    flat = F.pad(flat, (0, pw, 0, ph), mode=mode)
    return flat.reshape(b, t, c, h + ph, w + pw), ph, pw


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or tile_size >= length:
        return [0]
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("tile-overlap must be smaller than tile-size")
    starts = list(range(0, max(length - tile_size, 0) + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def spatial_weight(h: int, w: int, y0: int, x0: int, H: int, W: int, overlap: int):
    wy = np.ones(h, np.float32)
    wx = np.ones(w, np.float32)
    if overlap > 0:
        if y0 > 0:
            n = min(overlap, h)
            wy[:n] *= np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
        if y0 + h < H:
            n = min(overlap, h)
            wy[-n:] *= np.linspace(1.0, 0.0, n + 2, dtype=np.float32)[1:-1]
        if x0 > 0:
            n = min(overlap, w)
            wx[:n] *= np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
        if x0 + w < W:
            n = min(overlap, w)
            wx[-n:] *= np.linspace(1.0, 0.0, n + 2, dtype=np.float32)[1:-1]
    return wy[:, None] * wx[None, :]


def infer_tiled(model, tensor, tile_size, tile_overlap, amp, device):
    _, t, _, H, W = tensor.shape
    ys = tile_starts(H, tile_size, tile_overlap)
    xs = tile_starts(W, tile_size, tile_overlap)
    accum = np.zeros((t, H, W, 3), np.float32)
    weights = np.zeros((H, W), np.float32)

    for y0 in ys:
        y1 = min(y0 + tile_size, H) if tile_size > 0 else H
        for x0 in xs:
            x1 = min(x0 + tile_size, W) if tile_size > 0 else W
            tile = tensor[..., y0:y1, x0:x1].contiguous()
            th, tw = y1 - y0, x1 - x0
            tile, _, _ = pad_video_tensor(tile, 2)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                pred = model(tile)
            pred_np = (
                pred[0, :, :, :th, :tw]
                .float().cpu().permute(0, 2, 3, 1).numpy()
            )
            w = spatial_weight(th, tw, y0, x0, H, W, tile_overlap)
            accum[:, y0:y1, x0:x1] += pred_np * w[None, :, :, None]
            weights[y0:y1, x0:x1] += w
            del tile, pred, pred_np

    if np.any(weights <= 0):
        raise RuntimeError("Spatial tiling left uncovered pixels")
    return accum / weights[None, :, :, None]


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def main():
    args = parse_args()
    if args.clip_len < 1:
        raise ValueError("clip-len must be >= 1")
    if args.overlap < 0 or args.overlap >= args.clip_len:
        raise ValueError("overlap must satisfy 0 <= overlap < clip-len")
    if args.tile_size < args.min_tile_size:
        raise ValueError("tile-size must be >= min-tile-size")

    repo = Path(args.repo).resolve()
    Model, dynamic_backend = load_dstnetplus_base(repo)
    model = Model(
        num_feat=64,
        num_kernel_block=3,
        num_block=15,
        nonblind_denoise=False,
    )

    checkpoint = Path(args.checkpoint).resolve()
    raw = torch.load(checkpoint, map_location="cpu")
    state = unwrap_state_dict(raw)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {result}")

    frames = list_frames(args.input)
    height, width = inspect_frames(frames)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.to(device).eval()

    sums = [np.zeros((height, width, 3), np.float32) for _ in frames]
    frame_weights = np.zeros(len(frames), np.float32)
    chunks = temporal_chunks(len(frames), args.clip_len, args.overlap)
    effective_tiles = []
    started = time.time()

    with torch.inference_mode():
        for chunk_id, (start, end) in enumerate(chunks):
            arrays = [load_rgb_float(p) for p in frames[start:end]]
            tensor = (
                torch.from_numpy(np.stack(arrays))
                .permute(0, 3, 1, 2).unsqueeze(0)
                .to(device).contiguous()
            )

            current_tile = args.tile_size
            while True:
                try:
                    pred = infer_tiled(
                        model, tensor,
                        current_tile,
                        min(args.tile_overlap, max(current_tile // 4, 1)),
                        args.amp,
                        device,
                    )
                    effective_tiles.append(current_tile)
                    break
                except BaseException as exc:
                    if not is_oom(exc) or current_tile <= args.min_tile_size:
                        raise
                    next_tile = max(args.min_tile_size, ((current_tile // 2) // 2) * 2)
                    if next_tile >= current_tile:
                        raise
                    print(
                        f"CUDA OOM for chunk [{start},{end}) tile={current_tile}; "
                        f"retry tile={next_tile}"
                    )
                    torch.cuda.empty_cache()
                    current_tile = next_tile

            local_w = blend_weights(end - start, start, end, len(frames), args.overlap)
            for j, idx in enumerate(range(start, end)):
                sums[idx] += pred[j] * local_w[j]
                frame_weights[idx] += local_w[j]

            del arrays, tensor, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"DSTNet+ Base chunk {chunk_id + 1}/{len(chunks)} "
                f"[{start},{end}) tile={effective_tiles[-1]} backend={dynamic_backend}"
            )

    for idx, src in enumerate(frames):
        if frame_weights[idx] <= 0:
            raise RuntimeError(f"No output accumulated for frame {idx}")
        save_rgb_float(sums[idx] / frame_weights[idx], output / src.name)

    params = sum(p.numel() for p in model.parameters())
    write_json(output.parent / "run_metadata.json", {
        "model": "DSTNet+ Base",
        "paper": "TPAMI 2025",
        "training_checkpoint_dataset": "GoPro",
        "repo": str(repo),
        "checkpoint": str(checkpoint),
        "input": str(Path(args.input).resolve()),
        "output": str(output),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "num_feat": 64,
        "num_kernel_block": 3,
        "num_block": 15,
        "nonblind_denoise": False,
        "parameter_count": params,
        "clip_len": args.clip_len,
        "overlap": args.overlap,
        "requested_tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "min_tile_size": args.min_tile_size,
        "effective_tile_sizes": effective_tiles,
        "amp": args.amp,
        "dynamic_conv_backend": dynamic_backend,
        "runtime_seconds": time.time() - started,
        "torch": torch.__version__,
    })


if __name__ == "__main__":
    main()
