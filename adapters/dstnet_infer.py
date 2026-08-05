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
    p.add_argument("--clip-len", type=int, default=4)
    p.add_argument("--overlap", type=int, default=1)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--tile-overlap", type=int, default=64)
    p.add_argument("--min-tile-size", type=int, default=256)
    p.add_argument("--num-feat", type=int, default=64)
    p.add_argument("--num-block", type=int, default=15)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def pad_video_tensor(x: torch.Tensor, multiple: int = 4):
    """Pad B,T,C,H,W by reshaping to 4D first."""
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


def spatial_blend_weight(
    tile_h: int,
    tile_w: int,
    y0: int,
    x0: int,
    full_h: int,
    full_w: int,
    overlap: int,
) -> np.ndarray:
    wy = np.ones(tile_h, dtype=np.float32)
    wx = np.ones(tile_w, dtype=np.float32)

    if overlap > 0:
        if y0 > 0:
            n = min(overlap, tile_h)
            wy[:n] *= np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
        if y0 + tile_h < full_h:
            n = min(overlap, tile_h)
            wy[-n:] *= np.linspace(1.0, 0.0, n + 2, dtype=np.float32)[1:-1]
        if x0 > 0:
            n = min(overlap, tile_w)
            wx[:n] *= np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
        if x0 + tile_w < full_w:
            n = min(overlap, tile_w)
            wx[-n:] *= np.linspace(1.0, 0.0, n + 2, dtype=np.float32)[1:-1]

    return wy[:, None] * wx[None, :]


def infer_chunk_tiled(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    tile_size: int,
    tile_overlap: int,
    amp: bool,
    device: torch.device,
) -> np.ndarray:
    """Infer one B,T,C,H,W chunk using overlapping spatial tiles."""
    if tensor.ndim != 5 or tensor.shape[0] != 1:
        raise ValueError(f"Expected one B,T,C,H,W chunk, got {tuple(tensor.shape)}")

    _, time_len, _, height, width = tensor.shape
    y_starts = tile_starts(height, tile_size, tile_overlap)
    x_starts = tile_starts(width, tile_size, tile_overlap)

    accum = np.zeros((time_len, height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for y0 in y_starts:
        y1 = min(y0 + tile_size, height) if tile_size > 0 else height
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width) if tile_size > 0 else width
            tile = tensor[..., y0:y1, x0:x1].contiguous()
            tile_h, tile_w = y1 - y0, x1 - x0
            tile, _, _ = pad_video_tensor(tile, 4)

            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                pred = model(tile)

            pred_np = (
                pred[0, :, :, :tile_h, :tile_w]
                .float()
                .cpu()
                .permute(0, 2, 3, 1)
                .numpy()
            )
            blend = spatial_blend_weight(
                tile_h, tile_w, y0, x0, height, width, tile_overlap
            )
            accum[:, y0:y1, x0:x1] += pred_np * blend[None, :, :, None]
            weight_sum[y0:y1, x0:x1] += blend

            del tile, pred, pred_np

    if np.any(weight_sum <= 0):
        raise RuntimeError("DSTNet tiled inference produced uncovered pixels")
    return accum / weight_sum[None, :, :, None]


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def main():
    args = parse_args()
    if args.clip_len < 1:
        raise ValueError("clip-len must be at least 1")
    if args.overlap < 0 or args.overlap >= args.clip_len:
        raise ValueError("overlap must satisfy 0 <= overlap < clip-len")
    if args.tile_size < args.min_tile_size:
        raise ValueError("tile-size must be >= min-tile-size")

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
    effective_tile_sizes: list[int] = []
    started = time.time()

    with torch.inference_mode():
        for chunk_id, (start, end) in enumerate(chunks):
            arrays = [load_rgb_float(p) for p in frames[start:end]]
            tensor = (
                torch.from_numpy(np.stack(arrays))
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to(device)
                .contiguous()
            )

            current_tile = args.tile_size
            while True:
                try:
                    pred = infer_chunk_tiled(
                        model=model,
                        tensor=tensor,
                        tile_size=current_tile,
                        tile_overlap=min(args.tile_overlap, max(current_tile // 4, 1)),
                        amp=args.amp,
                        device=device,
                    )
                    effective_tile_sizes.append(current_tile)
                    break
                except BaseException as exc:
                    if not is_cuda_oom(exc) or current_tile <= args.min_tile_size:
                        raise
                    next_tile = max(args.min_tile_size, ((current_tile // 2) // 4) * 4)
                    if next_tile >= current_tile:
                        raise
                    print(
                        f"CUDA OOM for DSTNet chunk [{start}, {end}) at tile={current_tile}; "
                        f"retrying with tile={next_tile}"
                    )
                    del exc
                    torch.cuda.empty_cache()
                    current_tile = next_tile

            if pred.shape[0] != end - start:
                raise RuntimeError(
                    f"DSTNet returned {pred.shape[0]} frames for chunk length {end-start}"
                )
            local_w = blend_weights(end - start, start, end, len(frames), args.overlap)
            for j, idx in enumerate(range(start, end)):
                sums[idx] += pred[j] * local_w[j]
                weights[idx] += local_w[j]

            del tensor, pred, arrays
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"DSTNet chunk {chunk_id+1}/{len(chunks)}: [{start}, {end}); "
                f"dynamic_backend={dynamic_backend}; tile={effective_tile_sizes[-1]}"
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
        "requested_tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "min_tile_size": args.min_tile_size,
        "effective_tile_sizes": effective_tile_sizes,
        "amp": args.amp,
        "dynamic_conv_backend": dynamic_backend,
        "runtime_seconds": time.time() - started,
        "torch": torch.__version__,
    })


if __name__ == "__main__":
    main()
