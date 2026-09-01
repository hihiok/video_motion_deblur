#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from generic_restoration.benchmark_utils import (
    git_commit,
    list_frames,
    sha256_file,
    write_json,
)
from generic_restoration.planning import tile_starts


def feather_axis(length: int, overlap: int, touches_start: bool, touches_end: bool) -> torch.Tensor:
    weights = torch.ones(length, dtype=torch.float32)
    ramp_length = min(overlap, max(length // 2, 1))
    if ramp_length and not touches_start:
        weights[:ramp_length] = torch.linspace(1.0 / (ramp_length + 1), 1.0, ramp_length)
    if ramp_length and not touches_end:
        weights[-ramp_length:] = torch.linspace(1.0, 1.0 / (ramp_length + 1), ramp_length)
    return weights


def feather_mask(
    height: int,
    width: int,
    overlap: int,
    top: int,
    left: int,
    full_height: int,
    full_width: int,
) -> torch.Tensor:
    wy = feather_axis(height, overlap, top == 0, top + height == full_height)
    wx = feather_axis(width, overlap, left == 0, left + width == full_width)
    return wy[:, None] * wx[None, :]


def load_model(repo: Path, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    sys.path.insert(0, str(repo))
    from archs.realviformer_arch import RealViformer  # type: ignore

    model = RealViformer(
        num_feat=48,
        num_blocks=[2, 3, 4, 1],
        spynet_path=None,
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        merge_head=2,
        bias=False,
        LayerNorm_type="BiasFree",
        ch_compress=True,
        squeeze_factor=[4, 4, 4],
        masked=True,
    )
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("params", raw) if isinstance(raw, dict) else raw
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not strictly match official RealViformer: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    model.eval().to(device)
    return model, {"missing_keys": missing, "unexpected_keys": unexpected}


def load_tile(paths: list[Path], top: int, left: int, height: int, width: int) -> torch.Tensor:
    frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        crop = rgb[top : top + height, left : left + width]
        frames.append(torch.from_numpy(crop).permute(2, 0, 1))
    return torch.stack(frames, dim=0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official fixed-4x RealViformer checkpoint and normalize its result back to 1x."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-frames", required=True)
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=48)
    parser.add_argument("--core-frames", type=int, default=8)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not (repo / "archs" / "realviformer_arch.py").is_file():
        raise FileNotFoundError(f"Invalid RealViformer repository: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    paths = list_frames(args.input_frames)
    with Image.open(paths[0]) as first:
        width, height = first.size
    output = Path(args.output_frames).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.png"):
        old.unlink()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    model, loading = load_model(repo, checkpoint, device)
    if args.precision == "fp16":
        model.half()
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    autocast = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" and args.precision == "fp16"
        else contextlib.nullcontext()
    )

    y_starts = tile_starts(height, args.tile, args.overlap)
    x_starts = tile_starts(width, args.tile, args.overlap)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for core_start in range(0, len(paths), args.core_frames):
            core_end = min(core_start + args.core_frames, len(paths))
            context_start = max(0, core_start - args.warmup_frames)
            context_paths = paths[context_start:core_end]
            keep_start = core_start - context_start
            keep_count = core_end - core_start
            accum = torch.zeros((keep_count, 3, height, width), dtype=torch.float32)
            weights = torch.zeros((1, 1, height, width), dtype=torch.float32)

            for top in y_starts:
                tile_h = min(args.tile, height - top) if args.tile > 0 else height
                for left in x_starts:
                    tile_w = min(args.tile, width - left) if args.tile > 0 else width
                    batch = load_tile(context_paths, top, left, tile_h, tile_w).to(device, dtype=dtype)
                    pad_h = (-tile_h) % 4
                    pad_w = (-tile_w) % 4
                    if pad_h or pad_w:
                        batch = F.pad(batch, (0, pad_w, 0, pad_h), mode="reflect")
                    with autocast:
                        prediction4x = model(batch).clamp_(0, 1)
                    prediction4x = prediction4x[:, keep_start : keep_start + keep_count]
                    if pad_h:
                        prediction4x = prediction4x[..., : tile_h * 4, :]
                    if pad_w:
                        prediction4x = prediction4x[..., : tile_w * 4]
                    flat = prediction4x.flatten(0, 1).float()
                    normalized = F.interpolate(flat, size=(tile_h, tile_w), mode="area")
                    normalized = normalized.view(keep_count, 3, tile_h, tile_w).float().cpu()
                    mask = feather_mask(
                        tile_h,
                        tile_w,
                        args.overlap,
                        top,
                        left,
                        height,
                        width,
                    )
                    accum[:, :, top : top + tile_h, left : left + tile_w] += normalized * mask
                    weights[:, :, top : top + tile_h, left : left + tile_w] += mask
                    del batch, prediction4x, flat, normalized

            if torch.any(weights <= 0):
                raise RuntimeError("Spatial tiling left uncovered pixels")
            restored = (accum / weights).clamp_(0, 1)
            if not torch.isfinite(restored).all():
                raise RuntimeError("RealViformer produced NaN/Inf")
            for offset, tensor in enumerate(restored):
                array = tensor.mul(255).round().byte().permute(1, 2, 0).numpy()
                Image.fromarray(array, mode="RGB").save(output / f"{core_start + offset:08d}.png")
            print(f"RealViformer: {core_end}/{len(paths)} frames", flush=True)

    elapsed = time.perf_counter() - started
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    metadata = {
        "model": "RealViformer",
        "official_repo": "https://github.com/Yuehan717/RealViformer",
        "official_commit": git_commit(repo),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_loading": loading,
        "input_frame_count": len(paths),
        "output_frame_count": len(list_frames(output)),
        "input_size": [width, height],
        "output_size": [width, height],
        "inference_path": "official fixed-4x network, then area downsample to source size",
        "tile": args.tile,
        "overlap": args.overlap,
        "core_frames": args.core_frames,
        "warmup_frames": args.warmup_frames,
        "precision": args.precision,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_gib": peak_gib,
    }
    write_json(args.metadata, metadata)
    print(args.metadata)


if __name__ == "__main__":
    main()
