#!/usr/bin/env python3
"""Run a trained RT-Focuser-T6 checkpoint on a frame directory."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtf_t6.checkpoint import unwrap_state_dict
from rtf_t6.datasets import IMAGE_EXTENSIONS, read_rgb
from rtf_t6.inference import infer_spatial_tiles, owned_temporal_range, window_starts
from rtf_t6.model import RTFocuserT6


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((array * 255.0 + 0.5).astype("uint8"), mode="RGB").save(path)


def load_model(config: dict, checkpoint: Path, device: torch.device) -> RTFocuserT6:
    model = RTFocuserT6(**config.get("model", {}))
    payload = torch.load(checkpoint, map_location="cpu")
    state = unwrap_state_dict(payload)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(str(result))
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--temporal-overlap", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-overlap", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.window != 6:
        raise ValueError("This trained protocol is locked to a six-frame temporal window")
    input_dir = Path(args.input).expanduser().resolve()
    frames = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        raise FileNotFoundError(f"No input images in {input_dir}")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output}")
    device = torch.device(args.device)
    model = load_model(config, Path(args.checkpoint).expanduser().resolve(), device)
    starts = window_starts(len(frames), args.window, args.temporal_overlap)
    dtype = torch.float16 if args.amp and device.type == "cuda" else None
    started = time.time()
    written = 0
    with torch.inference_mode():
        for window_index, start in enumerate(starts):
            end = min(start + args.window, len(frames))
            clip = torch.stack([read_rgb(path) for path in frames[start:end]])
            clip = clip.unsqueeze(0).to(device)
            prediction = infer_spatial_tiles(
                model, clip, args.tile_size, args.tile_overlap, dtype
            )[0]
            own_start, own_end = owned_temporal_range(
                starts, window_index, end, args.window, len(frames)
            )
            for frame_index in range(own_start, own_end):
                save_rgb(prediction[frame_index - start], output / frames[frame_index].name)
                written += 1
            print(f"window {window_index + 1}/{len(starts)} [{start},{end}) wrote [{own_start},{own_end})")
    output_frames = sorted(path for path in output.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if written != len(frames) or len(output_frames) != len(frames):
        raise RuntimeError(f"Output count mismatch: input={len(frames)}, written={written}, files={len(output_frames)}")
    metadata = {
        "model": "RT-Focuser-T6 (Shift/DST temporal fusion)",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "input": str(input_dir),
        "output": str(output),
        "frame_count": len(frames),
        "window": args.window,
        "temporal_overlap": args.temporal_overlap,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "amp": bool(args.amp),
        "runtime_seconds": time.time() - started,
    }
    (output.parent / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("INFERENCE_PASS")


if __name__ == "__main__":
    main()
