#!/usr/bin/env python3
"""Strict inference wrapper for official Shift-Net Ours-s.

Color contract: files are RGB, tensors are RGB in [0, 1], and files are saved
with Pillow as RGB.  OpenCV is deliberately not used in this adapter.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    inspect_frames,
    list_frames,
    load_rgb_float,
    reflection_indices,
    save_rgb_float,
    unwrap_state_dict,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Official dasongli1/Shift-Net checkout")
    p.add_argument("--input", required=True, help="RGB frame directory")
    p.add_argument("--output", required=True, help="Output RGB frame directory")
    p.add_argument("--checkpoint", required=True, help="net_*_deblur_small.pth")
    p.add_argument("--one-len", type=int, default=48, help="Number of output frames per chunk")
    p.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true", help="Only enable after an FP32 baseline passes")
    return p.parse_args()


def load_small_model_class(repo: str | Path):
    repo = Path(repo).resolve()
    candidates = [
        repo / "basicsr" / "models" / "archs" / "gshift_deblur2.py",
        repo / "basicsr" / "archs" / "gshift_deblur2.py",
    ]
    architecture = next((p for p in candidates if p.is_file()), None)
    if architecture is None:
        raise FileNotFoundError("Ours-s architecture gshift_deblur2.py not found: " + ", ".join(map(str, candidates)))
    spec = importlib.util.spec_from_file_location("shiftnet_official_ours_s", architecture)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {architecture}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "GShiftNet"):
        raise ImportError(f"GShiftNet missing from {architecture}")
    return module.GShiftNet, architecture


def pad_video(x: torch.Tensor, multiple: int = 4):
    if x.ndim != 5:
        raise ValueError(f"Expected B,T,C,H,W, got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return x.contiguous()
    flat = x.reshape(b * t, c, h, w)
    mode = "reflect" if h > ph and w > pw and h > 1 and w > 1 else "replicate"
    flat = F.pad(flat, (0, pw, 0, ph), mode=mode)
    return flat.reshape(b, t, c, h + ph, w + pw).contiguous()


def rb_swap_check(reference_rgb: np.ndarray, output_rgb: np.ndarray):
    normal = float(np.mean(np.abs(output_rgb - reference_rgb)))
    swapped = float(np.mean(np.abs(output_rgb[..., ::-1] - reference_rgb)))
    if swapped + 0.02 < normal and swapped < 0.8 * normal:
        raise RuntimeError(
            f"Likely R/B swap: RGB MAE={normal:.5f}, swapped-RB MAE={swapped:.5f}. "
            "Do not publish these frames."
        )
    return normal, swapped


def main():
    args = parse_args()
    if args.one_len < 1:
        raise ValueError("--one-len must be positive")
    checkpoint_path = Path(args.checkpoint).resolve()
    if "small" not in checkpoint_path.name.lower():
        raise ValueError(f"Ours-s requires a *_small checkpoint, got {checkpoint_path.name}")

    GShiftNet, architecture = load_small_model_class(args.repo)
    frames = list_frames(args.input)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    height, width = inspect_frames(frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = GShiftNet(future_frames=2, past_frames=2).to(device).eval()
    parameter_count = sum(p.numel() for p in model.parameters())

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = unwrap_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    if args.fp16:
        model.half()

    n = len(frames)
    first_input = load_rgb_float(frames[0])
    first_output = None
    started = time.time()
    with torch.inference_mode():
        for start in range(0, n, args.one_len):
            end = min(start + args.one_len, n)
            ids = reflection_indices(start - 2, end + 2, n)
            arrays = [load_rgb_float(frames[i]) for i in ids]
            x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            x = pad_video(x)
            if args.fp16:
                x = x.half()
            prediction = model(x.contiguous())
            if prediction.ndim == 5 and prediction.shape[0] == 1:
                prediction = prediction[0]
            expected = end - start
            if prediction.ndim != 4 or prediction.shape[0] != expected:
                raise RuntimeError(
                    f"Ours-s returned {tuple(prediction.shape)} from {len(ids)} inputs; expected ({expected},C,H,W)"
                )
            prediction = prediction.float().cpu()[:, :, :height, :width].permute(0, 2, 3, 1).numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError("Model output contains NaN or Inf")
            prediction = np.clip(prediction, 0.0, 1.0)
            if first_output is None:
                first_output = prediction[0].copy()
            for local, frame_index in enumerate(range(start, end)):
                save_rgb_float(prediction[local], output_dir / frames[frame_index].name)
            print(f"Shift-Net Ours-s chunk [{start}, {end})")

    assert first_output is not None
    rgb_mae, swapped_mae = rb_swap_check(first_input, first_output)
    metadata = {
        "model": "Shift-Net Ours-s",
        "color_contract": "file RGB -> tensor RGB [0,1] -> file RGB (Pillow)",
        "repo": str(Path(args.repo).resolve()),
        "architecture_file": str(architecture),
        "architecture_required": "gshift_deblur2.py",
        "checkpoint": str(checkpoint_path),
        "strict_checkpoint_load": True,
        "parameter_count": parameter_count,
        "frame_count": n,
        "width": width,
        "height": height,
        "one_len": args.one_len,
        "dtype": "float16" if args.fp16 else "float32",
        "first_input_rgb_mean": first_input.mean(axis=(0, 1)).tolist(),
        "first_output_rgb_mean": first_output.mean(axis=(0, 1)).tolist(),
        "first_frame_rgb_mae": rgb_mae,
        "first_frame_rb_swapped_mae": swapped_mae,
        "runtime_seconds": time.time() - started,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    write_json(output_dir.parent / f"{output_dir.name}_metadata.json", metadata)
    print(f"PASS: strict Ours-s checkpoint, RGB contract, R/B check. params={parameter_count:,}")


if __name__ == "__main__":
    main()
