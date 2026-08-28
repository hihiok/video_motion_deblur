#!/usr/bin/env python3
"""Strict ESTRNN BSD 3ms-24ms inference with explicit RGB/BGR boundaries.

The official ESTRNN inference reads and writes with OpenCV, so the checkpoint's
model-facing channel order is BGR.  This adapter keeps user files RGB, converts
RGB->BGR immediately before the model, and converts BGR->RGB immediately after.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import inspect_frames, list_frames, load_rgb_float, reflection_indices, save_rgb_float, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Official zzh-tech/ESTRNN checkout")
    p.add_argument("--input", required=True, help="RGB frame directory")
    p.add_argument("--output", required=True, help="Output RGB frame directory")
    p.add_argument("--checkpoint", required=True, help="ESTRNN_C80B15_BSD_3ms24ms.tar")
    p.add_argument("--chunk-size", type=int, default=16, help="Number of saved outputs per inference chunk")
    p.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true", help="Only enable after an FP32 baseline passes")
    return p.parse_args()


def import_official(repo: str | Path):
    repo = Path(repo).resolve()
    required = [repo / "model" / "ESTRNN.py", repo / "para" / "parameter.py", repo / "data" / "utils.py"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete official ESTRNN repo: " + ", ".join(missing))
    sys.path.insert(0, str(repo))
    from data.utils import normalize, normalize_reverse  # type: ignore
    from model import Model  # type: ignore
    from para import Parameter  # type: ignore

    return Model, Parameter, normalize, normalize_reverse, repo


def strip_dataparallel_prefix(state):
    if not isinstance(state, dict):
        raise TypeError(f"Expected state dict, got {type(state).__name__}")
    result = {}
    for key, value in state.items():
        result[key[7:] if key.startswith("module.") else key] = value
    return result


def pad_video(x: torch.Tensor, multiple: int = 4):
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
            f"Likely R/B swap after ESTRNN: RGB MAE={normal:.5f}, swapped-RB MAE={swapped:.5f}"
        )
    return normal, swapped


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    checkpoint_path = Path(args.checkpoint).resolve()
    required_tokens = ("c80b15", "bsd", "3ms24ms")
    lower_name = checkpoint_path.name.lower()
    if not all(token in lower_name for token in required_tokens):
        raise ValueError(f"Expected ESTRNN_C80B15_BSD_3ms24ms checkpoint, got {checkpoint_path.name}")

    Model, Parameter, normalize, normalize_reverse, repo = import_official(args.repo)
    para = Parameter().args
    para.model = "ESTRNN"
    para.n_features = 16  # 5 * 16 = C80 in the official checkpoint name
    para.n_blocks = 15
    para.future_frames = 2
    para.past_frames = 2
    para.activation = "gelu"
    para.normalize = True
    para.centralize = True

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official ESTRNN hard-codes CUDA hidden-state allocation; use an available CUDA device")
    torch.cuda.set_device(device)
    model = Model(para).to(device).eval()
    parameter_count = sum(p.numel() for p in model.parameters())

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise KeyError("ESTRNN checkpoint must contain a state_dict mapping")
    state = strip_dataparallel_prefix(state)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    if args.fp16:
        model.half()

    frames = list_frames(args.input)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    height, width = inspect_frames(frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    first_input = load_rgb_float(frames[0])
    first_output = None
    started = time.time()
    val_range = 255.0

    with torch.inference_mode():
        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            ids = reflection_indices(start - para.past_frames, end + para.future_frames, n)
            rgb = np.stack([load_rgb_float(frames[i]) for i in ids])
            model_bgr_255 = np.ascontiguousarray(rgb[..., ::-1] * val_range)
            x = torch.from_numpy(model_bgr_255).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            x = pad_video(x)
            x = normalize(x, centralize=True, normalize=True, val_range=val_range)
            if args.fp16:
                x = x.half()

            prediction = model([x])
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            if prediction.ndim != 5 or prediction.shape[0] != 1:
                raise RuntimeError(f"Unexpected ESTRNN output shape {tuple(prediction.shape)}")
            prediction = prediction[0]
            expected = end - start
            if prediction.shape[0] != expected:
                raise RuntimeError(
                    f"ESTRNN returned {prediction.shape[0]} frames from {len(ids)} inputs; expected {expected}"
                )
            prediction = normalize_reverse(
                prediction.float(), centralize=True, normalize=True, val_range=val_range
            )
            model_bgr = prediction.cpu()[:, :, :height, :width].permute(0, 2, 3, 1).numpy() / val_range
            output_rgb = np.clip(model_bgr[..., ::-1], 0.0, 1.0)
            if not np.isfinite(output_rgb).all():
                raise RuntimeError("Model output contains NaN or Inf")
            if first_output is None:
                first_output = output_rgb[0].copy()
            for local, frame_index in enumerate(range(start, end)):
                save_rgb_float(output_rgb[local], output_dir / frames[frame_index].name)
            print(f"ESTRNN BSD 3ms-24ms chunk [{start}, {end})")

    assert first_output is not None
    rgb_mae, swapped_mae = rb_swap_check(first_input, first_output)
    metadata = {
        "model": "ESTRNN C80B15 BSD 3ms-24ms",
        "color_contract": "file RGB -> official model BGR -> file RGB (Pillow)",
        "repo": str(repo),
        "checkpoint": str(checkpoint_path),
        "checkpoint_shape": {"n_features": 16, "n_blocks": 15, "past": 2, "future": 2},
        "strict_checkpoint_load": True,
        "parameter_count": parameter_count,
        "frame_count": n,
        "width": width,
        "height": height,
        "chunk_size": args.chunk_size,
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
    print(f"PASS: strict C80B15 checkpoint, RGB<->BGR boundary, R/B check. params={parameter_count:,}")


if __name__ == "__main__":
    main()
