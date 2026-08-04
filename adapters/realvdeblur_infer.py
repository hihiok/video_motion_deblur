#!/usr/bin/env python3
"""Thin, logged launcher for the official OpenImagingLab RealVDeblur inference."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import inspect_frames, list_frames, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--wan-model-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--temporal-window-size", type=int, default=21)
    p.add_argument("--num-inference-steps", type=int, default=1)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    inference = repo / "inference.py"
    if not inference.is_file():
        raise FileNotFoundError(f"Official inference.py not found: {inference}")
    frames = list_frames(args.input)
    h, w = inspect_frames(frames)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(inference),
        "--input", str(Path(args.input).resolve()),
        "--output", str(out),
        "--wan_model_dir", str(Path(args.wan_model_dir).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--device", args.device,
        "--dtype", args.dtype,
        "--num_inference_steps", str(args.num_inference_steps),
        "--enable_twm",
        "--temporal_window_size", str(args.temporal_window_size),
    ]
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("--height and --width must be specified together")
        command += ["--height", str(args.height), "--width", str(args.width)]

    started = time.time()
    subprocess.run(command, cwd=repo, check=True)
    produced = list_frames(out)
    if len(produced) != len(frames):
        raise RuntimeError(f"RealVDeblur produced {len(produced)} frames; expected {len(frames)}")

    write_json(out.parent / "run_metadata.json", {
        "model": "RealVDeblur",
        "repo": str(repo),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "wan_model_dir": str(Path(args.wan_model_dir).resolve()),
        "input_frame_count": len(frames),
        "output_frame_count": len(produced),
        "input_width": w,
        "input_height": h,
        "dtype": args.dtype,
        "temporal_window_size": args.temporal_window_size,
        "num_inference_steps": args.num_inference_steps,
        "runtime_seconds": time.time() - started,
        "command": command,
    })


if __name__ == "__main__":
    main()
