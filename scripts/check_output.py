#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def frames(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted([p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts], key=key)


def read(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()

    inp, out = frames(args.input), frames(args.output)
    errors, warnings = [], []
    if len(inp) != len(out):
        errors.append(f"frame count: input={len(inp)}, output={len(out)}")
    count = min(len(inp), len(out))
    diffs, lap_vars, clip_ratios = [], [], []
    for i in range(count):
        a, b = read(inp[i]), read(out[i])
        if a.shape != b.shape:
            errors.append(f"shape mismatch frame {i}: {a.shape} vs {b.shape}")
            continue
        diff = float(np.abs(a - b).mean())
        diffs.append(diff)
        gray = b.mean(axis=2)
        lap = (-4 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))
        lap_vars.append(float(lap.var()))
        clip_ratios.append(float(((b < 1/255) | (b > 254/255)).mean()))
    mean_diff = float(np.mean(diffs)) if diffs else None
    if mean_diff is not None and mean_diff < 0.5 / 255:
        warnings.append("Output is nearly identical to input; possible identity/broken checkpoint.")
    if mean_diff is not None and mean_diff > 80 / 255:
        warnings.append("Output differs extremely from input; check RGB/range/checkpoint.")
    report = {
        "input_count": len(inp), "output_count": len(out),
        "mean_abs_input_output": mean_diff,
        "mean_laplacian_variance": float(np.mean(lap_vars)) if lap_vars else None,
        "mean_clip_ratio": float(np.mean(clip_ratios)) if clip_ratios else None,
        "errors": errors, "warnings": warnings,
        "passed": not errors,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
