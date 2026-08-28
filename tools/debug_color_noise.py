#!/usr/bin/env python3
"""Detect likely R/B swaps, blue drift, and suspicious high-frequency amplification."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def list_images(folder: Path):
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS), key=natural_key)
    if not files:
        raise ValueError(f"No images in {folder}")
    return files


def load_rgb(path: Path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def laplacian_rms(rgb: np.ndarray):
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    center = gray[1:-1, 1:-1]
    lap = -4.0 * center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.sqrt(np.mean(lap * lap) + 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--samples", type=int, default=32)
    args = p.parse_args()

    inputs = list_images(Path(args.input))
    outputs = list_images(Path(args.output))
    input_by_name = {p.name: p for p in inputs}
    pairs = [(input_by_name[p.name], p) for p in outputs if p.name in input_by_name]
    if not pairs:
        if len(inputs) != len(outputs):
            raise ValueError("No matching filenames and frame counts differ")
        pairs = list(zip(inputs, outputs))
    sample_count = min(max(1, args.samples), len(pairs))
    indices = np.linspace(0, len(pairs) - 1, sample_count, dtype=int)

    maes, swapped_maes, input_means, output_means, hf_ratios = [], [], [], [], []
    for index in indices:
        input_path, output_path = pairs[int(index)]
        inp, out = load_rgb(input_path), load_rgb(output_path)
        if inp.shape != out.shape:
            raise ValueError(f"Shape mismatch: {input_path} {inp.shape}, {output_path} {out.shape}")
        maes.append(float(np.mean(np.abs(out - inp))))
        swapped_maes.append(float(np.mean(np.abs(out[..., ::-1] - inp))))
        input_means.append(inp.mean(axis=(0, 1)))
        output_means.append(out.mean(axis=(0, 1)))
        hf_ratios.append(laplacian_rms(out) / max(laplacian_rms(inp), 1e-6))

    mae = float(np.mean(maes))
    swapped_mae = float(np.mean(swapped_maes))
    input_mean = np.mean(input_means, axis=0)
    output_mean = np.mean(output_means, axis=0)
    rb_swap_likely = bool(swapped_mae + 0.02 < mae and swapped_mae < 0.8 * mae)
    input_blue_minus_red = float(input_mean[2] - input_mean[0])
    output_blue_minus_red = float(output_mean[2] - output_mean[0])
    blue_drift = output_blue_minus_red - input_blue_minus_red
    hf_ratio_median = float(np.median(hf_ratios))

    report = {
        "input": str(Path(args.input).resolve()),
        "output": str(Path(args.output).resolve()),
        "paired_frames": len(pairs),
        "sampled_frames": sample_count,
        "rgb_mae_to_input": mae,
        "rb_swapped_mae_to_input": swapped_mae,
        "rb_swap_likely": rb_swap_likely,
        "input_rgb_mean": input_mean.tolist(),
        "output_rgb_mean": output_mean.tolist(),
        "blue_minus_red_drift": blue_drift,
        "blue_cast_warning": bool(blue_drift > 0.08),
        "median_laplacian_rms_ratio": hf_ratio_median,
        "texture_noise_warning": bool(hf_ratio_median > 3.0),
        "interpretation": {
            "rb_swap": "Hard failure if true; inspect RGB/BGR conversions.",
            "blue_cast": "Warning only; scene/model domain can also change channel means.",
            "texture_noise": "Warning only; inspect checkpoint/architecture, normalization, and FP16 before judging quality.",
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if rb_swap_likely:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
