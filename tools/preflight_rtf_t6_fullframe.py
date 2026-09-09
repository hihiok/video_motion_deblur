#!/usr/bin/env python3
"""Prove that native-resolution T=6 training fits before a formal run."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtf_t6.checkpoint import load_rtfocuser_pretrained
from rtf_t6.datasets import build_domain_sequences, load_clip
from rtf_t6.losses import VideoDeblurLoss
from rtf_t6.model import RTFocuserT6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    temporary.replace(path)


def image_area(sequence) -> int:
    with Image.open(sequence.blur[0]) as image:
        width, height = image.size
    return width * height


def check_config(config: dict) -> None:
    train = config["train"]
    validation = config.get("validation", {})
    if int(train.get("clip_length", 0)) != 6:
        raise ValueError("Full-frame preflight is locked to T=6")
    if str(train.get("spatial_mode", "")).lower() != "full_frame":
        raise ValueError("train.spatial_mode must be full_frame")
    if int(train.get("crop_size", -1)) != 0:
        raise ValueError("train.crop_size must be 0 (native full frame)")
    if int(validation.get("crop_size", -1)) != 0:
        raise ValueError("validation.crop_size must be 0 (native full frame)")
    if int(train.get("batch_size", 0)) != 1:
        raise ValueError("Native full-frame training requires batch_size=1")
    if not bool(config.get("model", {}).get("activation_checkpointing", False)):
        raise ValueError("activation_checkpointing must be enabled for the full-frame run")


def main() -> int:
    args = arguments()
    output = Path(args.output).expanduser().resolve()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    check_config(config)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the full-frame memory preflight")

    domains, roots = build_domain_sequences(config["datasets"], "train", 6)
    model = RTFocuserT6(**config.get("model", {}))
    initialization = load_rtfocuser_pretrained(model, args.pretrained)
    model = model.to(device).train()
    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    amp = bool(train_cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = VideoDeblurLoss(**config.get("loss", {})).to(device)
    loss_cfg = config.get("loss", {})
    loss_iteration = (
        int(loss_cfg.get("temporal_start_iter", 0))
        + int(loss_cfg.get("temporal_ramp_iters", 0))
        + 1
    )

    report = {
        "status": "IN_PROGRESS",
        "config": str(Path(args.config).expanduser().resolve()),
        "pretrained": str(Path(args.pretrained).expanduser().resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "roots": roots,
        "target_coverage": initialization["target_coverage"],
        "clip_length": 6,
        "spatial_mode": "full_frame",
        "crop_applied": False,
        "resize_applied": False,
        "activation_checkpointing": True,
        "domains": {},
    }

    try:
        for domain, sequences in sorted(domains.items()):
            sequence = max(sequences, key=image_area)
            start = max((sequence.length - 6) // 2, 0)
            indices = list(range(start, start + 6))
            with Image.open(sequence.blur[start]) as image:
                native_width, native_height = image.size
            blur, gt = load_clip(
                sequence,
                indices,
                crop_size=0,
                rng=random.Random(20260909),
                augment=False,
            )
            expected_shape = (6, 3, native_height, native_width)
            if tuple(blur.shape) != expected_shape or tuple(gt.shape) != expected_shape:
                raise RuntimeError(
                    f"Native-shape violation for {domain}/{sequence.name}: "
                    f"{tuple(blur.shape)} and {tuple(gt.shape)} vs {expected_shape}"
                )

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            blur = blur.unsqueeze(0).to(device, non_blocking=True)
            gt = gt.unsqueeze(0).to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                prediction = model(blur)
                loss, components = criterion(prediction, gt, loss_iteration)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for domain {domain}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            if not gradients_finite:
                raise FloatingPointError(f"Non-finite gradient for domain {domain}")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.get("clip_grad", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            report["domains"][domain] = {
                "sequence": sequence.name,
                "start": start,
                "native_shape": list(expected_shape),
                "output_shape": list(prediction.shape[1:]),
                "loss": float(loss.detach()),
                "loss_components": {
                    key: float(value.detach()) for key, value in components.items()
                },
                "gradient_norm": float(grad_norm),
                "elapsed_seconds": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            del blur, gt, prediction, loss, components

    except torch.cuda.OutOfMemoryError as error:
        report["status"] = "FULLFRAME_PREFLIGHT_OOM"
        report["error"] = str(error)
        write_report(output, report)
        print(json.dumps(report, indent=2))
        print("FULLFRAME_PREFLIGHT_OOM")
        return 3

    report["status"] = "FULLFRAME_PREFLIGHT_PASS"
    write_report(output, report)
    print(json.dumps(report, indent=2))
    print("FULLFRAME_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
