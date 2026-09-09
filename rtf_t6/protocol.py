"""Shared T3/T6 protocol checks; module names retain checkpoint compatibility."""
from __future__ import annotations

import hashlib
from pathlib import Path


def clip_length(config: dict) -> int:
    length = int(config["train"].get("clip_length", 6))
    if length not in (3, 6):
        raise ValueError(f"Supported training protocols are T=3 and T=6, got {length}")
    return length


def check_fullframe(config: dict) -> None:
    clip_length(config)
    train = config["train"]
    validation = config.get("validation", {})
    if train.get("spatial_mode") != "full_frame":
        raise ValueError("train.spatial_mode must be full_frame")
    if any(int(section.get("crop_size", -1)) != 0 for section in (train, validation)):
        raise ValueError("Training and validation crop_size must be 0")
    if any(int(section.get("batch_size", 1)) != 1 for section in (train, validation)):
        raise ValueError("Native full-frame batches must have batch_size=1")
    if not config.get("model", {}).get("activation_checkpointing", False):
        raise ValueError("Full-frame training requires activation_checkpointing")
    accumulation = int(train.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    for key in ("total_iters", "validate_every", "save_every", "samples_per_epoch"):
        value = int(train[key])
        if value <= 0 or value % accumulation:
            raise ValueError(f"{key} must be positive and divisible by gradient_accumulation")


def check_checkpoint_protocol(payload: dict, config: dict, *, resume: bool = False) -> None:
    saved = payload.get("config")
    if saved is None:
        if resume:
            raise ValueError("Resume checkpoint is missing its training config")
        return  # Official/bare model weights have no training protocol metadata.
    if clip_length(saved) != clip_length(config):
        raise ValueError("Checkpoint clip_length differs from the requested protocol")
    if resume:
        if saved.get("model", {}) != config.get("model", {}):
            raise ValueError("Resume model configuration differs")
        if saved.get("loss", {}) != config.get("loss", {}):
            raise ValueError("Resume loss configuration differs")
        for key in ("spatial_mode", "crop_size", "batch_size", "gradient_accumulation",
                    "total_iters", "lr", "min_lr", "warmup_iters", "ema_decay"):
            if saved["train"].get(key) != config["train"].get(key):
                raise ValueError(f"Resume train.{key} differs")
        accumulation = int(config["train"].get("gradient_accumulation", 1))
        if int(payload["iteration"]) % accumulation:
            raise ValueError("Resume checkpoint is not at an optimizer-update boundary")


def inference_window(config: dict, window: int | None, overlap: int | None) -> tuple[int, int]:
    length = clip_length(config)
    window = length if window is None else window
    overlap = (2 if length == 3 else 4) if overlap is None else overlap
    if window != length:
        raise ValueError(f"Inference window must match configured T={length}")
    if overlap < 0 or overlap >= window:
        raise ValueError("temporal_overlap must satisfy 0 <= overlap < window")
    return window, overlap


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
