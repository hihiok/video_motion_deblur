"""Checkpoint compatibility and atomic save helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(payload).__name__}")
    for key in ("params_ema", "params", "state_dict", "model", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            payload = value
            break
    state = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[7:]
        state[key] = value
    if not state:
        raise ValueError("No tensor state_dict was found in the checkpoint")
    return state


def load_rtfocuser_pretrained(
    model: nn.Module,
    checkpoint: str | Path,
    min_target_coverage: float = 0.98,
) -> dict[str, Any]:
    """Load official frame weights into the pruned video backbone.

    The video model stores frame modules below ``backbone``.  Source LD blocks
    that were intentionally removed are reported as dropped source keys; every
    retained target tensor must still be loaded.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    source = unwrap_state_dict(raw)
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    shape_mismatch = []
    dropped_source = []
    for source_key, tensor in source.items():
        candidates = [source_key]
        if not source_key.startswith("backbone."):
            candidates.insert(0, f"backbone.{source_key}")
        target_key = next((key for key in candidates if key in target), None)
        if target_key is None:
            dropped_source.append(source_key)
            continue
        if target[target_key].shape != tensor.shape:
            shape_mismatch.append(
                (source_key, tuple(tensor.shape), target_key, tuple(target[target_key].shape))
            )
            continue
        matched[target_key] = tensor

    total_target = sum(t.numel() for t in target.values())
    loaded_target = sum(target[key].numel() for key in matched)
    coverage = loaded_target / max(total_target, 1)
    missing_target = sorted(set(target) - set(matched))
    if coverage < min_target_coverage:
        raise RuntimeError(
            f"RT-Focuser checkpoint target coverage {coverage:.3%} is below "
            f"required {min_target_coverage:.3%}; missing={missing_target[:20]}, "
            f"shape_mismatch={shape_mismatch[:10]}"
        )
    result = model.load_state_dict(matched, strict=False)
    return {
        "checkpoint": str(checkpoint),
        "target_coverage": coverage,
        "loaded_target_elements": loaded_target,
        "total_target_elements": total_target,
        "matched_tensors": len(matched),
        "missing_target_keys": missing_target,
        "dropped_source_keys": sorted(dropped_source),
        "shape_mismatch": shape_mismatch,
        "load_missing_keys": list(result.missing_keys),
        "load_unexpected_keys": list(result.unexpected_keys),
    }


def atomic_torch_save(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
