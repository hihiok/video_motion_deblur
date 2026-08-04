#!/usr/bin/env python3
"""Shared utilities for frame-folder video restoration inference."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def list_frames(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {folder}")
    frames = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )
    if not frames:
        raise ValueError(f"No image frames found in {folder}")
    return frames


def inspect_frames(frames: Sequence[Path]) -> Tuple[int, int]:
    with Image.open(frames[0]) as im:
        size = im.convert("RGB").size
    for p in frames[1:]:
        with Image.open(p) as im:
            if im.convert("RGB").size != size:
                raise ValueError(f"Frame size mismatch at {p}: {im.size} != {size}")
    return size[1], size[0]


def load_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb_float(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB").save(path)


def import_repo(repo: str | Path) -> Path:
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Model repository not found: {repo}")
    sys.path.insert(0, str(repo))
    return repo


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("params", "params_ema", "state_dict", "model", "net_g"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    cleaned = {}
    for key, value in checkpoint.items():
        for prefix in ("module.", "net_g.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def temporal_chunks(length: int, clip_len: int, overlap: int) -> List[Tuple[int, int]]:
    if length <= 0:
        return []
    if clip_len <= 0:
        raise ValueError("clip_len must be positive")
    if overlap < 0 or overlap >= clip_len:
        raise ValueError("overlap must satisfy 0 <= overlap < clip_len")
    if length <= clip_len:
        return [(0, length)]
    stride = clip_len - overlap
    starts = list(range(0, max(length - clip_len + 1, 1), stride))
    last = length - clip_len
    if starts[-1] != last:
        starts.append(last)
    return [(s, min(s + clip_len, length)) for s in starts]


def blend_weights(length: int, start: int, end: int, total: int, overlap: int) -> np.ndarray:
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 1:
        return w
    ramp = min(overlap, length)
    if start > 0:
        w[:ramp] = np.linspace(1.0 / (ramp + 1), 1.0, ramp, dtype=np.float32)
    if end < total:
        w[-ramp:] = np.minimum(
            w[-ramp:], np.linspace(1.0, 1.0 / (ramp + 1), ramp, dtype=np.float32)
        )
    return w


def reflection_indices(start: int, end: int, total: int) -> List[int]:
    def reflect(i: int) -> int:
        if total <= 1:
            return 0
        period = 2 * total - 2
        j = i % period
        return j if j < total else period - j

    return [reflect(i) for i in range(start, end)]


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
