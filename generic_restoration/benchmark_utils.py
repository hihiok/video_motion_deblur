#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(value: str | Path) -> list[int | str]:
    name = Path(value).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_frames(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {root}")
    frames = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if not frames:
        raise RuntimeError(f"No image frames found in: {root}")
    return frames


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_frames(frames: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.name.encode("utf-8"))
        digest.update(b"\0")
        with frame.open("rb") as handle:
            while block := handle.read(4 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str], *, cwd: str | Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def command_json(command: list[str]) -> dict[str, Any]:
    raw = subprocess.check_output(command, text=True)
    return json.loads(raw)


def ffprobe_video(path: str | Path, include_frames: bool = False) -> dict[str, Any]:
    video = str(Path(path).resolve())
    entries = (
        "stream=index,codec_name,codec_type,width,height,pix_fmt,avg_frame_rate,"
        "r_frame_rate,time_base,start_time,duration,nb_frames,color_range,color_space,"
        "color_transfer,color_primaries"
    )
    if include_frames:
        entries += ":frame=best_effort_timestamp_time,pkt_duration_time"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        entries,
    ]
    if include_frames:
        command.append("-show_frames")
    command.extend(["-of", "json", video])
    data = command_json(command)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in: {video}")
    return data


def ffprobe_audio(path: str | Path) -> list[dict[str, Any]]:
    data = command_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,codec_type,sample_rate,channels,start_time,duration",
            "-of",
            "json",
            str(Path(path).resolve()),
        ]
    )
    return data.get("streams", [])


def rate_to_fraction(value: str | None, fallback: str = "25/1") -> Fraction:
    try:
        rate = Fraction(value or fallback)
    except (ValueError, ZeroDivisionError):
        rate = Fraction(fallback)
    if rate <= 0:
        rate = Fraction(fallback)
    return rate


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_commit(repo: str | Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def ensure_frame_geometry(frames: Iterable[Path]) -> tuple[int, int]:
    expected: tuple[int, int] | None = None
    for frame in frames:
        current = image_size(frame)
        if expected is None:
            expected = current
        elif current != expected:
            raise RuntimeError(f"Geometry mismatch: {frame} is {current}, expected {expected}")
    if expected is None:
        raise RuntimeError("Empty frame sequence")
    return expected
