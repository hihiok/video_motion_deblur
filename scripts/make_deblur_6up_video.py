#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create a 2x3 motion-deblur comparison video.

Fixed layout (user requested order):
    Row 1: Input | DSTNet | BSSTNet
    Row 2: Shift-Net+ | Turtle | RealVDeblur

Each tile is labelled at the top-left. Streams are aligned by natural filename
order and truncated to the shortest stream. Turtle prefers Frame_*_Pred.png.

For 1280x720 inputs, the default tile size is 640x360 and the final canvas is
1920x720. Video is encoded with ffmpeg/libx264 when available, otherwise the
script falls back to OpenCV mp4v.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


STREAM_SPECS = (
    ("Input", "input_dir", False),
    ("DSTNet", "dstnet_dir", False),
    ("BSSTNet", "bsstnet_dir", False),
    ("Shift-Net+", "shiftnet_dir", False),
    ("Turtle", "turtle_dir", True),
    ("RealVDeblur", "realvdeblur_dir", False),
)


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def list_images(directory: Path, turtle: bool = False) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if turtle:
        pred = [p for p in files if re.fullmatch(r"Frame_\d+_Pred\.png", p.name, flags=re.IGNORECASE)]
        if pred:
            files = pred

    files.sort(key=lambda p: natural_key(p.name))
    if not files:
        raise RuntimeError(f"No image frames found in: {directory}")
    return files


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def letterbox(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def draw_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.62, min(1.0, min(w / 640.0, h / 360.0) * 0.82))
    thickness = 2
    margin = max(8, int(round(min(w, h) * 0.025)))
    pad_x = 10
    pad_y = 8

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1, y1 = margin, margin
    x2 = min(w - 1, x1 + text_w + 2 * pad_x)
    y2 = min(h - 1, y1 + text_h + baseline + 2 * pad_y)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.52, out, 0.48, 0, dst=out)

    origin = (x1 + pad_x, y1 + pad_y + text_h)
    cv2.putText(out, text, origin, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def compose_frame(images: Sequence[np.ndarray], labels: Sequence[str], tile_w: int, tile_h: int) -> np.ndarray:
    if len(images) != 6 or len(labels) != 6:
        raise ValueError("Exactly six images and six labels are required")

    tiles = [draw_label(letterbox(img, tile_w, tile_h), label) for img, label in zip(images, labels)]
    row1 = np.concatenate(tiles[:3], axis=1)
    row2 = np.concatenate(tiles[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


class VideoSink:
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FfmpegSink(VideoSink):
    def __init__(self, output: Path, width: int, height: int, fps: float, crf: int):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self.proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin")

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with status {rc}")


class OpenCvSink(VideoSink):
    def __init__(self, output: Path, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open OpenCV VideoWriter: {output}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def build_sink(output: Path, width: int, height: int, fps: float, crf: int) -> Tuple[VideoSink, str]:
    if shutil.which("ffmpeg"):
        return FfmpegSink(output, width, height, fps, crf), "ffmpeg/libx264"
    return OpenCvSink(output, width, height, fps), "OpenCV/mp4v fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--dstnet-dir", required=True, type=Path)
    parser.add_argument("--bsstnet-dir", required=True, type=Path)
    parser.add_argument("--shiftnet-dir", required=True, type=Path)
    parser.add_argument("--turtle-dir", required=True, type=Path)
    parser.add_argument("--realvdeblur-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--tile-width", type=int, default=0)
    parser.add_argument("--tile-height", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF; lower is higher quality")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    streams: List[Tuple[str, List[Path]]] = []
    for label, arg_name, turtle in STREAM_SPECS:
        directory = getattr(args, arg_name)
        files = list_images(directory, turtle=turtle)
        streams.append((label, files))

    counts = {label: len(files) for label, files in streams}
    frame_count = min(counts.values())
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if frame_count <= 0:
        raise RuntimeError("No frames available after alignment")

    first = read_image(streams[0][1][0])
    input_h, input_w = first.shape[:2]
    tile_w = args.tile_width or max(1, input_w // 2)
    tile_h = args.tile_height or max(1, input_h // 2)
    output_w = tile_w * 3
    output_h = tile_h * 2

    # yuv420p requires even dimensions.
    if output_w % 2 or output_h % 2:
        raise ValueError(f"Output dimensions must be even for yuv420p: {output_w}x{output_h}")

    video_path = args.output_dir / "comparison_6grid.mp4"
    preview_path = args.output_dir / "preview_first_frame.png"
    summary_path = args.output_dir / "run_summary.txt"

    sink, encoder = build_sink(video_path, output_w, output_h, args.fps, args.crf)
    labels = [label for label, _ in streams]

    try:
        for idx in range(frame_count):
            images = [read_image(files[idx]) for _, files in streams]
            canvas = compose_frame(images, labels, tile_w, tile_h)
            if idx == 0:
                if not cv2.imwrite(str(preview_path), canvas):
                    raise RuntimeError(f"Failed to save preview: {preview_path}")
            sink.write(canvas)
            if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == frame_count:
                print(f"[{idx + 1}/{frame_count}]", flush=True)
    finally:
        sink.close()

    lines = [
        "MAKE_DEBLUR_6GRID_VIDEO_20260817",
        "STATUS: PASS",
        "",
        "Layout:",
        "Row1: Input | DSTNet | BSSTNet",
        "Row2: Shift-Net+ | Turtle | RealVDeblur",
        "",
        "Frame counts:",
    ]
    lines.extend(f"{label}: {count}" for label, count in counts.items())
    lines.extend(
        [
            "",
            f"Used frames: {frame_count}",
            f"Input resolution: {input_w}x{input_h}",
            f"Tile resolution: {tile_w}x{tile_h}",
            f"Output resolution: {output_w}x{output_h}",
            f"FPS: {args.fps}",
            f"Encoder: {encoder}",
            f"Video: {video_path}",
            f"Preview: {preview_path}",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"STATUS: FAIL\nERROR: {exc}", file=sys.stderr)
        raise
