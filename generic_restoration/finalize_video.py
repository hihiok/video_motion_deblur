#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from generic_restoration.benchmark_utils import (
    command_json,
    ensure_frame_geometry,
    list_frames,
    read_json,
    run_checked,
    write_json,
)


def write_concat(frames: list[Path], durations: list[float], target: Path) -> None:
    if len(frames) != len(durations):
        raise RuntimeError(f"Frame/duration mismatch: {len(frames)} vs {len(durations)}")
    lines = ["ffconcat version 1.0"]
    for frame, duration in zip(frames, durations):
        escaped = str(frame.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {max(duration, 1e-6):.12f}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore source timing/audio and encode a checked model output.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--crf", type=int, default=10)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    frames = list_frames(args.frames)
    expected_count = int(manifest["frame_count"])
    if len(frames) != expected_count:
        raise RuntimeError(f"Output frame count is {len(frames)}, expected {expected_count}")
    width, height = ensure_frame_geometry(frames)
    expected_size = (int(manifest["width"]), int(manifest["height"]))
    if (width, height) != expected_size:
        raise RuntimeError(f"Output size is {(width, height)}, expected {expected_size}")

    sample_indices = sorted(set([0, len(frames) // 2, len(frames) - 1]))
    input_frames = list_frames(manifest["paths"]["frames_smoke" if manifest.get("is_smoke_subset") else "frames_full"])
    sample_stats = []
    for index in sample_indices:
        output_array = np.asarray(Image.open(frames[index]).convert("RGB"), dtype=np.float32)
        input_array = np.asarray(Image.open(input_frames[index]).convert("RGB"), dtype=np.float32)
        sample_stats.append(
            {
                "index": index,
                "mean": float(output_array.mean()),
                "std": float(output_array.std()),
                "black_fraction": float((output_array <= 1).all(axis=2).mean()),
                "white_fraction": float((output_array >= 254).all(axis=2).mean()),
                "mean_abs_diff_from_input": float(np.abs(output_array - input_array).mean()),
            }
        )
    if all(item["std"] < 0.5 for item in sample_stats):
        raise RuntimeError("Output is effectively constant on all sampled frames")
    if all(item["black_fraction"] > 0.99 for item in sample_stats):
        raise RuntimeError("Output is effectively black on all sampled frames")

    durations = [float(value) for value in manifest.get("frame_durations", [])]
    if len(durations) != expected_count:
        durations = [1.0 / float(manifest["fps"])] * expected_count

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="restoration_mux_") as tmp:
        concat_file = Path(tmp) / "frames.ffconcat"
        write_concat(frames, durations, concat_file)
        common = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(manifest["source_video"]),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        copy_audio = common + ["-c:a", "copy", "-shortest", str(output)]
        print("+", " ".join(copy_audio), flush=True)
        result = subprocess.run(copy_audio)
        if result.returncode != 0:
            transcode_audio = common + ["-c:a", "aac", "-b:a", "192k", "-shortest", str(output)]
            run_checked(transcode_audio)

    encoded = command_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "json",
            str(output),
        ]
    )["streams"][0]
    encoded_count = int(encoded.get("nb_read_frames") or 0)
    if encoded_count != expected_count:
        raise RuntimeError(f"Encoded MP4 has {encoded_count} frames, expected {expected_count}")
    if (int(encoded["width"]), int(encoded["height"])) != expected_size:
        raise RuntimeError(f"Encoded MP4 geometry is wrong: {encoded}")

    report = {
        "status": "PASS",
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "source_sha256": manifest["source_sha256"],
        "fps_nominal": manifest["fps"],
        "median_frame_duration": statistics.median(durations),
        "audio_stream_count": len(manifest.get("audio_streams", [])),
        "output_video": str(output),
        "encoded_frame_count": encoded_count,
        "sample_physical_checks": sample_stats,
    }
    write_json(args.report, report)
    print(args.report)


if __name__ == "__main__":
    main()
