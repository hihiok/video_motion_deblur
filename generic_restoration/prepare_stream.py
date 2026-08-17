#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from generic_restoration.benchmark_utils import (
    ensure_frame_geometry,
    ffprobe_audio,
    ffprobe_video,
    list_frames,
    rate_to_fraction,
    run_checked,
    sha256_file,
    sha256_frames,
    write_json,
)


def extract_frames(source: Path, output: Path) -> list[Path]:
    if output.exists():
        existing = list_frames(output) if any(output.iterdir()) else []
        if existing:
            return existing
    output.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(output / "%08d.png"),
        ]
    )
    return list_frames(output)


def make_subset(source_frames: list[Path], output: Path, count: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.iterdir():
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    selected = source_frames[: min(count, len(source_frames))]
    for index, frame in enumerate(selected):
        target = output / f"{index:08d}.png"
        try:
            target.symlink_to(frame.resolve())
        except OSError:
            shutil.copy2(frame, target)
    return list_frames(output)


def encode_lossless(frames: Path, output: Path, rate: str) -> None:
    if output.is_file() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-framerate",
            rate,
            "-start_number",
            "0",
            "-i",
            str(frames / "%08d.png"),
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-pix_fmt",
            "bgr0",
            str(output),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode one canonical business stream for every model.")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--smoke-frames", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.input_video).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    work = Path(args.work_dir).resolve()
    manifest_path = work / "manifest.json"
    source_sha = sha256_file(source)

    if manifest_path.exists() and not args.force:
        import json

        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("source_sha256") != source_sha:
            raise RuntimeError(
                "Existing canonical data belongs to a different MP4. Use a new work directory or --force."
            )

    full_dir = work / "frames_full"
    smoke_dir = work / "frames_smoke"
    if args.force and work.exists():
        for path in (full_dir, smoke_dir):
            if path.exists():
                shutil.rmtree(path)
        for path in (work / "business_full_lossless.mkv", work / "business_smoke_lossless.mkv"):
            if path.exists():
                path.unlink()

    probe = ffprobe_video(source, include_frames=True)
    stream = probe["streams"][0]
    rate = rate_to_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    rate_text = f"{rate.numerator}/{rate.denominator}"

    frames = extract_frames(source, full_dir)
    width, height = ensure_frame_geometry(frames)
    smoke = make_subset(frames, smoke_dir, args.smoke_frames)
    encode_lossless(full_dir, work / "business_full_lossless.mkv", rate_text)
    encode_lossless(smoke_dir, work / "business_smoke_lossless.mkv", rate_text)

    timestamps = []
    durations = []
    for frame in probe.get("frames", []):
        try:
            timestamps.append(float(frame["best_effort_timestamp_time"]))
        except (KeyError, TypeError, ValueError):
            pass
        try:
            durations.append(float(frame["pkt_duration_time"]))
        except (KeyError, TypeError, ValueError):
            pass

    if len(timestamps) != len(frames):
        timestamps = [index / float(rate) for index in range(len(frames))]
    frame_durations = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    fallback_duration = (sum(frame_durations) / len(frame_durations)) if frame_durations else 1.0 / float(rate)
    frame_durations.append(durations[-1] if durations else fallback_duration)

    manifest = {
        "schema_version": 1,
        "source_video": str(source),
        "source_sha256": source_sha,
        "canonical_frames_sha256": sha256_frames(frames),
        "frame_count": len(frames),
        "smoke_frame_count": len(smoke),
        "width": width,
        "height": height,
        "avg_frame_rate": rate_text,
        "fps": float(rate),
        "timestamps": timestamps,
        "frame_durations": frame_durations,
        "video_stream": stream,
        "audio_streams": ffprobe_audio(source),
        "paths": {
            "frames_full": str(full_dir),
            "frames_smoke": str(smoke_dir),
            "video_full_lossless": str(work / "business_full_lossless.mkv"),
            "video_smoke_lossless": str(work / "business_smoke_lossless.mkv"),
        },
        "ffmpeg_proxy_environment_present": any(
            os.environ.get(name) for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
        ),
    }
    write_json(manifest_path, manifest)
    smoke_manifest = dict(manifest)
    smoke_manifest["frame_count"] = len(smoke)
    smoke_manifest["timestamps"] = timestamps[: len(smoke)]
    smoke_manifest["frame_durations"] = frame_durations[: len(smoke)]
    smoke_manifest["canonical_frames_sha256"] = sha256_frames(smoke)
    smoke_manifest["is_smoke_subset"] = True
    write_json(work / "manifest_smoke.json", smoke_manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
