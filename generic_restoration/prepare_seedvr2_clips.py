#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from generic_restoration.benchmark_utils import list_frames, read_json, run_checked, write_json
from generic_restoration.planning import plan_seed_chunks


def encode_chunk(frames: list[Path], target: Path, fps: str) -> None:
    with tempfile.TemporaryDirectory(prefix="seedvr2_clip_") as temporary:
        temp = Path(temporary)
        for index, source in enumerate(frames):
            destination = temp / f"{index:08d}.png"
            try:
                destination.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, destination)
        run_checked(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-framerate",
                fps,
                "-start_number",
                "0",
                "-i",
                str(temp / "%08d.png"),
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-g",
                "1",
                "-pix_fmt",
                "bgr0",
                str(target),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make bounded lossless clips for SeedVR2.")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-manifest", required=True)
    parser.add_argument("--core-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=4)
    args = parser.parse_args()

    frames = list_frames(args.frames)
    manifest = read_json(args.manifest)
    if len(frames) != int(manifest["frame_count"]):
        raise RuntimeError("Canonical frame count does not match its manifest")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("chunk_*.mkv"):
        stale.unlink()
    chunks = plan_seed_chunks(len(frames), args.core_frames, args.context_frames)
    for chunk in chunks:
        target = output / f"{chunk['name']}.mkv"
        encode_chunk(
            frames[int(chunk["clip_start"]) : int(chunk["clip_end"])],
            target,
            str(manifest["avg_frame_rate"]),
        )
        chunk["input_video"] = str(target)
    payload = {
        "source_frame_count": len(frames),
        "core_frames": args.core_frames,
        "context_frames": args.context_frames,
        "chunks": chunks,
    }
    write_json(args.chunk_manifest, payload)
    print(args.chunk_manifest)


if __name__ == "__main__":
    main()
