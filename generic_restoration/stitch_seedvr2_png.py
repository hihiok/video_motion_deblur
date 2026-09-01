#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from generic_restoration.benchmark_utils import list_frames, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep clip cores and reconstruct one SeedVR2 PNG sequence.")
    parser.add_argument("--chunk-manifest", required=True)
    parser.add_argument("--seed-output", required=True)
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    manifest = read_json(args.chunk_manifest)
    seed_output = Path(args.seed_output).resolve()
    output = Path(args.output_frames).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.png"):
        stale.unlink()

    written = 0
    chunk_reports = []
    for chunk in manifest["chunks"]:
        folder = seed_output / str(chunk["name"])
        frames = list_frames(folder)
        keep_start = int(chunk["keep_start"])
        keep_end = int(chunk["keep_end"])
        if len(frames) < keep_end:
            raise RuntimeError(
                f"{chunk['name']} produced {len(frames)} frames, but keep_end is {keep_end}"
            )
        kept = frames[keep_start:keep_end]
        for frame in kept:
            shutil.copy2(frame, output / f"{written:08d}.png")
            written += 1
        chunk_reports.append(
            {
                "name": chunk["name"],
                "produced": len(frames),
                "kept": len(kept),
            }
        )

    expected = int(manifest["source_frame_count"])
    if written != expected:
        raise RuntimeError(f"Stitched {written} frames, expected {expected}")
    write_json(
        args.report,
        {"status": "PASS", "frame_count": written, "chunks": chunk_reports},
    )
    print(args.report)


if __name__ == "__main__":
    main()
