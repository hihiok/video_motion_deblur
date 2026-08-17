#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from generic_restoration.benchmark_utils import git_commit, list_frames, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Record reproducibility data for an official external runner.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    weights = Path(args.weights).resolve()
    files = []
    if weights.is_file():
        files = [weights]
    elif weights.is_dir():
        files = sorted(
            path for path in weights.rglob("*") if path.is_file() and not path.is_symlink()
        )
    payload = {
        "model": args.model,
        "official_commit": git_commit(args.repo),
        "repo": str(Path(args.repo).resolve()),
        "mode": args.mode,
        "output_frame_count": len(list_frames(args.output_frames)),
        "weights": str(weights),
        "weight_file_count": len(files),
        "weight_total_bytes": sum(path.stat().st_size for path in files),
        "largest_weight_files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:10]
        ],
    }
    write_json(args.metadata, payload)
    print(args.metadata)


if __name__ == "__main__":
    main()
