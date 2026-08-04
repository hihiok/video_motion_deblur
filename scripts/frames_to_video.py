#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def key(p):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--crf", type=int, default=10)
    args = p.parse_args()
    folder = Path(args.frames).resolve()
    imgs = sorted([x for x in folder.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=key)
    if not imgs:
        raise RuntimeError(f"No frames: {folder}")
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for img in imgs:
            safe = str(img).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
            f.write(f"duration {1.0 / args.fps:.12f}\n")
        f.write(f"file '{str(imgs[-1]).replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'\n")
        concat = f.name
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat,
        "-vsync", "vfr", "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", str(out)
    ], check=True)
    print(out)


if __name__ == "__main__":
    main()
