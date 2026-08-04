#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def natural_key(name: str):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def ffprobe(video: Path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height,nb_frames,duration",
        "-of", "json", str(video),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    stream = data["streams"][0]
    rate = stream.get("avg_frame_rate", "0/1")
    a, b = rate.split("/")
    fps = float(a) / float(b) if float(b) else 0.0
    return {"fps": fps, **stream}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-frames", required=True)
    p.add_argument("--source-mp4", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = p.parse_args()

    src = Path(args.source_frames).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    frames = sorted([x for x in src.iterdir() if x.is_file() and x.suffix.lower() in exts], key=lambda x: natural_key(x.name))
    if not frames:
        raise RuntimeError(f"No frames in {src}")

    with Image.open(frames[0]) as im:
        width, height = im.convert("RGB").size
    for frame in frames:
        with Image.open(frame) as im:
            if im.convert("RGB").size != (width, height):
                raise RuntimeError(f"Size mismatch: {frame}")
        dst = out / frame.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(frame, dst)
        else:
            dst.symlink_to(frame)

    video_info = ffprobe(Path(args.source_mp4).resolve())
    manifest = {
        "source_frames": str(src),
        "source_mp4": str(Path(args.source_mp4).resolve()),
        "canonical_frames": str(out),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "first_frame": frames[0].name,
        "last_frame": frames[-1].name,
        "fps": video_info.get("fps", 0.0),
        "ffprobe": video_info,
    }
    manifest_path = out.parent / "manifests" / "input.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
