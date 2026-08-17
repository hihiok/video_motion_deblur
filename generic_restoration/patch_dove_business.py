#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix DOVE padding removal for non-4x inference.")
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    source = repo / "inference_script.py"
    text = source.read_text(encoding="utf-8")
    old = "remove_padding_and_extra_frames(video_generate, pad_f, pad_h*4, pad_w*4)"
    new = "remove_padding_and_extra_frames(video_generate, pad_f, pad_h*args.upscale, pad_w*args.upscale)"
    changed = False
    if new not in text:
        if old not in text:
            raise RuntimeError("Pinned DOVE padding context was not found")
        source.write_text(text.replace(old, new, 1), encoding="utf-8")
        changed = True
    marker = {
        "patch": "dove_upscale_padding_v1",
        "changed_this_run": changed,
        "reason": "Official source multiplied padding by 4 even when --upscale=1",
    }
    (repo / ".business_benchmark_patch.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
