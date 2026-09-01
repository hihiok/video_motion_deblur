#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from generic_restoration.benchmark_utils import list_frames


def fit(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed-index input/output review sheet.")
    parser.add_argument("--input-frames", required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=/path/to/frames")
    parser.add_argument("--output", required=True)
    parser.add_argument("--column-width", type=int, default=480)
    args = parser.parse_args()

    columns: list[tuple[str, list[Path]]] = [("INPUT", list_frames(args.input_frames))]
    for item in args.model:
        if "=" not in item:
            raise ValueError("--model must be NAME=/path/to/frames")
        name, folder = item.split("=", 1)
        columns.append((name, list_frames(folder)))
    count = len(columns[0][1])
    if any(len(frames) != count for _, frames in columns):
        raise RuntimeError("Preview sources have different frame counts")
    indices = sorted(set([0, count // 2, count - 1]))
    margin, label_h = 12, 30
    rows: list[list[Image.Image]] = []
    for index in indices:
        rows.append([fit(Image.open(frames[index]).convert("RGB"), args.column_width) for _, frames in columns])
    row_h = max(image.height for row in rows for image in row) + label_h
    canvas = Image.new(
        "RGB",
        (margin + len(columns) * (args.column_width + margin), margin + len(rows) * (row_h + margin)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, (frame_index, images) in enumerate(zip(indices, rows)):
        y = margin + row_index * (row_h + margin)
        for column_index, ((name, _), image) in enumerate(zip(columns, images)):
            x = margin + column_index * (args.column_width + margin)
            draw.text((x, y), f"{name} | frame {frame_index}", fill="black", font=font)
            canvas.paste(image, (x, y + label_h))
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=95)
    print(target)


if __name__ == "__main__":
    main()
