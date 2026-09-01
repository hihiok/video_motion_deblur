from __future__ import annotations

import math


def smallest_8n1_geq(value: int) -> int:
    if value <= 1:
        return 1
    return math.ceil((value - 1) / 8) * 8 + 1


def padded_target_size(
    width: int, height: int, scale: float, multiple: int = 128
) -> tuple[int, int, int, int]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    target_w = max(1, round(width * scale))
    target_h = max(1, round(height * scale))
    padded_w = math.ceil(target_w / multiple) * multiple
    padded_h = math.ceil(target_h / multiple) * multiple
    return target_w, target_h, padded_w, padded_h


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or tile >= length:
        return [0]
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    stride = tile - overlap
    starts = list(range(0, max(length - tile + 1, 1), stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def plan_seed_chunks(
    frame_count: int, core_frames: int, context_frames: int
) -> list[dict[str, int | str]]:
    if frame_count <= 0 or core_frames <= 0 or context_frames < 0:
        raise ValueError("Invalid chunk parameters")
    chunks = []
    for index, core_start in enumerate(range(0, frame_count, core_frames)):
        core_end = min(core_start + core_frames, frame_count)
        clip_start = max(0, core_start - context_frames)
        clip_end = min(frame_count, core_end + context_frames)
        chunks.append(
            {
                "name": f"chunk_{index:04d}",
                "core_start": core_start,
                "core_end": core_end,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "keep_start": core_start - clip_start,
                "keep_end": core_end - clip_start,
            }
        )
    return chunks
