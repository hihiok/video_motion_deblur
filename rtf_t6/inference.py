"""Memory-bounded temporal and spatial inference helpers."""
from __future__ import annotations

import torch


def window_starts(length: int, window: int, overlap: int) -> list[int]:
    if length <= 0 or window <= 0:
        raise ValueError("length and window must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must satisfy 0 <= overlap < window")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window - overlap))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_weights(length: int, is_first: bool, is_last: bool, overlap: int, device) -> torch.Tensor:
    weights = torch.ones(length, device=device)
    ramp = min(overlap, length)
    if ramp > 0 and not is_first:
        weights[:ramp] *= torch.linspace(1 / (ramp + 1), ramp / (ramp + 1), ramp, device=device)
    if ramp > 0 and not is_last:
        weights[-ramp:] *= torch.linspace(ramp / (ramp + 1), 1 / (ramp + 1), ramp, device=device)
    return weights


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or tile >= length:
        return [0]
    if overlap < 0 or overlap >= tile:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile")
    starts = list(range(0, length - tile + 1, tile - overlap))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _axis_weight(length: int, at_start: bool, at_end: bool, overlap: int, device) -> torch.Tensor:
    weight = torch.ones(length, device=device)
    ramp = min(overlap, length)
    if ramp and not at_start:
        weight[:ramp] *= torch.linspace(1 / (ramp + 1), ramp / (ramp + 1), ramp, device=device)
    if ramp and not at_end:
        weight[-ramp:] *= torch.linspace(ramp / (ramp + 1), 1 / (ramp + 1), ramp, device=device)
    return weight


def infer_spatial_tiles(
    model,
    video: torch.Tensor,
    tile_size: int,
    tile_overlap: int,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("Tiled inference expects shape 1,T,C,H,W")
    _, t, _, h, w = video.shape
    ys = tile_starts(h, tile_size, tile_overlap)
    xs = tile_starts(w, tile_size, tile_overlap)
    output = torch.zeros((1, t, 3, h, w), device=video.device, dtype=torch.float32)
    weights = torch.zeros((1, 1, 1, h, w), device=video.device, dtype=torch.float32)
    for y0 in ys:
        y1 = min(y0 + tile_size, h) if tile_size > 0 else h
        for x0 in xs:
            x1 = min(x0 + tile_size, w) if tile_size > 0 else w
            tile = video[..., y0:y1, x0:x1]
            enabled = autocast_dtype is not None and video.device.type == "cuda"
            with torch.autocast(device_type=video.device.type, dtype=autocast_dtype, enabled=enabled):
                prediction = model(tile)
            wy = _axis_weight(y1 - y0, y0 == 0, y1 == h, tile_overlap, video.device)
            wx = _axis_weight(x1 - x0, x0 == 0, x1 == w, tile_overlap, video.device)
            weight = (wy[:, None] * wx[None, :]).view(1, 1, 1, y1 - y0, x1 - x0)
            output[..., y0:y1, x0:x1] += prediction.float() * weight
            weights[..., y0:y1, x0:x1] += weight
    return output / weights.clamp_min(1e-8)


def infer_long_video(
    model,
    video: torch.Tensor,
    window: int = 6,
    temporal_overlap: int = 4,
    tile_size: int = 0,
    tile_overlap: int = 32,
    autocast_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("Long-video inference expects shape 1,T,C,H,W")
    total = video.shape[1]
    starts = window_starts(total, window, temporal_overlap)
    output = torch.zeros_like(video, dtype=torch.float32)
    weights = torch.zeros((1, total, 1, 1, 1), device=video.device, dtype=torch.float32)
    for start in starts:
        end = min(start + window, total)
        clip = video[:, start:end]
        prediction = infer_spatial_tiles(
            model, clip, tile_size, tile_overlap, autocast_dtype
        )
        temporal_weight = blend_weights(
            end - start,
            is_first=start == 0,
            is_last=end == total,
            overlap=temporal_overlap,
            device=video.device,
        ).view(1, -1, 1, 1, 1)
        output[:, start:end] += prediction * temporal_weight
        weights[:, start:end] += temporal_weight
    return output / weights.clamp_min(1e-8)


def owned_temporal_range(
    starts: list[int],
    index: int,
    end: int,
    window: int,
    total: int,
) -> tuple[int, int]:
    """Return the global frame interval owned by one overlapping window."""
    start = starts[index]
    previous_end = min(starts[index - 1] + window, total) if index else 0
    left = 0 if index == 0 else (previous_end + start) // 2
    right = end if index == len(starts) - 1 else (end + starts[index + 1]) // 2
    return max(start, left), min(end, right)
