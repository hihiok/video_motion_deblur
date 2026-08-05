#!/usr/bin/env python3
"""Run official BSSTNet on a frame sequence without requiring ground truth.

This wrapper follows the official test path: RAFT flow at 1/4 resolution and
256x256 spatial patch inference with 64-pixel overlap.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    blend_weights,
    import_repo,
    inspect_frames,
    list_frames,
    load_rgb_float,
    save_rgb_float,
    temporal_chunks,
    unwrap_state_dict,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--raft-checkpoint", required=True)
    p.add_argument("--clip-len", type=int, default=48)
    p.add_argument("--temporal-overlap", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--patch-overlap", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def starts_for_size(size: int, patch: int, overlap: int):
    if size <= patch:
        return [0]
    stride = patch - overlap
    starts = list(range(0, size - patch + 1, stride))
    last = size - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def feather_1d(length: int, overlap: int, first: bool, last: bool):
    w = torch.ones(length, dtype=torch.float32)
    ramp = min(overlap, length // 2)
    if ramp > 0 and not first:
        w[:ramp] = torch.linspace(1.0 / (ramp + 1), 1.0, ramp)
    if ramp > 0 and not last:
        w[-ramp:] = torch.linspace(1.0, 1.0 / (ramp + 1), ramp)
    return w


def pad_spatial(x: torch.Tensor, minimum: int = 256, multiple: int = 8):
    """Pad B,T,C,H,W by applying 2D padding to flattened frames."""
    if x.ndim != 5:
        raise ValueError(f"Expected B,T,C,H,W tensor, got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    target_h = max(minimum, ((h + multiple - 1) // multiple) * multiple)
    target_w = max(minimum, ((w + multiple - 1) // multiple) * multiple)
    ph, pw = target_h - h, target_w - w
    if ph == 0 and pw == 0:
        return x, ph, pw
    mode = "reflect" if h > 1 and w > 1 and ph < h and pw < w else "replicate"
    flat = x.reshape(b * t, c, h, w)
    flat = F.pad(flat, (0, pw, 0, ph), mode=mode)
    return flat.reshape(b, t, c, target_h, target_w), ph, pw


def get_bi_flows(raft, lq):
    b, t, c, h, w = lq.shape
    with torch.no_grad():
        lq1 = lq[:, :-1].reshape(b * (t - 1), c, h, w)
        lq2 = lq[:, 1:].reshape(b * (t - 1), c, h, w)
        lq1 = F.interpolate(lq1, scale_factor=0.5, mode="bilinear", align_corners=False)
        lq2 = F.interpolate(lq2, scale_factor=0.5, mode="bilinear", align_corners=False)
        forwards = raft(lq2, lq1).detach()
        backwards = raft(lq1, lq2).detach()
        forwards = F.interpolate(forwards, scale_factor=0.5, mode="bilinear", align_corners=False)
        backwards = F.interpolate(backwards, scale_factor=0.5, mode="bilinear", align_corners=False)
        forwards = forwards.view(b, t - 1, 2, h // 4, w // 4) / 2.0
        backwards = backwards.view(b, t - 1, 2, h // 4, w // 4) / 2.0
    return forwards, backwards


def run_spatial_patches(model, lq, fw, bw, patch: int, overlap: int):
    b, t, c, h, w = lq.shape
    if patch != 256:
        raise ValueError("Official BSSTNet architecture fixes fold_feat_size=(64,64); use patch-size 256")
    hs = starts_for_size(h, patch, overlap)
    ws = starts_for_size(w, patch, overlap)
    accum = torch.zeros((b, t, c, h, w), dtype=torch.float32, device="cpu")
    weight = torch.zeros_like(accum)

    for hi, y in enumerate(hs):
        wy = feather_1d(patch, overlap, hi == 0, hi == len(hs) - 1)
        for wi, x in enumerate(ws):
            wx = feather_1d(patch, overlap, wi == 0, wi == len(ws) - 1)
            window = (wy[:, None] * wx[None, :]).view(1, 1, 1, patch, patch)
            lq_patch = lq[..., y:y + patch, x:x + patch]
            fw_patch = fw[..., y // 4:y // 4 + patch // 4, x // 4:x // 4 + patch // 4]
            bw_patch = bw[..., y // 4:y // 4 + patch // 4, x // 4:x // 4 + patch // 4]
            with torch.cuda.amp.autocast(enabled=True):
                out = model(lq_patch.half(), fw_patch, bw_patch)
            out = out.float().cpu()
            accum[..., y:y + patch, x:x + patch] += out * window
            weight[..., y:y + patch, x:x + patch] += window
            del out, lq_patch, fw_patch, bw_patch
            torch.cuda.empty_cache()
    return accum / weight.clamp_min(1e-6)


def main():
    args = parse_args()
    repo = import_repo(args.repo)
    os.chdir(repo)

    raft_target = repo / "model_zoos" / "raft-things.pth"
    raft_target.parent.mkdir(parents=True, exist_ok=True)
    supplied_raft = Path(args.raft_checkpoint).resolve()
    if not supplied_raft.is_file():
        raise FileNotFoundError(supplied_raft)
    if raft_target.resolve() != supplied_raft:
        if raft_target.exists() or raft_target.is_symlink():
            raft_target.unlink()
        raft_target.symlink_to(supplied_raft)

    from basicsr.archs.BSST_arch import BSST
    from basicsr.archs.RAFT.raft import RAFT

    frames = list_frames(args.input)
    height, width = inspect_frames(frames)
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = BSST().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = unwrap_state_dict(ckpt)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"BSST checkpoint mismatch: {loaded}")

    raft = RAFT().to(device).eval()
    for p in raft.parameters():
        p.requires_grad_(False)

    sums = [np.zeros((height, width, 3), np.float32) for _ in frames]
    tw = np.zeros(len(frames), np.float32)
    chunks = temporal_chunks(len(frames), args.clip_len, args.temporal_overlap)
    started = time.time()

    for chunk_id, (start, end) in enumerate(chunks):
        arrays = [load_rgb_float(p) for p in frames[start:end]]
        lq = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        lq, _, _ = pad_spatial(lq, args.patch_size, 8)
        fw, bw = get_bi_flows(raft, lq)
        with torch.inference_mode():
            pred = run_spatial_patches(model, lq, fw, bw, args.patch_size, args.patch_overlap)
        pred = pred[0, :, :, :height, :width].permute(0, 2, 3, 1).numpy()
        local_w = blend_weights(end - start, start, end, len(frames), args.temporal_overlap)
        for j, idx in enumerate(range(start, end)):
            sums[idx] += pred[j] * local_w[j]
            tw[idx] += local_w[j]
        print(f"BSSTNet chunk {chunk_id+1}/{len(chunks)}: [{start}, {end})")
        del lq, fw, bw, pred
        torch.cuda.empty_cache()

    for idx, src in enumerate(frames):
        save_rgb_float(sums[idx] / max(tw[idx], 1e-6), out_dir / src.name)

    write_json(out_dir.parent / "run_metadata.json", {
        "model": "BSSTNet",
        "repo": str(repo),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "raft_checkpoint": str(supplied_raft),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "clip_len": args.clip_len,
        "temporal_overlap": args.temporal_overlap,
        "patch_size": args.patch_size,
        "patch_overlap": args.patch_overlap,
        "runtime_seconds": time.time() - started,
        "torch": torch.__version__,
    })


if __name__ == "__main__":
    main()
