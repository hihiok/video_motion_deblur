#!/usr/bin/env python3
"""Run official RT-Focuser on a business MP4 at native resolution.

This wrapper intentionally does not reimplement RT-Focuser. It imports the
model class from a checked-out official ReaganWu/RT-Focuser repository and
loads the official GoPro checkpoint.

Outputs:
  <output_dir>/frames/%08d.png
  <output_dir>/rt_focuser_output.mp4
  <output_dir>/preview_input_output.jpg
  <output_dir>/run_summary.txt
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    # Right/bottom-only padding keeps the original image coordinates unchanged.
    mode = "reflect" if h > pad_h and w > pad_w else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (h, w)


def tensor_from_bgr(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return t.unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0)


def bgr_from_tensor(x: torch.Tensor, h: int, w: int) -> np.ndarray:
    x = x[..., :h, :w].detach().float().clamp_(0, 1)[0]
    rgb = x.mul_(255.0).round_().byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ffmpeg_encode(frames_dir: Path, input_video: Path, output_video: Path, fps: float):
    # Loss-light H.264 encode, copying original audio when present.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-framerate", f"{fps:.12g}",
        "-i", str(frames_dir / "%08d.png"),
        "-i", str(input_video),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "10",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-video", required=True)
    ap.add_argument("--rt-repo", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true", help="Use CUDA autocast FP16 for inference")
    args = ap.parse_args()

    input_video = Path(args.input_video).resolve()
    rt_repo = Path(args.rt_repo).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else (
        rt_repo / "Pretrained_Weights" / "GoPro_RT_Focuser_Standard_256.pth"
    )
    output_dir = Path(args.output_dir).resolve()
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    if not (rt_repo / "model" / "rt_focuser_model.py").is_file():
        raise FileNotFoundError(f"Official RT-Focuser repo not found at {rt_repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if subprocess.run(["which", "ffmpeg"], stdout=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("ffmpeg is required")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    sys.path.insert(0, str(rt_repo))
    from model.rt_focuser_model import RT_Focuser_Standard  # noqa: E402

    model = RT_Focuser_Standard().to(device)
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS reported by OpenCV: {fps}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    first_input = None
    first_output = None
    processed = 0
    infer_seconds = 0.0

    # autocast API differs slightly across torch versions; use a no-op on CPU.
    amp_ctx = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            raise RuntimeError(f"Frame {processed} changed resolution: {frame.shape[1]}x{frame.shape[0]}")

        x = tensor_from_bgr(frame, device)
        x, (orig_h, orig_w) = pad_to_multiple(x, 16)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode(), amp_ctx():
            y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_seconds += time.perf_counter() - t0

        out = bgr_from_tensor(y, orig_h, orig_w)
        processed += 1
        out_path = frames_dir / f"{processed:08d}.png"
        if not cv2.imwrite(str(out_path), out):
            raise RuntimeError(f"Failed writing {out_path}")

        if processed == 1:
            first_input = frame.copy()
            first_output = out.copy()
        if processed == 1 or processed % 25 == 0:
            print(f"[{processed}/{expected_frames if expected_frames > 0 else '?'}]", flush=True)

    cap.release()
    if processed == 0:
        raise RuntimeError("No frames processed")

    if first_input is not None and first_output is not None:
        preview = np.concatenate([first_input, first_output], axis=1)
        cv2.putText(preview, "Input", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(preview, "RT-Focuser", (width + 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / "preview_input_output.jpg"), preview)

    output_video = output_dir / "rt_focuser_output.mp4"
    ffmpeg_encode(frames_dir, input_video, output_video, fps)

    peak_gib = 0.0
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    summary = output_dir / "run_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("RT-FOCUSER BUSINESS VIDEO INFERENCE\n")
        f.write("STATUS: PASS\n")
        f.write(f"INPUT_VIDEO: {input_video}\n")
        f.write(f"INPUT_SHA256: {sha256_file(input_video)}\n")
        f.write(f"INPUT_RESOLUTION: {width}x{height}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"EXPECTED_FRAMES: {expected_frames}\n")
        f.write(f"PROCESSED_FRAMES: {processed}\n")
        f.write(f"CHECKPOINT: {checkpoint}\n")
        f.write(f"CHECKPOINT_SHA256: {sha256_file(checkpoint)}\n")
        f.write(f"PARAMETERS: {params}\n")
        f.write(f"DEVICE: {device}\n")
        f.write(f"FP16_AUTOCAST: {args.fp16}\n")
        f.write(f"PURE_MODEL_INFERENCE_SECONDS: {infer_seconds:.6f}\n")
        f.write(f"MODEL_MS_PER_FRAME: {1000.0 * infer_seconds / processed:.3f}\n")
        f.write(f"MODEL_FPS: {processed / infer_seconds:.4f}\n")
        f.write(f"PEAK_TORCH_ALLOCATED_GIB: {peak_gib:.3f}\n")
        f.write(f"OUTPUT_FRAMES: {frames_dir}\n")
        f.write(f"OUTPUT_VIDEO: {output_video}\n")
        f.write(f"PREVIEW: {output_dir / 'preview_input_output.jpg'}\n")

    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
