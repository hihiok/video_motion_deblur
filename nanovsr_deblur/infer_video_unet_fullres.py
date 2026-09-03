import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.nanovsr_unet_fullres_deblur import NanoVSRFullResUNetDeblur


ARCHITECTURE = 'NanoVSRFullResUNetDeblur'


def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu')
    if ck.get('architecture') != ARCHITECTURE:
        raise RuntimeError(f'Unexpected architecture: {ck.get("architecture")}')
    cfg = ck.get('model_config', {})
    model = NanoVSRFullResUNetDeblur(
        base_channels=int(cfg.get('base_channels', 48)),
        mid_channels=int(cfg.get('mid_channels', 64)),
        bottleneck_channels=int(cfg.get('bottleneck_channels', 96)),
        fullres_blocks=int(cfg.get('fullres_blocks', 2)),
        mid_blocks=int(cfg.get('mid_blocks', 2)),
        bottleneck_blocks=int(cfg.get('bottleneck_blocks', 4)),
        grad_checkpoint=False,
    ).to(device).eval()
    model.load_state_dict(ck['model'], strict=True)
    return model, ck


def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f'No frames decoded from {path}')
    return frames, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--chunk', type=int, default=15)
    ap.add_argument('--overlap', type=int, default=4)
    ap.add_argument('--fp16', action='store_true')
    args = ap.parse_args()

    if args.chunk < 1:
        raise ValueError('--chunk must be >= 1')
    if args.overlap < 0 or args.overlap >= args.chunk:
        raise ValueError('--overlap must satisfy 0 <= overlap < chunk')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_model(args.checkpoint, device)
    frames, fps = load_video(args.input)
    n = len(frames)
    h, w = frames[0].shape[:2]
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (h, w):
            raise RuntimeError(f'Variable input resolution at frame {i}: {frame.shape[:2]} vs {(h,w)}')

    sums = np.zeros((n, h, w, 3), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    step = max(1, args.chunk - args.overlap)
    starts = list(range(0, n, step))
    if starts and starts[-1] + args.chunk < n:
        starts.append(max(0, n - args.chunk))

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for s in starts:
            e = min(n, s + args.chunk)
            arr = np.stack(frames[s:e]).astype(np.float32) / 255.0
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                y = model(x)[0]
            y = y.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            for j in range(e - s):
                sums[s + j] += y[j]
                counts[s + j] += 1.0
            print(f'chunk={s}:{e} size={w}x{h}', flush=True)
            if e == n:
                break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create output video: {out_path}')
    for i in range(n):
        rgb = np.clip(sums[i] / max(counts[i], 1.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    peak = 0.0
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f'ARCHITECTURE={ck.get("architecture")}', flush=True)
    print('RECURRENT_STATE=FULL_RESOLUTION', flush=True)
    print(f'CHECKPOINT_STEP={ck.get("step")}', flush=True)
    print(f'OUTPUT={out_path}', flush=True)
    print(f'FRAMES={n} FPS={fps:.3f} SIZE={w}x{h}', flush=True)
    print(f'PEAK_GPU_GIB={peak:.3f}', flush=True)


if __name__ == '__main__':
    main()
