import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.network_nanovnr_nafnet_rgb import NanoVNRNAFNetRGB


def load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu')
    if ck.get('architecture') != 'NanoVNRNAFNetRGB':
        raise RuntimeError(f'Unexpected architecture: {ck.get("architecture")}')
    model = NanoVNRNAFNetRGB(num_feat=12, grad_checkpoint=False).to(device).eval()
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
        raise RuntimeError(f'No frames decoded: {path}')
    return frames, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--chunk', type=int, default=15)
    ap.add_argument('--fp16', action='store_true')
    args = ap.parse_args()

    if args.chunk < 1:
        raise ValueError('--chunk must be >=1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_model(args.checkpoint, device)
    frames, fps = load_video(args.input)
    n = len(frames)
    h, w = frames[0].shape[:2]
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (h, w):
            raise RuntimeError(f'Variable frame size at {i}: {frame.shape[:2]} vs {(h,w)}')

    outputs = []
    prev_forward_feat = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for s in range(0, n, args.chunk):
            e = min(n, s + args.chunk)
            arr = np.stack(frames[s:e]).astype(np.float32) / 255.0
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                y, prev_forward_feat = model(x, prev_forward_feat=prev_forward_feat)
            # Carry forward state exactly across non-overlapping chunks.
            prev_forward_feat = prev_forward_feat.detach()
            y = y[0].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            outputs.extend(y)
            print(f'chunk={s}:{e} carry_forward_state=YES size={w}x{h}', flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create output: {out_path}')
    for rgb in outputs:
        img = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()

    peak = 0.0
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f'ARCHITECTURE={ck.get("architecture")}', flush=True)
    print(f'CHECKPOINT_STEP={ck.get("step")}', flush=True)
    print(f'OUTPUT={out_path}', flush=True)
    print(f'FRAMES={n} FPS={fps:.3f} SIZE={w}x{h}', flush=True)
    print(f'CHUNK={args.chunk} FORWARD_STATE_CARRY=YES BACKWARD_STATE_PER_CHUNK=RESET', flush=True)
    print(f'PEAK_GPU_GIB={peak:.3f}', flush=True)


if __name__ == '__main__':
    main()
