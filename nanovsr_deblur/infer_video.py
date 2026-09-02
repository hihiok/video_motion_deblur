import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.nanovsr_deblur import NanoVSRDeblur


def load_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: raise RuntimeError(f'No frames: {path}')
    return frames, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--num-feat', type=int, default=48)
    ap.add_argument('--num-blocks', type=int, default=12)
    ap.add_argument('--chunk', type=int, default=15)
    ap.add_argument('--overlap', type=int, default=4)
    ap.add_argument('--fp16', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location='cpu')
    saved_args = ck.get('args', {}) if isinstance(ck, dict) else {}
    nf = int(saved_args.get('num_feat', args.num_feat)); nb = int(saved_args.get('num_blocks', args.num_blocks))
    model = NanoVSRDeblur(nf, nb).to(device).eval()
    model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck, strict=True)

    frames, fps = load_video(args.input)
    n = len(frames); h, w = frames[0].shape[:2]
    sums = np.zeros((n, h, w, 3), np.float32); counts = np.zeros(n, np.float32)
    step = max(1, args.chunk - args.overlap)
    starts = list(range(0, n, step))
    if starts and starts[-1] + args.chunk < n: starts.append(n - args.chunk)

    with torch.no_grad():
        for s in starts:
            e = min(n, s + args.chunk)
            arr = np.stack(frames[s:e]).astype(np.float32) / 255.0
            x = torch.from_numpy(arr).permute(0,3,1,2).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=args.fp16): y = model(x)[0]
            y = y.float().clamp(0,1).permute(0,2,3,1).cpu().numpy()
            for j in range(e-s):
                sums[s+j] += y[j]; counts[s+j] += 1
            if e == n: break

    outp = Path(args.output); outp.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    for i in range(n):
        rgb = np.clip(sums[i] / max(counts[i], 1.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f'Wrote {outp} frames={n} fps={fps:.3f} size={w}x{h}')

if __name__ == '__main__':
    main()
