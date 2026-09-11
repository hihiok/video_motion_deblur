#!/usr/bin/env bash
# NanoVNRNAFNetRGB: first TWO COMPLETE test videos from GoPro, BSD and DVD.
# No training, architecture changes, crop, resize, spatial tiling, or dataset writes.
# Defaults: checkpoint 125000; T=15 non-overlap; forward-state carry within a video.
# Each video starts with hidden=None. All original frames are evaluated exactly once.
# MP4s are previews at FPS=25 (not a claim about dataset capture frame rate).
# Usage: bash run_nanovnr_test_first2.sh
# Overrides: GPU=0 CHUNK=15 PRECISION=fp16 FPS=25 CHECKPOINT=/absolute/file.pth
#            ROOT=... GOPRO_ROOT=... BSD_ROOT=... DVD_ROOT=... OUT=...
#            PYTHON_BIN=/path/to/python MODEL_FILE=/path/to/verified/model.py
# Proxy/TLS: uses existing shell/Conda proxy variables or PROXY_ENV_SH, never logs them.
# Company TLS inspection: curl --insecure is limited to a pinned source download;
# its exact Git blob hash is checked before import. No global Git config is changed.
set +x
set -Eeuo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,14p' "$0"; exit 0
fi
[[ $# -eq 0 ]] || { echo 'No positional arguments. Run with --help for environment overrides.' >&2; exit 2; }
export ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
export GOPRO_ROOT="${GOPRO_ROOT:-$ROOT/datasets/GoPro}"
export DVD_ROOT="${DVD_ROOT:-$ROOT/datasets/DVD}"
export BSD_ROOT="${BSD_ROOT:-/mnt/ssd1/z00919662/datasets/BSD}"
export CHECKPOINT="${CHECKPOINT:-$ROOT/runs/nanovnr_nafnet_rgb_fullframe_bsd_train_test_20260904/train/step_0125000.pth}"
export OUT="${OUT:-$ROOT/runs/nanovnr_test_first2_$(date +%Y%m%d_%H%M%S)_$$}"
export CHUNK="${CHUNK:-15}" PRECISION="${PRECISION:-fp16}" FPS="${FPS:-25}"
export DEVICE="${DEVICE:-cuda}"  # cpu is for tiny local smoke tests only.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
CONDA_ENV="${CONDA_ENV:-deblur_runtime}"
# Activate the existing environment. Never install/upgrade packages automatically.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    CONDA_SH=""
    for base in "${CONDA_EXE:-}"; do
      if [[ -n "$base" ]]; then
        p="$(dirname "$(dirname "$base")")/etc/profile.d/conda.sh"
        [[ ! -f "$p" ]] || CONDA_SH="$p"
      fi
    done
    if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
      base="$(conda info --base)"; CONDA_SH="$base/etc/profile.d/conda.sh"
    fi
    if [[ -z "$CONDA_SH" ]]; then
      for base in /mnt/ssd1/z00919662/anaconda3 "$HOME/anaconda3" "$HOME/miniconda3"; do
        if [[ -f "$base/etc/profile.d/conda.sh" ]]; then CONDA_SH="$base/etc/profile.d/conda.sh"; break; fi
      done
    fi
    [[ -f "$CONDA_SH" ]] || { echo "Activate $CONDA_ENV first, or set PYTHON_BIN to its python." >&2; exit 2; }
    set +u
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    set -u
  fi
  PYTHON_BIN="$(command -v python)"
fi
export PYTHON_BIN
# Do not source proxy.md (it is documentation, not necessarily executable shell).
proxy_file="${PROXY_ENV_SH:-${CONDA_PREFIX:-/nonexistent}/etc/conda/activate.d/proxy_env.sh}"
if [[ -f "$proxy_file" ]]; then
  set +u
  source "$proxy_file" >/dev/null 2>&1
  set +x
  set -u
fi
[[ -f "$CHECKPOINT" ]] || { echo "CHECKPOINT_MISSING: $CHECKPOINT. Set CHECKPOINT to the real 125k file." >&2; exit 2; }
for cmd in ffmpeg ffprobe; do command -v "$cmd" >/dev/null || { echo "Missing $cmd in PATH" >&2; exit 2; }; done
"$PYTHON_BIN" -c 'import torch, numpy, PIL' || { echo 'Use the existing deblur_runtime Python (torch/numpy/Pillow required).' >&2; exit 2; }
# Refuse to overwrite a previous evaluation directory.
if [[ -e "$OUT" ]]; then echo "OUT_ALREADY_EXISTS: $OUT. Choose a new OUT." >&2; exit 2; fi
mkdir -p "$OUT/_code"
export MODEL_REF=a75524d9ab7188ac21b6f570a9ed1fc96f488d41
export MODEL_BLOB=de3b8032940bd96413fc685ce15389de3e899a90
export MODEL_DEST="$OUT/_code/network_nanovnr_nafnet_rgb.py"
# Reuse the exact existing model when possible; do not checkout/reset the training repo.
LOCAL_MODEL="${MODEL_FILE:-${REPO:-$ROOT/video_motion_deblur_nanovnr_nafnet_rgb}/nanovsr_deblur/models/network_nanovnr_nafnet_rgb.py}"
verify_model() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib, os, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
b = p.read_bytes()
h = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
raise SystemExit(0 if h == os.environ['MODEL_BLOB'] else 1)
PY
}
if verify_model "$LOCAL_MODEL"; then
  cp "$LOCAL_MODEL" "$MODEL_DEST"
else
  if [[ -n "${MODEL_FILE:-}" && -f "$MODEL_FILE" ]]; then
    echo 'MODEL_FILE differs from the reviewed model. Stop and synchronize the source; no local edits are overwritten.' >&2; exit 2
  fi
  command -v curl >/dev/null || { echo 'Model not found locally; curl is required for the pinned GitHub source.' >&2; exit 2; }
  curl --insecure --fail --location --silent --show-error --retry 3 --connect-timeout 30 --max-time 180 \
    "https://raw.githubusercontent.com/hihiok/video_motion_deblur/$MODEL_REF/nanovsr_deblur/models/network_nanovnr_nafnet_rgb.py" \
    -o "$MODEL_DEST"
fi
verify_model "$MODEL_DEST" || { echo 'MODEL_HASH_MISMATCH: refusing to import unverified code.' >&2; exit 2; }
# GPU=0 overrides an inherited CUDA_VISIBLE_DEVICES. Otherwise preserve it or select one GPU.
if [[ "$DEVICE" == cuda ]]; then
  if [[ -n "${GPU:-}" && "$GPU" != auto ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
  elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_line="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | head -n1)"
    export CUDA_VISIBLE_DEVICES="${gpu_line%%,*}"
  fi
fi
cp "$0" "$OUT/_code/run_nanovnr_test_first2.sh"
"$PYTHON_BIN" - <<'PY' 2>&1 | tee "$OUT/run.log"
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

OUT = Path(os.environ['OUT']).resolve()
EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
BLURS = ('blur', 'blurry', 'input', 'blur_gamma')
GTS = ('gt', 'sharp', 'target', 'label')
CHUNK = int(os.environ['CHUNK'])
FPS = float(os.environ['FPS'])
PRECISION = os.environ['PRECISION']


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def natural(s):
    return tuple((1, int(x)) if x.isdigit() else (0, x.casefold()) for x in re.split(r'(\d+)', str(s)))


def contained(path, allowed):
    p = Path(path).resolve()
    try:
        p.relative_to(allowed)
    except ValueError as exc:
        raise RuntimeError(f'TEST_PATH_POLICY_VIOLATION: {path} -> {p}, allowed={allowed}') from exc
    return p


def images(path, allowed):
    return sorted([contained(p, allowed) for p in path.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS], key=lambda p: (natural(p.name), p.name))


def child_dirs(path, allowed):
    result = {}
    for p in sorted(path.iterdir(), key=lambda p: (natural(p.name), p.name)):
        if p.is_dir() and not p.name.startswith('.'):
            key = p.name.lower()
            if key in result:
                raise RuntimeError(f'Ambiguous directory casing below {path}: {p.name}')
            contained(p, allowed)
            result[key] = p
    return result


def discover(root, family):
    # Deliberately no generic root/config fallback. BSD ONLY uses BSD/test.
    split = Path(root).expanduser() / 'test'
    if not split.is_dir(): raise RuntimeError(f'Missing exact test split: {split}')
    allowed = split.resolve()
    found = []

    def expand(b, g, seq_id, depth=0):
        bf = images(b, allowed)
        gf = images(g, allowed)
        if bf or gf:
            found.append((seq_id or '__root__', b, g))
            return
        if depth >= 4: return
        bd, gd = child_dirs(b, allowed), child_dirs(g, allowed)
        if 'rgb' in bd:
            if 'rgb' not in gd: raise RuntimeError(f'Missing GT/RGB counterpart: {g}')
            expand(bd['rgb'], gd['rgb'], seq_id, depth + 1)
            return
        for key, sub in bd.items():
            if key in ('raw', 'annotations', 'train', 'val'): continue
            if key not in gd: raise RuntimeError(f'Missing matching GT sequence for {sub}')
            expand(sub, gd[key], '/'.join(filter(None, (seq_id, sub.name))), depth + 1)

    def scan(base, depth=0):
        dirs = child_dirs(base, allowed)
        b = next((dirs[n] for n in BLURS if n in dirs), None)
        g = next((dirs[n] for n in GTS if n in dirs), None)
        if b is not None and g is not None:
            prefix = base.relative_to(split).as_posix()
            expand(b, g, '' if prefix == '.' else prefix)
            return
        if depth >= 4: return
        for name, sub in dirs.items():
            if name in set(BLURS + GTS) | {'raw', 'train', 'training', 'val', 'validation', 'annotations'}: continue
            scan(sub, depth + 1)

    scan(split)
    found.sort(key=lambda row: (natural(row[0]), row[0]))
    if len(found) < 2:
        raise RuntimeError(f'{family}: need two complete video directories below {split}; found {len(found)}. No train/config fallback.')
    results = []
    seen = set()
    for seq_id, b, g in found[:2]:
        if seq_id in seen: raise RuntimeError(f'Duplicate sequence ID: {family}/{seq_id}')
        seen.add(seq_id)
        bf, gf = images(b, allowed), images(g, allowed)
        gm = {p.name: p for p in gf}
        if not bf or len(bf) != len(gf) or set(p.name for p in bf) != set(gm):
            raise RuntimeError(f'Exact frame-name/count mismatch: {b} ({len(bf)}) vs {g} ({len(gf)}). No index pairing.')
        pairs = [(p, gm[p.name]) for p in bf]
        results.append({'dataset': family, 'sequence': seq_id, 'test_root': str(allowed), 'pairs': pairs})
    return results


def load_rgb(path):
    with Image.open(path) as im:
        if im.mode not in ('RGB', 'RGBA', 'L', 'P'):
            raise RuntimeError(f'Unexpected image mode {im.mode}: {path}. Refusing implicit bit-depth conversion.')
        return np.array(im.convert('RGB'), dtype=np.uint8, copy=True)


class Movie:
    def __init__(self, path, width, height):
        self.path = Path(path)
        self.log = self.path.with_suffix('.ffmpeg.log').open('wb')
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-threads', '1',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
               '-framerate', str(FPS), '-i', 'pipe:0', '-an',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-filter_threads', '1',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-threads', '1', '-movflags', '+faststart', str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        self.count = 0

    def write(self, rgb):
        try:
            self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f'FFmpeg failed; see {self.path.with_suffix(".ffmpeg.log")}') from exc
        self.count += 1

    def close(self, success=True):
        broken = False
        try:
            if not self.proc.stdin.closed:
                try: self.proc.stdin.close()
                except BrokenPipeError: broken = True
            if not success and self.proc.poll() is None:
                self.proc.terminate()
            try: code = self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait()
                raise RuntimeError(f'FFmpeg finalization timed out: {self.path}')
        finally:
            self.log.close()
        if success and (code or broken): raise RuntimeError(f'FFmpeg exit={code}: {self.path}')
        if success:
            # Count actual encoded frames rather than trusting container metadata.
            info = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'json', str(self.path)
            ]))['streams'][0]
            if int(info['nb_read_frames']) != self.count:
                raise RuntimeError(f'Encoded frame count mismatch: {self.path}: {info}')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def test_video(item, model, device):
    family, seq = item['dataset'], item['sequence']
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', seq) + '_' + hashlib.sha1(seq.encode()).hexdigest()[:6]
    dest = OUT / family / safe
    dest.mkdir(parents=True, exist_ok=False)
    pairs = item['pairs']
    first = load_rgb(pairs[0][0]); h, w = first.shape[:2]
    header = Image.new('RGB', (w * 3, 32), (0, 0, 0))
    try: font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError: font = ImageFont.load_default()
    draw = ImageDraw.Draw(header)
    for k, label in enumerate(('Input', 'NanoVNR NAFNet RGB', 'GT')):
        draw.text((k * w + 8, 4), label, fill=(255, 255, 255), font=font)
    header_array = np.array(header)
    writers = []
    success = False
    prev = None  # never carry state into a different video/dataset
    numbers = []
    inference_seconds = 0.0
    previews = {0, len(pairs) // 2, len(pairs) - 1}
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    print(f'BEGIN {family}/{seq}: frames={len(pairs)} native={w}x{h} chunk={CHUNK}', flush=True)
    try:
        writers.append(Movie(dest / 'deblur.mp4', w, h))
        writers.append(Movie(dest / 'input_output_gt.mp4', w * 3, h + 32))
        with (dest / 'per_frame.csv').open('w', newline='') as f:
            cw = csv.writer(f)
            cw.writerow(['frame_index_0based', 'filename', 'input_psnr_rgb_db', 'output_psnr_rgb_db', 'gain_db'])
            with torch.inference_mode():
                for start in range(0, len(pairs), CHUNK):
                    end = min(len(pairs), start + CHUNK)
                    inputs, targets = [], []
                    for bp, gp in pairs[start:end]:
                        bi, gi = load_rgb(bp), load_rgb(gp)
                        if bi.shape != gi.shape or bi.shape != (h, w, 3):
                            raise RuntimeError(f'Native shape mismatch: {bp} {bi.shape}; {gp} {gi.shape}; expected {(h,w,3)}')
                        inputs.append(bi); targets.append(gi)
                    inp = np.stack(inputs)
                    gt = np.stack(targets)
                    x = torch.from_numpy(inp).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
                    y = torch.from_numpy(gt).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=device.type == 'cuda' and PRECISION == 'fp16'):
                        pred, state = model(x, prev_forward_feat=prev)
                    prev = state.detach()
                    if device.type == 'cuda': torch.cuda.synchronize(device)
                    inference_seconds += time.perf_counter() - t0
                    if pred.shape != x.shape or prev.shape != (1, 12, h, w):
                        raise RuntimeError('Model output/hidden shape mismatch')
                    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(prev).all()):
                        raise RuntimeError('NaN/Inf prediction or hidden; no silent precision fallback')
                    p = pred[0].float().clamp(0, 1)
                    # Native RGB metrics BEFORE PNG quantization / lossy MP4 encoding.
                    inp_mse = (x[0] - y[0]).square().mean(dim=(-3, -2, -1))
                    out_mse = (p - y[0]).square().mean(dim=(-3, -2, -1))
                    # Same 1e-12 floor as the earlier PSNR script (120 dB ceiling).
                    inp_psnr = (-10 * inp_mse.clamp_min(1e-12).log10()).cpu().tolist()
                    out_psnr = (-10 * out_mse.clamp_min(1e-12).log10()).cpu().tolist()
                    rgb = (p.permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
                    for j, frame in enumerate(rgb):
                        idx = start + j
                        row = [idx, pairs[idx][0].name, inp_psnr[j], out_psnr[j], out_psnr[j] - inp_psnr[j]]
                        cw.writerow(row); numbers.append((inp_psnr[j], out_psnr[j]))
                        writers[0].write(frame)
                        trip = np.concatenate((header_array, np.concatenate((inp[j], frame, gt[j]), axis=1)), axis=0)
                        writers[1].write(trip)
                        if idx in previews:
                            Image.fromarray(trip).save(dest / f'preview_{idx:06d}.png')
                    f.flush()
                    del x, y, pred, state, p, rgb, inp, gt, inputs, targets
                    print(f'  {family}/{seq} {end}/{len(pairs)} output_PSNR={np.mean([v[1] for v in numbers]):.4f}', flush=True)
        success = True
    finally:
        close_errors = []
        for writer in writers:
            try: writer.close(success=success)
            except Exception as exc: close_errors.append(str(exc))
        prev = None
        if close_errors and success: raise RuntimeError('; '.join(close_errors))
    if len(numbers) != len(pairs): raise RuntimeError('Not all original frames were evaluated')
    a = np.array(numbers, dtype=np.float64)
    result = {
        'dataset': family, 'sequence': seq, 'frames': len(pairs), 'width': w, 'height': h,
        'input_psnr_rgb_db': float(a[:, 0].mean()), 'output_psnr_rgb_db': float(a[:, 1].mean()),
        'gain_db': float((a[:, 1] - a[:, 0]).mean()), 'chunk': CHUNK, 'precision': PRECISION,
        'forward_state_carry': True, 'reset_between_videos': True,
        'inference_seconds': inference_seconds,
        'model_inference_fps': len(pairs) / max(inference_seconds, 1e-9),
        'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.0,
        'preview_fps_assumed': FPS, 'metric': 'mean_per_frame_RGB_PSNR_native_no_border_crop_max120dB',
        'deblur_mp4': str(dest / 'deblur.mp4'), 'comparison_mp4': str(dest / 'input_output_gt.mp4'),
    }
    dump(dest / 'metrics.json', result)
    print(f"DONE {family}/{seq}: input={result['input_psnr_rgb_db']:.4f}, output={result['output_psnr_rgb_db']:.4f}, gain={result['gain_db']:+.4f} dB", flush=True)
    return result


def main():
    if CHUNK < 1 or not math.isfinite(FPS) or FPS <= 0 or PRECISION not in ('fp16', 'fp32'):
        raise RuntimeError('Require CHUNK>=1, positive finite FPS, PRECISION=fp16 or fp32')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(os.environ['DEVICE'])
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this Python environment')
    if device.type not in ('cpu', 'cuda'): raise RuntimeError('Only CUDA inference and explicit CPU smoke mode are supported')
    torch.backends.cudnn.benchmark = False
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    encoders = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.STDOUT).decode()
    if 'libx264' not in encoders: raise RuntimeError('FFmpeg has no libx264 encoder; use an existing ffmpeg build with libx264')
    selection = []
    for family, env in [('GoPro', 'GOPRO_ROOT'), ('BSD', 'BSD_ROOT'), ('DVD', 'DVD_ROOT')]:
        selection.extend(discover(os.environ[env], family))
    dump(OUT / 'selection.json', [dict(item, pairs=[[str(b), str(g)] for b, g in item['pairs']]) for item in selection])
    for item in selection: print(f"SELECTED {item['dataset']}/{item['sequence']} frames={len(item['pairs'])}", flush=True)
    model_path = Path(os.environ['MODEL_DEST'])
    spec = importlib.util.spec_from_file_location('verified_nanovnr_model', str(model_path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ckpath = Path(os.environ['CHECKPOINT']).resolve()
    # User's own checkpoint: safe weights-only loading; never execute arbitrary pickle fallback.
    ck = torch.load(str(ckpath), map_location='cpu', weights_only=True)
    if not isinstance(ck, dict) or ck.get('architecture') != 'NanoVNRNAFNetRGB':
        raise RuntimeError('Not a NanoVNRNAFNetRGB checkpoint')
    model = module.NanoVNRNAFNetRGB(num_feat=12, grad_checkpoint=False)
    model.load_state_dict(ck['model'], strict=True)
    params = sum(v.numel() for v in model.parameters())
    if params != 414923: raise RuntimeError(f'Unexpected parameter count: {params}')
    model = model.to(device).eval()
    metadata = {
        'checkpoint': str(ckpath), 'checkpoint_sha256': sha256_file(ckpath),
        'checkpoint_step': ck.get('step'), 'recipe_id': ck.get('recipe_id'),
        'model_source_commit': os.environ['MODEL_REF'], 'model_source_blob': os.environ['MODEL_BLOB'],
        'model_source_sha256': sha256_file(model_path), 'params': params,
        'torch': torch.__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'chunk': CHUNK, 'precision': PRECISION if device.type == 'cuda' else 'fp32',
        'protocol': 'first_2_naturally_sorted_COMPLETE_test_videos_each_domain_nonoverlap_state_carry',
        'same_as_old_first100_sliding_clip_protocol': False,
        'preview_fps_assumed': FPS,
    }
    dump(OUT / 'metadata.json', metadata)
    print('CHECKPOINT=' + str(ckpath), 'STEP=' + str(ck.get('step')), 'PARAMS=' + str(params), flush=True)
    del ck
    rows = []
    fields = ['dataset', 'sequence', 'frames', 'width', 'height', 'input_psnr_rgb_db', 'output_psnr_rgb_db', 'gain_db',
              'chunk', 'precision', 'peak_gpu_gib', 'deblur_mp4', 'comparison_mp4']
    with (OUT / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); f.flush()
        for item in selection:
            row = test_video(item, model, device)
            rows.append(row); writer.writerow(row); f.flush()
            if device.type == 'cuda': torch.cuda.empty_cache()
    domains = {}
    for family in ('GoPro', 'BSD', 'DVD'):
        subset = [r for r in rows if r['dataset'] == family]
        n = sum(r['frames'] for r in subset)
        domains[family] = {'videos': len(subset), 'frames': n,
            'frame_weighted_output_psnr_rgb_db': sum(r['frames'] * r['output_psnr_rgb_db'] for r in subset) / n,
            'frame_weighted_input_psnr_rgb_db': sum(r['frames'] * r['input_psnr_rgb_db'] for r in subset) / n}
    dump(OUT / 'summary.json', {'status': 'PASS', 'metadata': metadata, 'videos': rows, 'domains': domains})
    print('\nSTATUS: PASS\nCOMPLETE_TEST_VIDEOS: 6\nTRAINING: NO\nOUTPUT_ROOT: ' + str(OUT), flush=True)
    print('SUMMARY_CSV: ' + str(OUT / 'summary.csv'), flush=True)
    print('Visual review: each dataset/video/input_output_gt.mp4 (Input | Output | GT).', flush=True)
    print('Metrics are this six-video streaming preview, NOT the old 100 sliding clips / full test-set benchmark.', flush=True)


try:
    main()
except Exception as exc:
    dump(OUT / 'FAILED.json', {'status': 'FAILED', 'error_type': type(exc).__name__, 'error': str(exc)})
    print('\nSTATUS: FAILED. Completed outputs are preserved. No resize/crop/chunk fallback was applied.', flush=True)
    traceback.print_exc()
    raise SystemExit(1)
PY
