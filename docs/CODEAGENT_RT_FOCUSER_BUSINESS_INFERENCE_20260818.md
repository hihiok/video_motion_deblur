# CODEAGENT_RT_FOCUSER_BUSINESS_INFERENCE_20260818

## Goal

Run the official **RT-Focuser** GoPro deblurring checkpoint on the business video:

```text
/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Run **full inference directly**. Do not do a smoke-only run.

Use the wrapper already prepared in this repository:

```text
scripts/run_rt_focuser_business_video.py
```

Do **not** rewrite, replace, or reimplement RT-Focuser. Do **not** modify the wrapper unless the user explicitly asks for a new patch.

Official upstream:

```text
https://github.com/ReaganWu/RT-Focuser.git
```

Pin upstream to:

```text
4c8e12d28c2801f34cc1153e9ad8702b7bce657a
```

Official checkpoint:

```text
Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth
```

The official repository contains this checkpoint directly. The wrapper uses the PyTorch checkpoint and CUDA; do not use the official `Inference_Video_ONNX.py`, because that script explicitly creates an ONNX Runtime session with `CPUExecutionProvider`.

---

## 0. Proxy and SSL

The project uses an internal proxy. Credentials must remain in the server-local secure activation file and must not be committed to the public GitHub repository.

Load the existing proxy configuration first:

```bash
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]; then
    source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
elif [ -f "/mnt/ssd1/z00919662/anaconda3/envs/RVRT/etc/conda/activate.d/proxy_env.sh" ]; then
    source "/mnt/ssd1/z00919662/anaconda3/envs/RVRT/etc/conda/activate.d/proxy_env.sh"
fi

# Skip Git SSL verification in this internal proxy environment.
git config --global http.sslVerify false

# If pip is needed, tolerate the internal TLS interception.
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"
```

Print only whether proxy variables are set. **Never print proxy URLs or credentials.**

```bash
python - <<'PY'
import os
print('http_proxy set:', bool(os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')))
print('https_proxy set:', bool(os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')))
PY
```

---

## 1. Fixed paths

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
INPUT_VIDEO="$ROOT/input/xiaobieli38_trimmed.mp4"
BENCH_REPO="$ROOT/video_motion_deblur_rt_focuser"
RT_REPO="$ROOT/envs/RT-Focuser"
OUT_DIR="$ROOT/runs/rt_focuser_20260818"
```

Check the input immediately:

```bash
test -f "$INPUT_VIDEO" || { echo "MISSING_INPUT: $INPUT_VIDEO"; exit 1; }
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames \
  -of default=nw=1 "$INPUT_VIDEO"
sha256sum "$INPUT_VIDEO"
```

Do not resize the business video. The intended model input is the native source resolution.

---

## 2. Get the prepared benchmark code

```bash
cd "$ROOT"

if [ ! -d "$BENCH_REPO/.git" ]; then
    git clone -b agent/rt-focuser-business-inference \
      https://github.com/hihiok/video_motion_deblur.git "$BENCH_REPO"
else
    cd "$BENCH_REPO"
    git fetch origin agent/rt-focuser-business-inference
    git checkout agent/rt-focuser-business-inference
    git reset --hard origin/agent/rt-focuser-business-inference
fi

cd "$BENCH_REPO"
git status --short
git rev-parse HEAD

test -f scripts/run_rt_focuser_business_video.py || {
    echo "MISSING_WRAPPER: scripts/run_rt_focuser_business_video.py"
    exit 1
}
```

Do not edit this repository during execution.

---

## 3. Clone official RT-Focuser and pin the revision

```bash
mkdir -p "$ROOT/envs"

if [ ! -d "$RT_REPO/.git" ]; then
    git clone https://github.com/ReaganWu/RT-Focuser.git "$RT_REPO"
fi

cd "$RT_REPO"
git fetch origin
git checkout --detach 4c8e12d28c2801f34cc1153e9ad8702b7bce657a

git rev-parse HEAD
sha256sum Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth
ls -lh Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth
```

Expected checkpoint file size is about 23.9 MB. If Git LFS/pointer corruption is observed, stop and report it; do not substitute another checkpoint silently.

---

## 4. Python environment

Prefer an already-working CUDA PyTorch environment on this server. `RVRT` is a known-good candidate.

```bash
source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate RVRT

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA_NOT_AVAILABLE')
PY
```

Check the lightweight runtime dependencies:

```bash
python - <<'PY'
import cv2, numpy
print('cv2:', cv2.__version__)
print('numpy:', numpy.__version__)
PY
```

Only if `cv2` or `numpy` is missing, install the minimum missing package with SSL trust enabled. Do **not** reinstall torch/torchvision when the existing CUDA environment already works.

Example fallback:

```bash
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  opencv-python-headless numpy
```

---

## 5. Select GPU

Pick the GPU with the most free memory rather than assuming GPU 0.

```bash
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')

echo "PHYSICAL_GPU=$GPU"
export CUDA_VISIBLE_DEVICES="$GPU"
```

Inside Python this selected GPU becomes `cuda:0`.

---

## 6. Full native-resolution inference

Clear only the target output directory, not any existing benchmark outputs from other models.

```bash
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

cd "$BENCH_REPO"

python scripts/run_rt_focuser_business_video.py \
  --input-video "$INPUT_VIDEO" \
  --rt-repo "$RT_REPO" \
  --output-dir "$OUT_DIR" \
  --device cuda:0 \
  --fp16
```

Important requirements:

- Process **all frames**.
- Do not resize to 256x256.
- Do not crop the source video.
- Do not use ONNX CPU inference.
- The wrapper pads only when dimensions are not divisible by 16 and crops back to the exact source size. For 1280x720, no padding is required because both dimensions are divisible by 16.
- Save lossless PNG frames in `$OUT_DIR/frames`.
- Generate `$OUT_DIR/rt_focuser_output.mp4`.
- Preserve source audio when available.
- Measure pure synchronized model inference time separately from video decode/PNG write/ffmpeg encode.

---

## 7. Validate outputs

```bash
cat "$OUT_DIR/run_summary.txt"

find "$OUT_DIR/frames" -maxdepth 1 -type f -name '*.png' | wc -l
ls -lh "$OUT_DIR/rt_focuser_output.mp4"
ls -lh "$OUT_DIR/preview_input_output.jpg"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames \
  -of default=nw=1 "$OUT_DIR/rt_focuser_output.mp4"
```

The output frame count and resolution must match the input. If they do not, set `STATUS: FAIL` and report the exact mismatch.

---

## 8. Final response

Return a compact report containing exactly these key items:

```text
RT_FOCUSER_BUSINESS_INFERENCE_20260818
STATUS: PASS/FAIL
UPSTREAM_COMMIT: ...
CHECKPOINT: ...
CHECKPOINT_SHA256: ...
INPUT_VIDEO: ...
INPUT_RESOLUTION: ...
INPUT_FRAMES: ...
PHYSICAL_GPU: ...
TORCH/CUDA: ...
PARAMETERS: ...
FP16_AUTOCAST: true/false
MODEL_MS_PER_FRAME: ...
MODEL_FPS: ...
PEAK_TORCH_ALLOCATED_GIB: ...
OUTPUT_FRAMES: ...
OUTPUT_VIDEO: ...
PREVIEW: ...
```

If anything fails, do not invent replacement code. Report the failing command and traceback/error text.
