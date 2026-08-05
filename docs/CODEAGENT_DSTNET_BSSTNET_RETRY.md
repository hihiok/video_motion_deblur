# CodeAgent retry: DSTNet OOM and BSSTNet conda SSL

## Scope

- Shift-Net+ already passes. Do not reinstall or modify it.
- Ignore RealVDeblur until `diffusion_pytorch_model.safetensors` finishes downloading.
- Fix and validate DSTNet and BSSTNet only.
- Do not modify the official model architecture files.

## 1. Update benchmark code

```bash
cd /mnt/ssd1/z00919662/motion_deblur/benchmark_code
git pull --ff-only

git log -5 --oneline
grep -n "tile-size" adapters/dstnet_infer.py
test -f scripts/run_dstnet_low_memory.sh
test -f scripts/setup_bsstnet_conda_no_ssl.sh
```

## 2. DSTNet low-memory 24-frame smoke test

Use the existing 24-frame smoke folder when present:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
SMOKE=$ROOT/benchmark/smoke/source_24

find "$SMOKE" -maxdepth 1 -type l -o -type f | wc -l
```

Run all three DSTNet checkpoints with temporal and spatial tiling:

```bash
cd "$CODE"

ROOT="$ROOT" \
CODE="$CODE" \
DST_ENV=deblur_runtime \
INPUT_SRC="$SMOKE" \
INPUT="$ROOT/benchmark/smoke/dstnet_input_frames" \
DST_CLIP_LEN=4 \
DST_TEMPORAL_OVERLAP=1 \
DST_TILE_SIZE=512 \
DST_TILE_OVERLAP=64 \
DST_MIN_TILE_SIZE=256 \
GPU=0 \
bash scripts/run_dstnet_low_memory.sh
```

The adapter automatically retries a failed 512 tile with a smaller tile down to 256. Expected log examples:

```text
DSTNet chunk 1/... dynamic_backend=pytorch_unfold; tile=512
```

or, after an OOM retry:

```text
CUDA OOM ... retrying with tile=256
DSTNet chunk 1/... dynamic_backend=pytorch_unfold; tile=256
```

Required output for each checkpoint:

```text
benchmark/outputs/dstnet_gopro/check_report.json
benchmark/outputs/dstnet_dvd/check_report.json
benchmark/outputs/dstnet_bsd/check_report.json
```

Each report must contain:

```json
{
  "input_count": 24,
  "output_count": 24,
  "passed": true
}
```

### DSTNet emergency minimum-memory retry

Only if tile 256 with four frames still OOM:

```bash
ROOT="$ROOT" \
CODE="$CODE" \
DST_ENV=deblur_runtime \
INPUT_SRC="$SMOKE" \
INPUT="$ROOT/benchmark/smoke/dstnet_input_frames" \
DST_CLIP_LEN=2 \
DST_TEMPORAL_OVERLAP=0 \
DST_TILE_SIZE=256 \
DST_TILE_OVERLAP=32 \
DST_MIN_TILE_SIZE=192 \
GPU=0 \
bash scripts/run_dstnet_low_memory.sh
```

Do not resize the input video. Do not reduce the DSTNet channel count or block count.

## 3. DSTNet full sequence

Run only after the 24-frame smoke test passes:

```bash
ROOT="$ROOT" \
CODE="$CODE" \
DST_ENV=deblur_runtime \
DST_CLIP_LEN=4 \
DST_TEMPORAL_OVERLAP=1 \
DST_TILE_SIZE=512 \
DST_TILE_OVERLAP=64 \
DST_MIN_TILE_SIZE=256 \
GPU=0 \
bash scripts/run_dstnet_low_memory.sh
```

The full sequence contains 452 frames. The adapter processes temporal chunks independently and never sends all 452 frames to the GPU at once.

## 4. Create the official BSSTNet conda environment without SSL verification

The official stack is required:

```text
Python 3.8
Torch 1.9.1+cu111
Torchvision 0.10.1+cu111
Torchaudio 0.9.1
mmcv-full 1.7.1
```

Run the prepared installer:

```bash
cd "$CODE"

ROOT="$ROOT" \
BSST_ENV=bsstnet \
GPU=0 \
bash scripts/setup_bsstnet_conda_no_ssl.sh
```

The installer does all of the following:

```text
conda config --set ssl_verify false
CONDA_SSL_VERIFY=false
git http.sslVerify=false
pip --trusted-host for PyPI, PyTorch and OpenMMLab
```

It does not print or save proxy credentials.

Required validation at the end:

```text
torch: 1.9.1+cu111
torchvision: 0.10.1+cu111
mmcv: 1.7.1
CUDA available: True
torchvision deform_conv2d: (1, 1, 8, 8)
```

If conda still reports a certificate error, confirm the setting and retry the same installer; do not change versions:

```bash
conda config --show ssl_verify
CONDA_SSL_VERIFY=false conda create --insecure -n bsstnet python=3.8 pip -y
```

Then rerun:

```bash
bash scripts/setup_bsstnet_conda_no_ssl.sh
```

## 5. BSSTNet smoke test

Run only when the environment and all three weights exist:

```bash
ls -lh \
  "$ROOT/benchmark/weights/bsstnet/BSST_gopro.pth" \
  "$ROOT/benchmark/weights/bsstnet/BSST_dvd.pth" \
  "$ROOT/benchmark/weights/bsstnet/raft-things.pth"
```

```bash
ROOT="$ROOT" \
CODE="$CODE" \
BSST_ENV=bsstnet \
INPUT_SRC="$SMOKE" \
INPUT="$ROOT/benchmark/smoke/bsstnet_input_frames" \
GPU=0 \
bash run_all.sh --model=bsstnet
```

Do not let `run_all.sh` fall back to the current Python. BSSTNet must run in the `bsstnet` conda environment because `torchvision.ops.deform_conv2d` must match Torch 1.9.1/CUDA 11.1.

## 6. Final report

Report separately:

```text
DSTNet GOPRO: pass/fail, effective tile size, peak VRAM if available
DSTNet DVD: pass/fail, effective tile size, peak VRAM if available
DSTNet BSD: pass/fail, effective tile size, peak VRAM if available
BSSTNet environment: pass/fail and exact package versions
BSSTNet GoPro/DVD: pass/fail and frame count
```

One model failure must not prevent testing the other model.
