# CodeAgent Task — DSTNet+ Base (TPAMI 2025) GoPro Checkpoint Business Inference

## Goal

Run **DSTNet+ Base (TPAMI 2025)** on the existing 452-frame business stream and produce one final output MP4.

This task is intentionally narrow:

- Model: **DSTNet+ Base**, not DSTNet 2023 and not DSTNet+ Large.
- Checkpoint training dataset: **GoPro**, matching the previously used Shift-Net-s GoPro Ours-s checkpoint for fair subjective comparison.
- Checkpoint: `DSTNetPlus_base_gopro.pth` only.
- Do not run DVD/BSD checkpoints.
- Do not calculate PSNR/SSIM because the business stream has no GT.
- Do not modify the official DSTNet+ network source.
- The final requested artifact is `output.mp4` plus output-integrity metadata.

## Fixed paths

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
INPUT_FRAMES=$ROOT/input/xiaobieli38_trimmed
INPUT_MP4=$ROOT/input/xiaobieli38_trimmed.mp4
```

Expected input:

```text
452 frames
1280x720
```

Final output directory:

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/dstnetplus_base_gopro_2025/
├── frames/
├── check_report.json
├── run_metadata.json
├── checkpoint.sha256
└── output.mp4
```

## Important model identity

Official DSTNet+ Base network configuration:

```text
DSTNetPlus_Final
num_feat=64
num_kernel_block=3
num_block=15
nonblind_denoise=False
```

Official GoPro Base checkpoint:

```text
DSTNetPlus_base_gopro.pth
```

Do not substitute:

```text
DSTNetPlus_base_dvd.pth
DSTNetPlus_base_bsd*.pth
DSTNetPlus_L_*.pth
```

## SSL handling

This server is behind an internal HTTPS-inspection proxy. Certificate verification may be disabled for this task.

Do not print proxy credentials.

```bash
conda config --set ssl_verify false || true
git config --global http.sslVerify false
export CONDA_SSL_VERIFY=false
export GIT_SSL_NO_VERIFY=true
export PYTHONHTTPSVERIFY=0
```

HTTP 403 is not an SSL error; do not repeatedly retry certificate settings if the server explicitly returns 403.

## Step 1 — Update benchmark repository

```bash
set -e

ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code

cd "$CODE"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true

git pull --ff-only
git log -8 --oneline
```

Confirm these files exist:

```bash
test -f adapters/dstnetplus_compat.py
test -f adapters/dstnetplus_infer.py
test -f scripts/run_dstnetplus_base_gopro_2025.sh
```

## Step 2 — Do not build a new legacy environment

Use the already working benchmark environment:

```text
deblur_runtime
```

The benchmark compatibility layer deliberately avoids requiring the full BasicSR/CuPy package stack.

Verify:

```bash
conda run -n deblur_runtime python - <<'PY'
import torch
import numpy
from PIL import Image
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('numpy:', numpy.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable')
PY
```

Do not install or downgrade PyTorch unless this command fails for a genuine missing-runtime reason.

## Step 3 — Run the prepared one-command pipeline

Before launch:

```bash
nvidia-smi
```

Do not kill another user's process.

Then execute:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
CODE=/mnt/ssd1/z00919662/motion_deblur/benchmark_code \
DSTNETPLUS_ENV=deblur_runtime \
GPU=0 \
bash /mnt/ssd1/z00919662/motion_deblur/benchmark_code/scripts/run_dstnetplus_base_gopro_2025.sh
```

The runner will automatically:

1. clone/pin official `sunny2109/DSTNet-plus` source if needed;
2. download the official `DSTNetPlus_base_gopro.pth` release asset with SSL verification disabled;
3. validate the checkpoint can be parsed;
4. verify the input has exactly 452 frames;
5. instantiate only the official Base architecture (`64/3/15`);
6. run tiled temporal inference;
7. validate 452 output frames and dimensions;
8. convert the output frames to the source FPS;
9. generate `output.mp4`.

## Compatibility policy

The official DSTNet+ architecture source is loaded directly.

The released dynamic depthwise convolution normally uses a CuPy JIT CUDA kernel. To avoid old-package dependency failures in the current benchmark environment, the adapter provides an algebraically equivalent PyTorch `unfold` implementation while preserving:

- official architecture topology;
- module/state-dict names;
- official checkpoint weights;
- 64 feature channels;
- 3 progressive kernel blocks;
- 15 residual blocks;
- GoPro Base checkpoint.

Do not edit `envs/dstnetplus_repo/basicsr/archs/dstnetplus_deblur_arch.py`.

## Default low-memory inference settings

The runner defaults to:

```text
clip_len=8
temporal_overlap=2
tile_size=384
tile_overlap=64
min_tile_size=192
AMP=true
```

The adapter automatically retries a smaller spatial tile on CUDA OOM.

These are inference-memory controls only; do not resize the source frames.

## If OOM still occurs

First confirm GPU occupancy:

```bash
nvidia-smi
```

If GPU 0 has significant unrelated allocation, do not kill it. Record the occupancy.

If enough memory is available but the default still OOMs, retry conservatively:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
CODE=/mnt/ssd1/z00919662/motion_deblur/benchmark_code \
DSTNETPLUS_ENV=deblur_runtime \
DSTNETPLUS_CLIP_LEN=4 \
DSTNETPLUS_TEMPORAL_OVERLAP=1 \
DSTNETPLUS_TILE_SIZE=256 \
DSTNETPLUS_TILE_OVERLAP=32 \
DSTNETPLUS_MIN_TILE_SIZE=128 \
GPU=0 \
bash /mnt/ssd1/z00919662/motion_deblur/benchmark_code/scripts/run_dstnetplus_base_gopro_2025.sh
```

Do not lower the 1280x720 input resolution.

## Required validation

After success:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
OUT=$ROOT/benchmark/outputs/dstnetplus_base_gopro_2025

find "$OUT/frames" -maxdepth 1 -type f | wc -l
cat "$OUT/check_report.json"
cat "$OUT/run_metadata.json"
ls -lh "$OUT/output.mp4"
```

PASS requires all of:

```text
output frame count = 452
check_report.json passed = true
no unreadable frames
no size mismatch
output.mp4 exists and is non-empty
```

Also verify metadata says:

```text
model = DSTNet+ Base
paper = TPAMI 2025
training_checkpoint_dataset = GoPro
num_feat = 64
num_kernel_block = 3
num_block = 15
nonblind_denoise = false
```

## Visual sanity check

Extract a few frames from the resulting MP4 or inspect PNG outputs around the beginning, middle, and end.

Check for:

- obvious spatial tile seams;
- temporal discontinuity near clip boundaries;
- color-channel swaps;
- black/NaN frames;
- incorrect frame ordering.

If any of these occur, classify the run as a pipeline failure even if the frame count is 452.

Do not calculate no-reference image scores just to declare success.

## Required final CodeAgent report

Return:

```text
DSTNETPLUS_BASE_2025_BUSINESS_INFERENCE

STATUS: PASS / FAIL
MODEL: DSTNet+ Base (TPAMI 2025)
CHECKPOINT: DSTNetPlus_base_gopro.pth
CHECKPOINT_TRAINING_DATASET: GoPro
CHECKPOINT_SHA256:
OFFICIAL_REPO_COMMIT:

INPUT_FRAMES: 452
INPUT_RESOLUTION: 1280x720
INPUT_FPS:

OUTPUT_FRAMES:
CHECK_REPORT_PASSED:
OUTPUT_MP4:
OUTPUT_MP4_SIZE:

PARAMETER_COUNT:
DYNAMIC_CONV_BACKEND:
CLIP_LEN:
TEMPORAL_OVERLAP:
REQUESTED_TILE_SIZE:
MIN_EFFECTIVE_TILE_SIZE:
RUNTIME_SECONDS:
PEAK_GPU_MEMORY_IF_AVAILABLE:

VISUAL_SANITY: PASS / FAIL
NOTES:
```

The task is complete when `output.mp4` exists and the 452-frame integrity check passes.
