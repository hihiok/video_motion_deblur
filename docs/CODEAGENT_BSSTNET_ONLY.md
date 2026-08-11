# CodeAgent Task — BSSTNet Only: Environment, Weights, Smoke Test, and Full 452-Frame Inference

## 0. Scope — STRICT

This task handles **BSSTNet only**.

Do NOT run, modify, reinstall, or debug:

- Shift-Net+
- DSTNet
- RealVDeblur
- Turtle
- RVRT

Do not overwrite outputs from those models.

Goal:

1. Build a clean official BSSTNet environment.
2. Obtain and validate official BSSTNet + RAFT checkpoints.
3. Run GoPro and DVD BSSTNet on the 24-frame smoke input.
4. Validate output integrity.
5. Only for checkpoints that pass smoke, run the full 452-frame business sequence.
6. Generate MP4 outputs and a final report.

Do **not** compute PSNR/SSIM because this business stream has no GT.

---

## 1. Fixed project paths

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
BSST_REPO=$ROOT/envs/bsstnet_repo
BENCH=$ROOT/benchmark
BSST_ENV=bsstnet
GPU=0

SMOKE=$BENCH/smoke/source_24
FULL_SRC=$ROOT/input/xiaobieli38_trimmed
FULL_MP4=$ROOT/input/xiaobieli38_trimmed.mp4
```

Expected official repository already exists at:

```text
/mnt/ssd1/z00919662/motion_deblur/envs/bsstnet_repo
```

Expected benchmark repository:

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark_code
```

---

## 2. Update benchmark code and read SSL instructions

```bash
set -u

ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
cd "$CODE"

export GIT_SSL_NO_VERIFY=true
git config --global http.sslVerify false

git pull --ff-only

test -f docs/CODEAGENT_DISABLE_SSL.md
cat docs/CODEAGENT_DISABLE_SSL.md
```

Apply the SSL-bypass rules in that file before any network installation.

If this file exists, source the company proxy without printing its contents:

```bash
if [ -f "$ROOT/company_proxy.md" ]; then
    source "$ROOT/company_proxy.md"
fi
```

**Never print proxy usernames, passwords, tokens, or the complete proxy URL.**

---

## 3. Global no-SSL configuration

The server is behind a trusted internal TLS-inspection proxy. For this task, certificate verification may be disabled.

```bash
conda config --set ssl_verify false

git config --global http.sslVerify false

export CONDA_SSL_VERIFY=false
export GIT_SSL_NO_VERIFY=true
export PYTHONHTTPSVERIFY=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip config set global.trusted-host \
"pypi.org files.pythonhosted.org download.pytorch.org download.openmmlab.com conda.anaconda.org github.com raw.githubusercontent.com objects.githubusercontent.com drive.google.com docs.google.com"
```

Check only non-secret settings:

```bash
conda config --show ssl_verify
git config --global --get http.sslVerify
```

Expected:

```text
ssl_verify: False
false
```

Do not stop on certificate verification errors. Do not confuse SSL errors with HTTP 403/407 errors.

---

# PART A — Create the Official BSSTNet Environment

## 4. Official software stack

Use this stack:

```text
Python        3.8
PyTorch       1.9.1+cu111
torchvision   0.10.1+cu111
torchaudio    0.9.1
mmcv-full     1.7.1
```

Do not use the shared `deblur_runtime` environment for BSSTNet.

Do not upgrade BSSTNet to current PyTorch unless this task explicitly requests it.

---

## 5. Remove only a broken BSSTNet environment

First inspect:

```bash
conda env list
```

If `bsstnet` exists, test it:

```bash
conda run -n bsstnet python - <<'PY'
import sys
print(sys.version)
PY
```

If it is a valid Python 3.8 environment, keep it and continue.

If the environment exists but Python itself is broken, remove **only** that environment:

```bash
CONDA_SSL_VERIFY=false conda env remove -n bsstnet -y
```

Never manually `cp -r` a conda environment directory. A raw directory copy is not a valid relocation method.

---

## 6. Create Python 3.8 without using repo.anaconda.com

The previous failure was HTTP 403 from `repo.anaconda.com`, which is different from SSL verification.

Avoid the default Anaconda channels and try conda-forge only:

```bash
CONDA_SSL_VERIFY=false conda create \
  -n bsstnet \
  --override-channels \
  -c conda-forge \
  python=3.8 pip -y
```

This should avoid `repo.anaconda.com`.

### Fallback if conda-forge is also blocked

Search for an existing healthy Python 3.8 conda environment:

```bash
for ENV_PATH in $(conda env list | awk '/\// {print $NF}'); do
    if [ -x "$ENV_PATH/bin/python" ]; then
        VER=$($ENV_PATH/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
        if [ "$VER" = "3.8" ]; then
            echo "FOUND_PY38=$ENV_PATH"
        fi
    fi
done
```

If a healthy Python 3.8 environment exists, clone it **offline**:

```bash
CONDA_SSL_VERIFY=false conda create \
  -n bsstnet \
  --clone <EXISTING_PYTHON38_ENV_NAME_OR_PATH> \
  --offline -y
```

Do not manually copy directories.

If neither conda-forge nor an existing local Python 3.8 environment is available, stop the environment-creation stage and report exactly:

```text
BSSTNET_ENV_BLOCKED: no reachable Python 3.8 conda channel and no local Python 3.8 environment
```

Do not modify other model environments.

---

## 7. Install packaging and numerical dependencies

```bash
PIP="conda run -n bsstnet python -m pip"

$PIP install \
  'pip<24.1' 'setuptools<70' wheel \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

$PIP install \
  numpy==1.23.5 \
  scipy==1.10.1 \
  scikit-image==0.21.0 \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

---

## 8. Install official Torch / torchvision / torchaudio

```bash
$PIP install \
  torch==1.9.1+cu111 \
  torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html \
  --trusted-host download.pytorch.org \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

Verify immediately:

```bash
conda run -n bsstnet python - <<'PY'
import torch, torchvision
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
PY
```

Expected major versions:

```text
torch: 1.9.1+cu111
torchvision: 0.10.1+cu111
```

---

## 9. Install mmcv-full 1.7.1

First use the prebuilt historical OpenMMLab CUDA 11.1 / Torch 1.9 wheel index:

```bash
$PIP install \
  mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
  --trusted-host download.openmmlab.com \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

If pip starts compiling mmcv from source instead of installing a wheel, stop that attempt and report the exact platform/wheel mismatch. Do not spend hours compiling unless no wheel exists.

Verify:

```bash
conda run -n bsstnet python - <<'PY'
import mmcv
print('mmcv:', mmcv.__version__)
PY
```

Required:

```text
mmcv: 1.7.1
```

---

## 10. Install inference-only dependencies and BSSTNet

```bash
$PIP install \
  addict future lmdb opencv-python Pillow pyyaml requests tqdm yapf einops ninja \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

cd "$BSST_REPO"
BASICSR_EXT=True conda run -n bsstnet python setup.py develop
```

Do not install unnecessary training/logging dependencies unless import errors prove they are required for inference.

---

## 11. Validate CUDA deformable convolution

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n bsstnet python - <<'PY'
import torch
import torchvision
import mmcv
from torchvision.ops import deform_conv2d

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('mmcv:', mmcv.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')

x = torch.randn(1, 1, 8, 8, device='cuda')
offset = torch.zeros(1, 18, 8, 8, device='cuda')
weight = torch.randn(1, 1, 3, 3, device='cuda')
y = deform_conv2d(x, offset, weight, padding=(1, 1))
print('deform_conv2d:', tuple(y.shape))
PY
```

Required:

```text
CUDA available: True
deform_conv2d: (1, 1, 8, 8)
```

Do not proceed to inference if this test fails.

---

# PART B — Official Weights

## 12. Required files

Canonical checkpoint locations:

```text
$BENCH/weights/bsstnet/BSST_gopro.pth
$BENCH/weights/bsstnet/BSST_dvd.pth
$BENCH/weights/bsstnet/raft-things.pth
```

Check first:

```bash
mkdir -p "$BENCH/weights/bsstnet"

ls -lh "$BENCH/weights/bsstnet" || true
```

If all three already exist and are non-empty, do not redownload them.

---

## 13. Download official BSSTNet weight folder only if files are missing

Official weight folder used by this project:

```text
https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8
```

Install gdown in a working Python environment if needed:

```bash
python -m pip install -U gdown \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

Then attempt:

```bash
TMP=$BENCH/tmp/bsstnet_drive
rm -rf "$TMP"
mkdir -p "$TMP"

gdown --folder \
  'https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8' \
  -O "$TMP"
```

Copy exact files only:

```bash
find "$TMP" -type f -iname 'BSST_gopro.pth' -exec cp -f {} "$BENCH/weights/bsstnet/BSST_gopro.pth" \;
find "$TMP" -type f -iname 'BSST_dvd.pth'   -exec cp -f {} "$BENCH/weights/bsstnet/BSST_dvd.pth" \;
find "$TMP" -type f -iname 'raft-things.pth' -exec cp -f {} "$BENCH/weights/bsstnet/raft-things.pth" \;
```

If Google Drive is blocked by the company proxy, do not substitute random unofficial mirrors. Report exactly which of the three files is missing so the user can manually place it.

---

## 14. Validate weight files

```bash
for F in \
  "$BENCH/weights/bsstnet/BSST_gopro.pth" \
  "$BENCH/weights/bsstnet/BSST_dvd.pth" \
  "$BENCH/weights/bsstnet/raft-things.pth"
do
    if [ ! -s "$F" ]; then
        echo "MISSING: $F"
        exit 20
    fi
    ls -lh "$F"
    file "$F"
done

sha256sum \
  "$BENCH/weights/bsstnet/BSST_gopro.pth" \
  "$BENCH/weights/bsstnet/BSST_dvd.pth" \
  "$BENCH/weights/bsstnet/raft-things.pth" \
  | tee "$BENCH/manifests/bsstnet_weights.sha256"
```

Check that the `.pth` files are not HTML pages or Git-LFS pointer text.

Test checkpoint deserialization using the BSSTNet environment:

```bash
conda run -n bsstnet python - <<'PY'
import torch
from pathlib import Path

root = Path('/mnt/ssd1/z00919662/motion_deblur/benchmark/weights/bsstnet')
for name in ['BSST_gopro.pth', 'BSST_dvd.pth', 'raft-things.pth']:
    p = root / name
    print('\nFILE', p, 'bytes=', p.stat().st_size)
    obj = torch.load(str(p), map_location='cpu')
    if isinstance(obj, dict):
        print('keys:', list(obj.keys())[:20])
    else:
        print('type:', type(obj))
PY
```

Do not proceed if deserialization fails.

---

# PART C — Prepare 24-Frame Smoke Input

## 15. Verify or create the smoke folder

```bash
SMOKE=$BENCH/smoke/source_24
mkdir -p "$SMOKE"

COUNT=$(find "$SMOKE" -maxdepth 1 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)

echo "smoke_count=$COUNT"
```

If `COUNT != 24`, recreate it from the first 24 canonical business frames without recompression:

```bash
rm -rf "$SMOKE"
mkdir -p "$SMOKE"

python3 - <<'PY'
from pathlib import Path
import shutil, re

src = Path('/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed')
dst = Path('/mnt/ssd1/z00919662/motion_deblur/benchmark/smoke/source_24')

def key(p):
    parts = re.split(r'(\d+)', p.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]

files = sorted([p for p in src.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}], key=key)
if len(files) < 24:
    raise RuntimeError(f'Only {len(files)} source frames')

for p in files[:24]:
    shutil.copy2(p, dst / p.name)
print('created', len(list(dst.iterdir())), 'frames')
PY
```

Verify exactly 24 frames before inference.

---

# PART D — 24-Frame Smoke Tests

## 16. Common smoke settings

For smoke testing use:

```text
clip_len = 24
temporal_overlap = 8
patch_size = 256
patch_overlap = 64
```

BSSTNet's spatial patch size must remain 256 for this adapter. Do not change it to 128 or 512.

Run GoPro and DVD independently. A failure in one must not stop the other.

Before every GPU run:

```bash
nvidia-smi
```

Do not kill another user's GPU process.

Use only:

```bash
CUDA_VISIBLE_DEVICES=0
```

---

## 17. Smoke — GoPro checkpoint

```bash
OUT=$BENCH/outputs/bsstnet_gopro
rm -rf "$OUT"
mkdir -p "$OUT/frames" "$BENCH/logs"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
conda run -n bsstnet python \
"$CODE/adapters/bsstnet_infer.py" \
  --repo "$BSST_REPO" \
  --input "$SMOKE" \
  --output "$OUT/frames" \
  --checkpoint "$BENCH/weights/bsstnet/BSST_gopro.pth" \
  --raft-checkpoint "$BENCH/weights/bsstnet/raft-things.pth" \
  --clip-len 24 \
  --temporal-overlap 8 \
  --patch-size 256 \
  --patch-overlap 64 \
  --device cuda:0 \
  2>&1 | tee "$BENCH/logs/bsstnet_gopro_smoke.log"
```

Do not classify as PASS yet.

Validate:

```bash
python3 "$CODE/scripts/check_output.py" \
  --input "$SMOKE" \
  --output "$OUT/frames" \
  --report "$OUT/check_report.json"

cat "$OUT/check_report.json"
find "$OUT/frames" -maxdepth 1 -type f | wc -l
```

PASS requires:

```text
input_count = 24
output_count = 24
errors = []
passed = true
```

---

## 18. Smoke — DVD checkpoint

```bash
OUT=$BENCH/outputs/bsstnet_dvd
rm -rf "$OUT"
mkdir -p "$OUT/frames"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
conda run -n bsstnet python \
"$CODE/adapters/bsstnet_infer.py" \
  --repo "$BSST_REPO" \
  --input "$SMOKE" \
  --output "$OUT/frames" \
  --checkpoint "$BENCH/weights/bsstnet/BSST_dvd.pth" \
  --raft-checkpoint "$BENCH/weights/bsstnet/raft-things.pth" \
  --clip-len 24 \
  --temporal-overlap 8 \
  --patch-size 256 \
  --patch-overlap 64 \
  --device cuda:0 \
  2>&1 | tee "$BENCH/logs/bsstnet_dvd_smoke.log"
```

Validate exactly as for GoPro.

---

## 19. Smoke sanity gate

For each successful checkpoint, inspect at least 3 output frames manually or generate a montage:

- early frame
- middle frame
- late frame

Check for:

- black/constant output
- RGB/BGR swap
- severe brightness shift
- checkerboard artifacts
- tile seams
- obvious temporal discontinuity
- output accidentally identical to input

Only a checkpoint with `check_report.json passed=true` and visually plausible frames may proceed to full inference.

---

# PART E — Prepare Full 452-Frame Input

## 20. Build canonical full input

```bash
FULL_INPUT=$BENCH/input_frames

python3 "$CODE/scripts/prepare_input.py" \
  --source-frames "$FULL_SRC" \
  --source-mp4 "$FULL_MP4" \
  --output "$FULL_INPUT"
```

Verify:

```bash
COUNT=$(find "$FULL_INPUT" -maxdepth 1 -type f | wc -l)
echo "full_count=$COUNT"

if [ "$COUNT" -ne 452 ]; then
    echo "ERROR: expected 452 frames"
    exit 30
fi
```

Read FPS:

```bash
FPS=$(python3 - <<'PY'
import json
p='/mnt/ssd1/z00919662/motion_deblur/benchmark/manifests/input.json'
print(json.load(open(p))['fps'])
PY
)
echo "FPS=$FPS"
```

---

# PART F — Full 452-Frame Inference

## 21. Full-run strategy

Run only checkpoints whose smoke test passed.

First attempt:

```text
clip_len = 48
temporal_overlap = 16
patch_size = 256
patch_overlap = 64
```

This matches the intended long-sequence usage of the current adapter and the official test configuration's 48-frame sequence length.

If CUDA OOM occurs, keep spatial patch size 256 and reduce only temporal clip length:

```text
48 / overlap 16
→ 24 / overlap 8
→ 12 / overlap 4
→ 8 / overlap 2
```

Use the largest clip length that succeeds.

Do not resize the input.

Do not change BSSTNet channels, blocks, attention structure, or checkpoint.

---

## 22. Full GoPro inference

Run only if GoPro smoke passed.

```bash
OUT=$BENCH/outputs/bsstnet_gopro
rm -rf "$OUT/frames"
mkdir -p "$OUT/frames"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
conda run -n bsstnet python \
"$CODE/adapters/bsstnet_infer.py" \
  --repo "$BSST_REPO" \
  --input "$FULL_INPUT" \
  --output "$OUT/frames" \
  --checkpoint "$BENCH/weights/bsstnet/BSST_gopro.pth" \
  --raft-checkpoint "$BENCH/weights/bsstnet/raft-things.pth" \
  --clip-len 48 \
  --temporal-overlap 16 \
  --patch-size 256 \
  --patch-overlap 64 \
  --device cuda:0 \
  2>&1 | tee "$BENCH/logs/bsstnet_gopro_full.log"
```

If OOM, retry 24/8, then 12/4, then 8/2. Clear the partial output directory before each retry.

After a successful run:

```bash
python3 "$CODE/scripts/check_output.py" \
  --input "$FULL_INPUT" \
  --output "$OUT/frames" \
  --report "$OUT/check_report.json"

cat "$OUT/check_report.json"
```

Required:

```text
input_count = 452
output_count = 452
errors = []
passed = true
```

---

## 23. Full DVD inference

Run only if DVD smoke passed.

Use the same procedure as GoPro with:

```text
checkpoint = $BENCH/weights/bsstnet/BSST_dvd.pth
output     = $BENCH/outputs/bsstnet_dvd/frames
log        = $BENCH/logs/bsstnet_dvd_full.log
```

Start with 48/16 and use the same OOM fallback:

```text
48/16 → 24/8 → 12/4 → 8/2
```

Validate 452 frames and `passed=true`.

---

# PART G — Generate MP4 Outputs

## 24. Encode only valid 452-frame outputs

For GoPro if full run passed:

```bash
python3 "$CODE/scripts/frames_to_video.py" \
  --frames "$BENCH/outputs/bsstnet_gopro/frames" \
  --output "$BENCH/outputs/bsstnet_gopro/output_full.mp4" \
  --fps "$FPS"
```

For DVD if full run passed:

```bash
python3 "$CODE/scripts/frames_to_video.py" \
  --frames "$BENCH/outputs/bsstnet_dvd/frames" \
  --output "$BENCH/outputs/bsstnet_dvd/output_full.mp4" \
  --fps "$FPS"
```

Do not encode a full-result MP4 from partial output.

---

# PART H — Required Final Report

## 25. Return this exact summary

```text
BSSTNet Environment
-------------------
Python:
PyTorch:
torchvision:
torchaudio:
mmcv-full:
Torch CUDA build:
CUDA available:
deform_conv2d test: PASS/FAIL

Weights
-------
BSST_gopro.pth: path / bytes / sha256
BSST_dvd.pth: path / bytes / sha256
raft-things.pth: path / bytes / sha256

Smoke Test
----------
GoPro: PASS/FAIL
GoPro output frames:
GoPro check_report passed:
GoPro exact error if failed:

DVD: PASS/FAIL
DVD output frames:
DVD check_report passed:
DVD exact error if failed:

Full 452-Frame Inference
------------------------
GoPro: PASS/FAIL/NOT RUN
GoPro output frames:
GoPro successful clip_len / overlap:
GoPro runtime seconds:
GoPro MP4 path:
GoPro exact error if failed:

DVD: PASS/FAIL/NOT RUN
DVD output frames:
DVD successful clip_len / overlap:
DVD runtime seconds:
DVD MP4 path:
DVD exact error if failed:

GPU / Memory
------------
GPU model:
Peak memory if measured:
OOM fallback used: yes/no

Code Changes
------------
Official BSSTNet code modified: NO
Benchmark adapter modified: NO unless explicitly required and documented
```

---

# 26. Failure rules

1. Do not modify official BSSTNet network architecture to make the test pass.
2. Do not use random unofficial checkpoints.
3. Do not replace Torch 1.9.1 with Torch 2.x simply because installation is easier.
4. Do not manually copy a conda environment directory.
5. Do not report PASS based only on process exit status.
6. Smoke PASS requires exactly 24 valid frames + `check_report.json passed=true`.
7. Full PASS requires exactly 452 valid frames + `check_report.json passed=true`.
8. If GoPro fails, still attempt DVD smoke; if DVD fails, still attempt GoPro smoke.
9. If one full checkpoint fails, continue the other checkpoint independently.
10. Do not touch outputs or environments for Shift-Net+, DSTNet, RealVDeblur, Turtle, or RVRT.

The task ends after BSSTNet GoPro/DVD have either completed full inference successfully or have exact reproducible failure evidence.