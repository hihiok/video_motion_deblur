# CodeAgent task: train RT-Focuser + Shift-Net-s + DSTNet T=6

## Goal and non-negotiable constraints

Run the prepared code from GitHub. Do not write, rewrite or patch model/training
code on the server.

- Baseline/checkpoint: official RT-Focuser Standard GoPro checkpoint.
- Temporal input: exactly T=6 consecutive frames.
- Training data: GoPro + BSD + DVD, balanced 1:1:1.
- Candidate parameters and per-output-frame arithmetic must not exceed the
  original RT-Focuser Standard model.
- Run strict dataset, checkpoint, unit-test and complexity gates before formal
  training.
- Do not kill another user's GPU process. Pick a genuinely free GPU.
- If a source-code problem is found, stop and report it; do not make local code
  edits. Preserve `git status --short` as empty.

Repository:

```text
https://github.com/hihiok/video_motion_deblur.git
branch: agent/rtfocuser-shift-dst-t6-v1
minimum code commit: TO_BE_FILLED
```

## Proxy and SSL setup

The server is behind an authenticated HTTPS-inspection proxy. Proxy credentials
must stay in the existing protected conda activation file and must never be
printed, pasted into logs, or committed to this public repository.

After activating the selected environment, source its existing proxy file when
present:

```bash
PROXY_FILE="$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
if [[ -f "$PROXY_FILE" ]]; then
  source "$PROXY_FILE"
fi

export GIT_SSL_NO_VERIFY=true
export CONDA_SSL_VERIFY=false
export PYTHONHTTPSVERIFY=0
git config --global http.sslVerify false
conda config --set ssl_verify false || true
```

Do not run commands that echo proxy environment values. HTTP 401/403/407 is an
authentication or authorization issue, not an SSL-certificate issue.

## Fixed paths

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
CONFIG=$CODE/configs/rtf_t6_gopro_bsd_dvd.yaml
RUN=$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_v1
```

The config safely probes these existing dataset candidates:

```text
GoPro: $ROOT/datasets/GoPro, $ROOT/datasets/gopro, $ROOT/datasets/GOPRO_Large
BSD:   $ROOT/datasets/BSD, /mnt/ssd1/z00919662/datasets/BSD,
       /mnt/ssd1/z00919662/RVRT/datasets/BSD
DVD:   $ROOT/datasets/DVD, $ROOT/datasets/dvd
```

Do not move, rename or rewrite source datasets. The audit understands
`split/{blur,gt}/sequence`, `split/sequence/{blur,sharp}`, and
`{blur,gt}/split/sequence` layouts.

## Step 1: update the exact GitHub branch

```bash
set -euo pipefail

ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code

if [[ ! -d "$CODE/.git" ]]; then
  git clone https://github.com/hihiok/video_motion_deblur.git "$CODE"
fi
cd "$CODE"
git fetch origin agent/rtfocuser-shift-dst-t6-v1
git switch --detach origin/agent/rtfocuser-shift-dst-t6-v1
git status --short
git rev-parse HEAD
```

PASS requires an empty `git status --short`. The checked-out HEAD must contain
the minimum commit listed above or a descendant on the same branch:

```bash
git merge-base --is-ancestor TO_BE_FILLED HEAD
```

## Step 2: select the existing CUDA environment

Prefer `deblur_runtime`; use `RVRT` only if the first environment is absent.

```bash
source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx deblur_runtime; then
  ENV_NAME=deblur_runtime
elif conda env list | awk '{print $1}' | grep -qx RVRT; then
  ENV_NAME=RVRT
else
  echo "HUMAN_ACTION_REQUIRED: no deblur_runtime or RVRT conda environment"
  exit 2
fi

conda activate "$ENV_NAME"
PROXY_FILE="$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
if [[ -f "$PROXY_FILE" ]]; then source "$PROXY_FILE"; fi
export GIT_SSL_NO_VERIFY=true CONDA_SSL_VERIFY=false PYTHONHTTPSVERIFY=0
git config --global http.sslVerify false
conda config --set ssl_verify false || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY

python -m pip install -r "$CODE/requirements_rtf_t6.txt"
```

Do not downgrade or replace the installed CUDA PyTorch build.

## Step 3: locate and verify the official RT-Focuser checkpoint

```bash
for p in \
  "$ROOT/envs/RT-Focuser/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth" \
  "$ROOT/envs/rt_focuser_repo/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth" \
  "$ROOT/benchmark/weights/rt_focuser/GoPro_RT_Focuser_Standard_256.pth"; do
  if [[ -s "$p" ]]; then PRETRAINED="$p"; break; fi
done
```

If it is absent, clone the official MIT-licensed repository; its checkpoint is
tracked in the repository:

```bash
if [[ -z "${PRETRAINED:-}" ]]; then
  RT_REPO="$ROOT/envs/RT-Focuser"
  if [[ ! -d "$RT_REPO/.git" ]]; then
    git clone https://github.com/ReaganWu/RT-Focuser.git "$RT_REPO"
  fi
  PRETRAINED="$RT_REPO/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth"
fi

test -s "$PRETRAINED"
sha256sum "$PRETRAINED"
```

Expected SHA256:

```text
6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb
```

Stop on a different hash. Do not substitute an ONNX/int8 file or a Shift-Net,
DSTNet or RT-Focuser-SPPF checkpoint.

## Step 4: choose a free GPU and run preflight gates

```bash
nvidia-smi
# Select a free GPU without killing any process.
GPU=0
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

cd "$CODE"
python -m pytest -q

mkdir -p "$RUN/audit"
python tools/audit_rtf_t6_data.py \
  --config "$CONFIG" \
  --output "$RUN/audit/dataset_audit.json"

python tools/check_rtf_t6_complexity.py \
  --config "$CONFIG" \
  --pretrained "$PRETRAINED" \
  --output "$RUN/audit/complexity.json"
```

Required signals:

```text
all pytest tests passed
DATA_AUDIT_PASS
COMPLEXITY_GATE_PASS
target_coverage = 1.0
candidate_parameters = 5710131
candidate_parameters < baseline_parameters
candidate_total_estimated_ops_per_frame < baseline_conv_macs_per_frame
```

The data audit also requires exact filename pairing, equal image dimensions,
non-identical sampled blur/GT, no train/val path overlap and no logical frame-ID
overlap. If it fails, stop. Report the missing root or exact mismatch and set
`HUMAN_ACTION_REQUIRED: YES`; do not silently change the dataset split.

## Step 5: record a small official-baseline evaluation

This is a two-sequence-per-domain pre-training anchor, not the final benchmark:

```bash
python tools/eval_rtf_t6.py \
  --architecture rtfocuser_baseline \
  --config "$CONFIG" \
  --checkpoint "$PRETRAINED" \
  --output "$RUN/audit/rtfocuser_baseline_2seq.json" \
  --max-sequences-per-domain 2 \
  --window 6 \
  --temporal-overlap 4 \
  --tile-size 384 \
  --tile-overlap 48 \
  --device cuda:0 \
  --amp
```

Every domain must report finite PSNR/SSIM and `input_psnr`. An output below the
input baseline is a pipeline warning that must be investigated before training.

## Step 6: four-iteration training smoke test

```bash
ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" ENV_NAME="$ENV_NAME" \
PRETRAINED="$PRETRAINED" GPU="$GPU" \
bash "$CODE/scripts/run_rtf_t6_training.sh" smoke 2>&1 | \
tee "$RUN/audit/smoke.log"
```

PASS requires:

```text
RTF_T6_SMOKE_PASS
finite total/pixel/fft/edge/temporal losses
best_balanced_psnr.pth exists
latest.pth exists
```

## Step 7: launch formal 180k-iteration training

Keep CPU load bounded at the committed four data workers. Do not increase it.

```bash
mkdir -p "$RUN"
nohup env \
  ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" ENV_NAME="$ENV_NAME" \
  PRETRAINED="$PRETRAINED" GPU="$GPU" \
  bash "$CODE/scripts/run_rtf_t6_training.sh" formal \
  > "$RUN/launcher.log" 2>&1 &

TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
sleep 5
ps -fp "$TRAIN_PID"
tail -80 "$RUN/launcher.log"
```

Monitor through at least iteration 500. Confirm loss is finite, the selected
GPU is active, only four loader workers are used, and checkpoints/logs appear
under:

```text
$RUN/checkpoints/latest.pth
$RUN/checkpoints/best_balanced_psnr.pth
$RUN/train_metrics.jsonl
```

Do not restart from zero after interruption. Resume with:

```bash
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE" \
python "$CODE/tools/train_rtf_t6.py" \
  --config "$CONFIG" \
  --resume "$RUN/checkpoints/latest.pth"
```

## Step 8: final full-resolution evaluation after training

```bash
python tools/eval_rtf_t6.py \
  --architecture t6 \
  --config "$CONFIG" \
  --checkpoint "$RUN/checkpoints/best_balanced_psnr.pth" \
  --output "$RUN/eval_best_full.json" \
  --window 6 \
  --temporal-overlap 4 \
  --tile-size 384 \
  --tile-overlap 48 \
  --device cuda:0 \
  --amp
```

Report GoPro/BSD/DVD separately plus the balanced mean. Do not claim success
from training loss alone. Compare PSNR, SSIM, input PSNR and
`temporal_residual_l1` against the official RT-Focuser anchor and inspect the
business-video output for flicker, trails and six-frame seams.

## Required final report

Return all of the following:

```text
STATUS
GITHUB_BRANCH
GIT_COMMIT
WORKTREE_CLEAN
ENV_NAME / PYTHON / TORCH / CUDA
GPU
PRETRAINED_PATH / SHA256 / TARGET_COVERAGE
RESOLVED_GOPRO_ROOT / TRAIN+VAL SEQUENCES+FRAMES
RESOLVED_BSD_ROOT / TRAIN+VAL SEQUENCES+FRAMES
RESOLVED_DVD_ROOT / TRAIN+VAL SEQUENCES+FRAMES
UNIT_TEST_RESULT
DATA_AUDIT_RESULT
PARAMETER_GATE_RESULT
COMPUTE_GATE_RESULT
BASELINE_2SEQ_METRICS_BY_DOMAIN
SMOKE_RESULT
FORMAL_TRAIN_PID
CURRENT_ITERATION / LOSS / LR
LATEST_CHECKPOINT
BEST_CHECKPOINT
HUMAN_ACTION_REQUIRED: YES or NO
```
