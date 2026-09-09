# CodeAgent task: restart RT-Focuser T6 with native full-frame training

## Goal and non-negotiable constraints

Stop the currently running 256x256 random-crop experiment after verifying that
the target processes belong to this task. Preserve all old logs and checkpoints.
Then run the prepared full-frame code from GitHub.

- Baseline/checkpoint: official RT-Focuser Standard GoPro checkpoint.
- Temporal input/output during training: exactly T=6 in and T=6 out; supervise all six frames.
- Spatial input: every sampled frame at its native full HxW resolution.
- No random crop, center crop, resize, thumbnail, or spatial tiling in training.
- A training sample is six consecutive native-resolution frames, not an entire
  video loaded at once.
- Training data: GoPro + BSD + DVD, balanced 1:1:1.
- Batch size: 1. Gradient accumulation: 2. DataLoader workers: 4.
- Activation checkpointing is enabled only to reduce training memory. It must
  not change inference parameters or arithmetic.
- Candidate parameters and per-output-frame arithmetic must remain below the
  original RT-Focuser Standard model.
- Do not write, rewrite, or patch source code on the server. If a source-code
  problem is found, stop and report it so the code can be synchronized through
  GitHub. Keep `git status --short` empty.
- Do not kill any process unless its full command line proves it is the old
  RT-Focuser crop-training task owned by the current Unix user.
- Never fall back to a crop or resize after OOM.

Repository:

```text
https://github.com/hihiok/video_motion_deblur.git
branch: agent/rtfocuser-shift-dst-t6-fullframe-v2
minimum code commit: 9d0e3e673d694535a97ea3850c90033b684cca52
```

## Proxy and SSL setup

Run the following before GitHub or pip/conda network access:

```bash
export http_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export HTTPS_PROXY="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"

git config --global http.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy https://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080
git config --global http.proxy http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080

export GIT_SSL_NO_VERIFY=true
export CONDA_SSL_VERIFY=false
export PYTHONHTTPSVERIFY=0
git config --global http.sslVerify false
conda config --set ssl_verify false || true
```

Do not echo proxy environment variables into logs or the final report.

## Fixed paths

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
OLD_CONFIG=$CODE/configs/rtf_t6_gopro_bsd_dvd.yaml
OLD_RUN=$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_v1
CONFIG=$CODE/configs/rtf_t6_gopro_bsd_dvd_fullframe.yaml
RUN=$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_fullframe_v2
BRANCH=agent/rtfocuser-shift-dst-t6-fullframe-v2
MIN_COMMIT=9d0e3e673d694535a97ea3850c90033b684cca52
```

Do not delete or reuse `$OLD_RUN`. The new full-frame run has a different output
directory so that the two protocols cannot be mixed.

## Step 1: inspect and stop only the old crop-training process

First record the old checkpoint and process state:

```bash
set -euo pipefail

ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
OLD_CONFIG=$CODE/configs/rtf_t6_gopro_bsd_dvd.yaml
OLD_RUN=$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_v1

mkdir -p "$OLD_RUN/audit"
date -Is | tee "$OLD_RUN/audit/stopped_for_fullframe_v2.txt"
if [[ -s "$OLD_RUN/checkpoints/latest.pth" ]]; then
  sha256sum "$OLD_RUN/checkpoints/latest.pth" | \
    tee -a "$OLD_RUN/audit/stopped_for_fullframe_v2.txt"
fi

pgrep -u "$(id -u)" -af 'tools/train_rtf_t6.py|run_rtf_t6_training.sh' | \
  tee -a "$OLD_RUN/audit/stopped_for_fullframe_v2.txt" || true
```

Inspect the printed command lines. Terminate only current-user processes whose
arguments reference `tools/train_rtf_t6.py` with the old
`rtf_t6_gopro_bsd_dvd.yaml`, plus their verified launcher parent if it is
`run_rtf_t6_training.sh formal`. Previous reported PIDs were 17722 and 18353,
but do not trust stale PIDs without checking them again.

Use SIGTERM first:

```bash
# Replace with only the re-verified old-task PIDs. Do not paste stale PIDs blindly.
OLD_TASK_PIDS="<verified pid list>"
ps -o user=,pid=,ppid=,etime=,cmd= -p ${OLD_TASK_PIDS// /,}
kill -TERM $OLD_TASK_PIDS

for _ in $(seq 1 30); do
  alive=0
  for pid in $OLD_TASK_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then alive=1; fi
  done
  [[ "$alive" -eq 0 ]] && break
  sleep 1
done

for pid in $OLD_TASK_PIDS; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "Old task did not stop after SIGTERM: $pid"
    exit 3
  fi
done

echo "OLD_CROP_TRAINING_STOPPED $(date -Is)" | \
  tee -a "$OLD_RUN/audit/stopped_for_fullframe_v2.txt"
```

Do not use broad `pkill`, do not kill another user's process, and do not remove
the old run directory.

## Step 2: fetch the exact full-frame branch

```bash
set -euo pipefail

ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
BRANCH=agent/rtfocuser-shift-dst-t6-fullframe-v2
MIN_COMMIT=9d0e3e673d694535a97ea3850c90033b684cca52

if [[ ! -d "$CODE/.git" ]]; then
  git clone https://github.com/hihiok/video_motion_deblur.git "$CODE"
fi
cd "$CODE"
git fetch origin "$BRANCH"
git switch --detach "origin/$BRANCH"
git merge-base --is-ancestor "$MIN_COMMIT" HEAD
test -z "$(git status --short)"
git rev-parse HEAD
```

PASS requires the minimum commit or a descendant and an empty worktree.

## Step 3: activate the existing CUDA environment

```bash
source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx deblur_runtime; then
  ENV_NAME=deblur_runtime
elif conda env list | awk '{print $1}' | grep -qx RVRT; then
  ENV_NAME=RVRT
else
  echo "HUMAN_ACTION_REQUIRED: YES — no deblur_runtime or RVRT environment"
  exit 2
fi

conda activate "$ENV_NAME"
export PYTHONUNBUFFERED=1
python - <<'PY'
import torch
print("python/torch/cuda:", torch.__version__, torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY

python -m pip install -r "$CODE/requirements_rtf_t6.txt"
```

Do not replace or downgrade the installed CUDA PyTorch build.

## Step 4: locate and verify the official RT-Focuser checkpoint

```bash
for p in \
  "$ROOT/envs/RT-Focuser/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth" \
  "$ROOT/envs/rt_focuser_repo/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth" \
  "$ROOT/benchmark/weights/rt_focuser/GoPro_RT_Focuser_Standard_256.pth"; do
  if [[ -s "$p" ]]; then PRETRAINED="$p"; break; fi
done

test -s "${PRETRAINED:-}"
sha256sum "$PRETRAINED"
```

Required SHA256:

```text
6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb
```

Stop on a different hash. Do not substitute another RT-Focuser variant.

## Step 5: prove that the config cannot crop or resize

```bash
CONFIG=$CODE/configs/rtf_t6_gopro_bsd_dvd_fullframe.yaml
RUN=$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_fullframe_v2

python - <<'PY'
import yaml
from pathlib import Path

p = Path("/mnt/ssd1/z00919662/motion_deblur/benchmark_code/configs/rtf_t6_gopro_bsd_dvd_fullframe.yaml")
c = yaml.safe_load(p.read_text())
assert c["train"]["clip_length"] == 6
assert c["train"]["spatial_mode"] == "full_frame"
assert c["train"]["crop_size"] == 0
assert c["validation"]["crop_size"] == 0
assert c["train"]["batch_size"] == 1
assert c["train"]["gradient_accumulation"] == 2
assert c["train"]["workers"] == 4
assert c["model"]["activation_checkpointing"] is True
print("FULLFRAME_CONFIG_PASS")
PY
```

The training code also rejects `spatial_mode=full_frame` if either crop size is
nonzero or if batch size is not one.

## Step 6: choose a genuinely free GPU and run static gates

```bash
nvidia-smi
# Select a free GPU. Do not kill another user's process.
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

Required results:

```text
12 tests passed
DATA_AUDIT_PASS
COMPLEXITY_GATE_PASS
target_coverage = 1.0
candidate_parameters = 5710131 < baseline_parameters = 5864531
candidate_total_estimated_ops_per_frame = 948874368
baseline_conv_macs_per_frame = 962155648
```

Activation checkpointing is training-only and therefore must not alter the
parameter or inference-compute gate.

## Step 7: mandatory native full-frame CUDA preflight

This step performs one complete forward, loss, backward, gradient check, and
optimizer update for the largest-resolution sampled training sequence in each
of GoPro, BSD, and DVD. It enables the temporal losses and records peak GPU
memory. It does not save or reuse the temporary weights.

```bash
ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" ENV_NAME="$ENV_NAME" \
PRETRAINED="$PRETRAINED" GPU="$GPU" RUN="$RUN" \
bash "$CODE/scripts/run_rtf_t6_fullframe_training.sh" preflight 2>&1 | \
  tee "$RUN/audit/fullframe_preflight.log"
```

PASS requires all of the following:

- `FULLFRAME_PREFLIGHT_PASS`.
- Three domains are present.
- Every domain reports a native shape `[6, 3, H, W]` and identical output shape.
- `crop_applied=false` and `resize_applied=false`.
- Finite total, pixel, FFT, edge, temporal, and acceleration losses.
- Finite gradients and one successful optimizer step for each domain.
- Peak allocated/reserved GPU memory and elapsed time are recorded.

If this step reports `FULLFRAME_PREFLIGHT_OOM`, stop. Do not lower resolution,
change T, introduce crops, or edit code locally. Return the report with
`HUMAN_ACTION_REQUIRED: YES` so the GitHub code can be revised centrally.

## Step 8: optional baseline anchor using the same native full frames

```bash
python tools/eval_rtf_t6.py \
  --architecture rtfocuser_baseline \
  --config "$CONFIG" \
  --checkpoint "$PRETRAINED" \
  --output "$RUN/audit/rtfocuser_baseline_2seq_fullframe.json" \
  --max-sequences-per-domain 2 \
  --window 6 \
  --temporal-overlap 4 \
  --tile-size 0 \
  --tile-overlap 0 \
  --device cuda:0 \
  --amp
```

Do not explain an output below input PSNR as domain mismatch without checking
pairing, range, RGB order, and visual output first.

## Step 9: launch the new formal training from the official checkpoint

Do not resume from the old random-crop checkpoint. Start this new protocol from
the official RT-Focuser checkpoint.

```bash
mkdir -p "$RUN"
nohup env \
  PYTHONUNBUFFERED=1 \
  ROOT="$ROOT" CODE="$CODE" CONFIG="$CONFIG" ENV_NAME="$ENV_NAME" \
  PRETRAINED="$PRETRAINED" GPU="$GPU" RUN="$RUN" \
  bash "$CODE/scripts/run_rtf_t6_fullframe_training.sh" formal \
  > "$RUN/launcher.log" 2>&1 &

TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
sleep 10
ps -fp "$TRAIN_PID"
tail -100 "$RUN/launcher.log"
```

The first training batch must emit an event like:

```json
{"event":"full_frame_first_batch","shape":[1,6,3,H,W],"crop_applied":false,"resize_applied":false}
```

Confirm GPU activity and monitor until at least iteration 100. Because native
720p has about 14.06 times as many pixels as a 256x256 crop, training will be
much slower than the old run. Report the measured seconds/iteration and
estimated time to 180k; do not silently change the iteration count.

Expected outputs:

```text
$RUN/checkpoints/latest.pth
$RUN/checkpoints/best_balanced_psnr.pth
$RUN/train_metrics.jsonl
$RUN/resolved_config.yaml
```

Resume only this full-frame run after interruption:

```bash
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE" \
python "$CODE/tools/train_rtf_t6.py" \
  --config "$CONFIG" \
  --resume "$RUN/checkpoints/latest.pth"
```

Before resuming, verify that the checkpoint's saved config has
`spatial_mode=full_frame` and both crop sizes equal zero.

## Step 10: final evaluation after training completes

Evaluate the official baseline and candidate with the identical native
full-frame protocol. Use `tile-size=0`; do not switch to tiled inference for one
model only.

```bash
python tools/eval_rtf_t6.py \
  --architecture t6 \
  --config "$CONFIG" \
  --checkpoint "$RUN/checkpoints/best_balanced_psnr.pth" \
  --output "$RUN/eval_best_native_fullframe.json" \
  --window 6 \
  --temporal-overlap 4 \
  --tile-size 0 \
  --tile-overlap 0 \
  --device cuda:0 \
  --amp
```

Report GoPro, BSD, and DVD separately plus their balanced mean. Compare PSNR,
SSIM, input PSNR, and temporal residual L1 against the official RT-Focuser
anchor. Inspect business-video output for flicker, trails, double edges, face
deformation, and a six-frame periodic pulse. Do not claim success from training
loss alone.

## Required immediate report

Return all fields below after the new run reaches iteration 100, or immediately
when a gate fails:

```text
STATUS
GITHUB_BRANCH
GIT_COMMIT
WORKTREE_CLEAN
OLD_CROP_TRAINING_STOPPED / OLD_LAST_ITERATION / OLD_CHECKPOINT_SHA256
ENV_NAME / PYTHON / TORCH / CUDA
GPU / GPU_MODEL
PRETRAINED_PATH / SHA256 / TARGET_COVERAGE
RESOLVED_GOPRO_ROOT / TRAIN+VAL SEQUENCES+FRAMES
RESOLVED_BSD_ROOT / TRAIN+VAL SEQUENCES+FRAMES
RESOLVED_DVD_ROOT / TRAIN+VAL SEQUENCES+FRAMES
UNIT_TEST_RESULT
DATA_AUDIT_RESULT
PARAMETER_GATE_RESULT
COMPUTE_GATE_RESULT
FULLFRAME_CONFIG_RESULT
FULLFRAME_PREFLIGHT_RESULT_BY_DOMAIN
NATIVE_SHAPES_BY_DOMAIN
PEAK_GPU_MEMORY_BY_DOMAIN
SECONDS_PER_STEP_BY_DOMAIN
BASELINE_2SEQ_FULLFRAME_METRICS_BY_DOMAIN
FORMAL_TRAIN_PID
FIRST_FORMAL_BATCH_SHAPE / CROP_APPLIED / RESIZE_APPLIED
CURRENT_ITERATION / LOSS_COMPONENTS / LR / SECONDS_PER_ITERATION
ESTIMATED_180K_COMPLETION_TIME
LATEST_CHECKPOINT
BEST_CHECKPOINT
SOURCE_CODE_MODIFIED: NO
HUMAN_ACTION_REQUIRED: YES or NO
```

If CodeAgent changes any source file, do not continue training from that
uncommitted version. Return the exact diff and commit or otherwise synchronize
the changes back before proceeding.
