# WaveShift T6 AMP recovery and continuation

## 1. Task and authority

Resume the interrupted WaveShift run using the committed AMP fix. This document
supersedes the old immediate stop on an AMP gradient overflow and the old Step F
resume command in `CODEAGENT_NANOVNR_WAVESHIFT_PAGF_FULLFRAME_20260904.md`.
CodeAgent executes committed code only; do not edit, patch, or commit source.
Do not ask again before the bounded recovery or continuation authorized here.

Repository: https://github.com/hihiok/video_motion_deblur.git

Branch: `agent/nanovnr-waveshift-pagf-t6-fullframe-20260907`

Required fix and tests commit: `422bf5548ba05a6d8c7277980e4bed316c6a1895`
(the final documentation HEAD must contain this commit).

Preserve the original run and every checkpoint/log/report. Leave the unrelated
NAFNet baseline on physical GPU 6 running and untouched. Use a separate checkout
and a separate recovery output directory below, so old `latest.pth`, `train.log`,
and `FINAL_REPORT.txt` cannot be overwritten by these commands.

## 2. Exact scope of the fix

- The model, parameters, GSTS/PAGF/edge-aware/RepConv, T=6, batch=1, native full
  frames, family-balanced sampling, Charbonnier-only, Adam, LR, and clip=0.5
  are unchanged. Do not crop, resize, offload, switch precision, or lower LR.
- Initial scaler: 4096, growth interval: 10000 successful updates. A legacy
  checkpoint without scaler state logs `LEGACY_CHECKPOINT_SCALER_INITIALIZED`.
  New checkpoints save and restore scaler state and overflow counters.
- Check loss finiteness, backward, unscale, inspect gradient elements BEFORE
  clipping. For nonfinite gradient elements under AMP, GradScaler skips the
  optimizer update and reduces scale. Recompute the SAME loaded batch.
- At most eight retries after the initial attempt (nine attempts total).
  A failed attempt changes neither optimizer state nor scheduler/step count.
  Only a successful update advances the cosine scheduler and training step.
- Nonfinite loss, nonfinite gradient without AMP, aggregate norm overflow with
  finite elements, or exhausted recovery remain fatal. Do not mask gradients
  with `nan_to_num`, drop the batch, or continue using invalid weights.
- Events include source, sequence, tensor shape, step, affected parameters,
  and scale before/after, in `amp_events.jsonl` and the training log. Fatal
  runtime errors write a `failure_step_*.json`; no failed checkpoint is saved.
- This addresses recovery behavior, not a proven numerical root cause. A
  stable loss before step 5843 does not prove why backward overflowed.
- Model/Adam/scheduler resume from step 5000; data sampling is reseeded. The
  old checkpoint has no sampler/RNG position, so this is NOT bitwise-exact
  replay of the old step-5843 sample. Crossing step 5843 is a stability check.

## 3. Environment, proxy, Git, and preserved artifacts

Use the existing working RVRT environment. Do not upgrade/downgrade PyTorch or
the NVIDIA driver. Load the server's existing proxy configuration without
printing or publishing proxy credentials. SSL configuration below is the
previously requested server-specific setting.

Run the following blocks in the same Bash session; use `pipefail` so `tee`
cannot hide Python failures.

```bash
set -e -o pipefail
source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate RVRT
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
git config --global http.sslVerify false

ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_waveshift_amp_fix_20260907
OLD_RUN=$ROOT/runs/nanovnr_waveshift_pagf_fullframe_t6_bsd3ms24ms_20260907
RUN=$OLD_RUN/amp_recovery_20260907
BRANCH=agent/nanovnr-waveshift-pagf-t6-fullframe-20260907
REQUIRED_FIX_COMMIT=422bf5548ba05a6d8c7277980e4bed316c6a1895
GOPRO=$ROOT/datasets/GoPro
DVD=$ROOT/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms
INPUT=$ROOT/input/xiaobieli38_trimmed.mp4
RESUME_5000=$OLD_RUN/train/step_0005000.pth

test -s "$RESUME_5000"
test -s "$OLD_RUN/train/latest.pth"
test -s "$OLD_RUN/train/train.log"
test -s "$OLD_RUN/FINAL_REPORT.txt"
if [ -d "$RUN" ]; then
  echo 'HUMAN_ACTION_REQUIRED: YES; RECOVERY_DIRECTORY_ALREADY_EXISTS'
  exit 2
fi
mkdir -p "$RUN/verify" "$RUN/preflight" "$RUN/train"
sha256sum "$RESUME_5000" "$OLD_RUN/train/latest.pth" \
  "$OLD_RUN/train/train.log" "$OLD_RUN/FINAL_REPORT.txt" \
  > "$RUN/verify/original_artifacts.sha256"

if [ ! -e "$REPO" ]; then
  git clone --single-branch --branch "$BRANCH" \
    https://github.com/hihiok/video_motion_deblur.git "$REPO"
else
  test -z "$(git -C "$REPO" status --porcelain)"
  test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
  git -C "$REPO" fetch origin "$BRANCH"
  git -C "$REPO" pull --ff-only origin "$BRANCH"
fi
cd "$REPO"
git merge-base --is-ancestor "$REQUIRED_FIX_COMMIT" HEAD
git merge-base --is-ancestor 997c96092eb6dd447bd88da51dc12b81c37b6972 HEAD
git rev-parse HEAD | tee "$RUN/verify/github_head.txt"
test -z "$(git status --porcelain)"
```

Verify the exact final HEAD supplied with the user's instruction, not only
ancestry. If a later unexpected source commit exists, stop and report it.
Do not use `reset --hard`, change the baseline checkout, delete artifacts, or
kill unrelated processes. If this recovery directory already exists, report
its state instead of rerunning these blocks or replacing files.

## 4. Verify code and select a free GPU

```bash
cd "$REPO/nanovsr_deblur"
python -m py_compile amp_training.py train_nanovnr_waveshift_pagf_fullframe.py \
  tests/test_waveshift_amp.py
OMP_NUM_THREADS=1 python -m unittest discover -s tests \
  -p 'test_*waveshift*.py' -v 2>&1 | tee "$RUN/verify/tests_cpu.log"
python audit_nanovnr_waveshift_pagf.py --device cpu \
  2>&1 | tee "$RUN/verify/architecture.log"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free \
  --format=csv | tee "$RUN/verify/gpus.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv | tee "$RUN/verify/gpu_processes.txt"
```

Require 14/14 CPU tests and architecture audit PASS. Select a GPU with no
compute process and greatest free memory, excluding physical GPU 6 even if
it later becomes idle. Export its observed UUID as `CUDA_VISIBLE_DEVICES`:

```bash
export CUDA_VISIBLE_DEVICES="$(python - <<'PY'
import csv, subprocess
def query(kind, fields):
    return subprocess.check_output(
        ['nvidia-smi', '--query-' + kind + '=' + fields,
         '--format=csv,noheader,nounits'], text=True)
busy = {s.strip() for s in query('compute-apps', 'gpu_uuid').splitlines() if s.strip()}
rows = list(csv.reader(query('gpu', 'index,uuid,memory.free').splitlines()))
candidates = [(int(free.strip()), uuid.strip()) for index, uuid, free in rows
              if index.strip() != '6' and uuid.strip() not in busy]
if not candidates:
    raise SystemExit('HUMAN_ACTION_REQUIRED: YES; NO_FREE_GPU')
print(max(candidates)[1])
PY
)"
test -n "$CUDA_VISIBLE_DEVICES"
WAVESHIFT_TEST_DEVICE=cuda OMP_NUM_THREADS=1 python -m unittest discover \
  -s tests -p 'test_waveshift_amp.py' -v \
  2>&1 | tee "$RUN/verify/tests_cuda.log"
```

Require 7/7 on real CUDA; no CPU fallback. These tests intentionally inject
nonfinite gradients to verify skip/retry/exhaustion behavior using GradScaler,
and run the actual T6 WaveShift model with FP16 and video checkpoint backward.
Their injected events are test evidence, not overflow counts from training.

Reuse existing successful data audit/profile logs after verifying the model
and dataset code are unchanged from `997c960`. Keep BSD strictly under
`BSD_3ms24ms/train` for training and `BSD_3ms24ms/test` for evaluation/audit.
Preserve the original audited family/window/native-resolution counts. Never
read `/BSD/train`, `/BSD/test`, or a different exposure configuration.

## 5. Native T6 preflight with the fixed update function

Run on the chosen GPU using the new code. This makes real training updates
on disposable random models, one per actual family/resolution pair; it does
not alter the resume checkpoint. Do not infer feasibility from small patches.

```bash
TRAIN_ARGS=(
  --variant waveshift_edge
  --gopro-root "$GOPRO" --dvd-root "$DVD" --bsd-root "$BSD"
  --num-frames 6 --total-iterations 150000 --workers 2
  --lr 3e-4 --eta-min 1e-7 --save-every 5000
  --amp --grad-checkpoint
)
python train_nanovnr_waveshift_pagf_fullframe.py "${TRAIN_ARGS[@]}" \
  --output-dir "$RUN/preflight" --preflight-only \
  2>&1 | tee "$RUN/preflight/preflight.log"
```

Require `PREFLIGHT_STATUS=PASS` for all actual pairs, including DVD 1080p.
Any CUDA failure stops this task. The known driver/NVML secondary assertion
may still mask OOM; do not label it an AMP recovery or silently retry it.

## 6. Resume 5000 -> 6000 as a bounded stability gate

```bash
python train_nanovnr_waveshift_pagf_fullframe.py "${TRAIN_ARGS[@]}" \
  --output-dir "$RUN/train" --resume "$RESUME_5000" --stop-after-step 6000 \
  2>&1 | tee "$RUN/train/resume_5000_to_6000.log"

export WAVESHIFT_VERIFY_CHECKPOINT="$RUN/train/step_0006000.pth"
python - <<'PY'
import os, torch
c = torch.load(os.environ['WAVESHIFT_VERIFY_CHECKPOINT'], map_location='cpu')
assert c['step'] == 6000
assert c['scheduler']['last_epoch'] == 6000
assert c['scheduler']['T_max'] == 150000
assert c['amp_policy'] == 'same_batch_retry_v1'
assert c['scaler']['scale'] > 0
assert c['scaler']['growth_interval'] == 10000
assert all(torch.isfinite(x).all().item() for x in c['model'].values())
for state in c['optimizer']['state'].values():
    for value in state.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all().item()
print('RESUME_6000_GATE=PASS', c['amp_stats'], 'scale', c['scaler']['scale'])
PY
sha256sum -c "$RUN/verify/original_artifacts.sha256"
```

The startup must show step 5000 restored, legacy scaler initialized to 4096,
and no optimizer/scheduler reset. At 6000 require successful exit, finite model
and optimizer states, persisted scaler, and unchanged original file hashes.
Report overflow/recovered counts and scales. Do not treat any recovered event
alone as failure. If the bounded policy exhausts, stop and return the JSON
diagnostic plus event log. Do not restart with a different limit or lower LR.

## 7. Automatically continue 6000 -> 150000

When the gate passes, continue without another permission request:

```bash
python train_nanovnr_waveshift_pagf_fullframe.py "${TRAIN_ARGS[@]}" \
  --output-dir "$RUN/train" --resume "$RUN/train/step_0006000.pth" \
  2>&1 | tee "$RUN/train/resume_6000_to_150000.log"
```

Require `CHECKPOINT_SCALER_RESTORED`, step 6000, original Adam and the single
150000-step cosine horizon. Save each 5000 successful updates and at 150000.
Monitor logs; do not count failed attempts toward the 150000 updates.

Execute the original document's Steps G-J and its final quality/report protocol
using this recovery `$RUN` and `$REPO`. Evaluate 50k/75k/100k/125k/150k using
the identical GoPro first-100 T6 RGB FP16 protocol, select BEST_T6, then matched
center T4/T5/T6, comparable baseline if available, native business inference,
video audit, and user visual review. Reuse original profile evidence because
no architecture code changed. Do not claim the loss fix proves better PSNR.
Do not compare the old T15 reference against T6 as a controlled improvement.

## 8. Stop rules and final report additions

Stop on code/environment/audit failure, occupied/no usable GPU, wrong BSD
paths or model/recipe, OOM, nonfinite loss, non-AMP invalid gradients, norm
failure, exhausted AMP retries, corrupt checkpoint, or invalid video outputs.
Preserve evidence and report `HUMAN_ACTION_REQUIRED: YES` and exact blocker.
The former rule stopping on the FIRST AMP gradient overflow is superseded
only by the bounded policy above; there is no blanket permission to ignore NaN.

Write a NEW `FINAL_REPORT.txt` in `$RUN`, preserving the original one. Include
all original required fields and add:

```text
AMP_POLICY: same_batch_retry_v1
AMP_FIX_COMMIT_PRESENT: YES / NO
SOURCE_CODE_MODIFIED_BY_CODEAGENT: NO
ORIGINAL_ARTIFACT_HASHES_UNCHANGED: YES / NO
CPU_TESTS: 14/14 PASS / FAIL
CUDA_AMP_TESTS: 7/7 PASS / FAIL
RESUME_SOURCE: <original step_0005000.pth>
RESUME_DATA_ORDER: RESEEDED_NOT_EXACT_SAMPLER_REPLAY
LEGACY_SCALER_INITIAL_SCALE: 4096
SCALER_GROWTH_INTERVAL: 10000
RESUME_6000_GATE: PASS / FAIL
SCALER_RESTORED_AT_6000: YES / NO
AMP_OVERFLOW_ATTEMPTS: <count>
AMP_RECOVERED_BATCHES: <count>
AMP_FINAL_SCALE: <value>
TRAIN_SUCCESSFUL_UPDATES: <checkpoint step>
AMP_EVENT_LOG: <path, or NONE if no events>
FAILURE_DIAGNOSTIC: <path or NONE>
GPU6_BASELINE_UNTOUCHED: YES / NO
EFFECT_NUMERICAL_STATUS: VERIFIED / UNVERIFIED
SUBJECTIVE_QUALITY_STATUS: PENDING_USER_REVIEW / <user assessment>
```

Do not report training completion before 150000 successful updates or report
quality PASS without the requested evaluation and user review.
