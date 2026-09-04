# CODEAGENT TASK — NanoVNR WaveShift-PAGF native-full-frame training

## 0. Goal and non-negotiable rule

Train and evaluate the already implemented `NanoVNRWaveShiftPAGF` model. The
goal is not merely to finish training: it must be compared with the existing
RGB full-frame NAFNet baseline under the same evaluation protocol, and any
quality regression or protocol mismatch must be reported honestly.

Do not write, redesign, or modify code in this task. Do not change model width,
loss, temporal lengths, dataset policy, or resolution to make a failed run pass.

## 1. Repository

Repository:

`https://github.com/hihiok/video_motion_deblur.git`

Branch:

`agent/nanovnr-waveshift-pagf-fullframe-20260904`

Required code commit (the checked-out HEAD may be a later documentation-only
commit, but this commit must be its ancestor):

`254af1898fa8374a1d7de27e1c7d9abb5cb28d3e`

Primary files:

- `nanovsr_deblur/models/network_nanovnr_waveshift_pagf.py`
- `nanovsr_deblur/train_nanovnr_waveshift_pagf_fullframe.py`
- `nanovsr_deblur/audit_nanovnr_waveshift_pagf.py`
- `nanovsr_deblur/profile_nanovnr_waveshift_pagf.py`
- `nanovsr_deblur/eval_gopro_nanovnr_waveshift_pagf.py`
- `nanovsr_deblur/eval_gopro_context_matched.py`
- `nanovsr_deblur/infer_video_nanovnr_waveshift_pagf.py`
- `nanovsr_deblur/audit_video_output.py`
- `nanovsr_deblur/tests/test_nanovnr_waveshift_pagf.py`

## 2. Network that must be used

Primary variant:

`waveshift_edge`

Architecture:

`NanoVNRWaveShiftPAGF`

Required data flow:

```text
RGB video
  -> 3x3 stem, 3 -> 12 channels
  -> fixed Haar DWT at native full-frame feature resolution
     -> LL: 12 channels at H/2 x W/2
        -> GSTS radius 2, cardinal spatial candidates
        -> GSTS radius 4, diagonal spatial candidates
        -> PAGF-gated forward and backward additive recurrence
        -> independent forward/backward NAF U-Nets
        -> pixel-wise bidirectional/current fusion
     -> HF: 36 channels at H/2 x W/2
        -> subband-preserving lightweight processing
        -> learnable Laplacian-initialized edge-aware residual
  -> inverse Haar IWT
  -> reparameterizable RGB reconstruction
  -> input RGB + learned residual
```

Hard architecture assertions:

- RGB input, `feat_extract: 3 -> 12`.
- Haar LL/HF decomposition is fixed and exactly invertible.
- Recurrent computation is at half spatial resolution.
- GSTS is used on LL only, never HF or full-resolution features.
- Two GSTS blocks with half-resolution spatial radii `[2, 4]`.
- GSTS temporal shift itself has zero parameters/MACs; its fusion convolutions
  still count toward parameters/MACs.
- PAGF performs pixel-wise current/history selection.
- Forward and backward propagation parameters are independent.
- NAF propagation channels remain `[24, 32, 48, 72]`.
- Recurrence uses a direct additive state update.
- HF edge-aware convolution starts from a Laplacian kernel and has a learned
  zero-initialized residual scale to reduce early ringing risk.
- Train-time RepConv branches must fuse to ordinary 3x3 convolution for deploy.
- Reparameterization max absolute difference must be below `2e-5`.
- `prev_forward_feat` / `next_forward_feat` carries the half-resolution LL state.
- Chunk inference must use GSTS halo frames without applying recurrence twice to
  the halo. Only the non-overlapping core frames update the carried state.

Do not add optical flow, deformable convolution, attention, transformer blocks,
foreground masks, background-preservation losses, perceptual loss, temporal
loss, SSIM loss, edge loss, frequency loss, GAN loss, or distillation.

The repository also exposes `haar_pagf` and `waveshift` as controlled ablations.
Do not train them in this task unless the user explicitly starts a follow-up
ablation task.

## 3. Dataset and resolution policy

```bash
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms
```

BSD hard policy:

- The BSD root must be exactly
  `/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms`.
- Training may read only
  `/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms/train`.
- Audit/evaluation may read only
  `/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms/test`.
- Do not discover or sample any sibling BSD exposure/configuration directory.
- Do not copy, rename, relabel, regenerate, or modify dataset files.

GoPro/DVD/BSD training uses family-balanced sampling, approximately `1:1:1`.

Resolution policy:

- native full frame;
- no crop or patch training;
- no resize;
- no forced 720p or 1080p;
- no aspect-ratio conversion;
- batch size 1.

If a sample is 1280x720, train on the full 1280x720 frame. If it is 1920x1080,
train on full 1920x1080. If BSD is 640x480, train on full 640x480.

## 4. Training recipe

Phase 1:

- steps 1-50000;
- T=7.

Phase 2:

- steps 50001-150000;
- T=30.

The only change at step 50001 is T=7 -> T=30. Do not reset optimizer or
scheduler.

- Loss: Charbonnier only.
- Optimizer: Adam, betas `(0.9, 0.99)`.
- LR: `3e-4 -> 1e-7` with one cosine schedule over 150000 steps.
- Gradient clipping: 0.5.
- AMP enabled.
- Gradient checkpointing enabled.
- Workers <= 2.
- Random initialization.
- Save every 5000 steps.

## 5. Step A — proxy, SSL, environment, and safe Git sync

The server is behind an HTTPS-inspection proxy. Before Git/network operations,
disable Git SSL verification as required on this server and load the existing
credential-bearing proxy environment. Never print, copy, log, or commit proxy
credentials.

```bash
git config --global http.sslVerify false

source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate RVRT
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true

ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovnr_waveshift_pagf
RUN=$ROOT/runs/nanovnr_waveshift_pagf_fullframe_20260904
BRANCH=agent/nanovnr-waveshift-pagf-fullframe-20260904
EXPECTED_CODE_COMMIT=254af1898fa8374a1d7de27e1c7d9abb5cb28d3e
INPUT=$ROOT/input/xiaobieli38_trimmed.mp4
GOPRO=$ROOT/datasets/GoPro
DVD=$ROOT/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms

mkdir -p "$ROOT" "$RUN"
cd "$ROOT"
if [ ! -d "$REPO/.git" ]; then
  git clone --branch "$BRANCH" --single-branch \
    https://github.com/hihiok/video_motion_deblur.git "$REPO"
else
  cd "$REPO"
  if [ -n "$(git status --porcelain)" ]; then
    echo "HUMAN_ACTION_REQUIRED: YES"
    echo "REASON: EXISTING_REPO_HAS_UNCOMMITTED_CHANGES"
    git status --short
    exit 2
  fi
  git fetch origin "$BRANCH"
  git switch "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

cd "$REPO"
git merge-base --is-ancestor "$EXPECTED_CODE_COMMIT" HEAD || {
  echo "HUMAN_ACTION_REQUIRED: YES"
  echo "REASON: REQUIRED_CODE_COMMIT_MISSING"
  exit 2
}
git status --short --branch
git rev-parse HEAD
```

Do not use `git reset --hard`. Do not modify or commit anything from CodeAgent.

Check runtime:

```bash
cd "$REPO/nanovsr_deblur"
python - <<'PY'
import cv2, numpy, PIL, torch, torchvision
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_devices', torch.cuda.device_count())
PY
```

If imports are missing, install only `nanovsr_deblur/requirements.txt` into the
active environment. Do not upgrade/downgrade PyTorch if CUDA already works. If
the environment cannot run the committed code, stop with
`HUMAN_ACTION_REQUIRED: YES` and the exact missing/incompatible dependency.

## 6. Step B — syntax, unit, architecture, and Git cleanliness checks

```bash
cd "$REPO/nanovsr_deblur"
python -m py_compile \
  models/network_nanovnr_waveshift_pagf.py \
  train_nanovnr_waveshift_pagf_fullframe.py \
  audit_nanovnr_waveshift_pagf.py \
  profile_nanovnr_waveshift_pagf.py \
  eval_gopro_nanovnr_waveshift_pagf.py \
  eval_gopro_context_matched.py \
  infer_video_nanovnr_waveshift_pagf.py \
  audit_video_output.py \
  tests/test_nanovnr_waveshift_pagf.py

python -m unittest discover -s tests -p 'test_nanovnr_waveshift_pagf.py' -v \
  2>&1 | tee "$RUN/unit_tests.log"

python audit_nanovnr_waveshift_pagf.py --device cpu \
  2>&1 | tee "$RUN/architecture_audit_cpu.log"
```

Required:

- all 6 unit tests pass;
- `HAAR_ROUNDTRIP_MAX_ABS_DIFF < 1e-6`;
- deploy output/state max difference `< 2e-5`;
- architecture audit PASS;
- no non-finite gradient.

After tests, verify no source changed:

```bash
cd "$REPO"
git status --porcelain
```

The output must be empty. `__pycache__` may be ignored by Git, but no tracked or
untracked source file may be created/changed. If source changed, stop and report.

## 7. Step C — mandatory dataset audit

```bash
test -d "$GOPRO" || { echo "MISSING=$GOPRO"; exit 2; }
test -d "$DVD"   || { echo "MISSING=$DVD"; exit 2; }
test -d "$BSD/train" || { echo "MISSING=$BSD/train"; exit 2; }
test -d "$BSD/test"  || { echo "MISSING=$BSD/test"; exit 2; }

cd "$REPO/nanovsr_deblur"
python audit_fullframe_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  2>&1 | tee "$RUN/dataset_audit.log"
```

Required output:

```text
FULL_FRAME_AUDIT_STATUS=PASS
RANDOM_CROP=NO
RESIZE=NO
BSD_ROOT_SPLIT_ONLY=YES
BSD_ALLOWED_SPLITS=train,test
BSD_NESTED_CONFIG_SPLITS=FORBIDDEN
```

Report GoPro/DVD/BSD native resolutions and T=7/T=30 window counts. Audit every
printed BSD path. Any BSD training path outside `$BSD/train`, or test path
outside `$BSD/test`, is an immediate stop:

```text
HUMAN_ACTION_REQUIRED: YES
REASON: BSD_3MS24MS_ROOT_OR_SPLIT_POLICY_VIOLATION
BAD_PATH: <path>
```

## 8. Step D — model profiles

Use T=1 for memory-safe per-frame profiling. Profile baseline and improved model
at the same resolutions and report MACs, FLOPs, parameters, and relative change.

```bash
cd "$REPO/nanovsr_deblur"
for HW in "360 640" "720 1280" "1080 1920"; do
  set -- $HW
  H=$1
  W=$2
  python profile_nanovnr_nafnet_rgb.py --height "$H" --width "$W" --frames 1 \
    | tee "$RUN/profile_baseline_${W}x${H}.txt"
  python profile_nanovnr_waveshift_pagf.py \
    --variant waveshift_edge --height "$H" --width "$W" --frames 1 \
    | tee "$RUN/profile_waveshift_train_${W}x${H}.txt"
  python profile_nanovnr_waveshift_pagf.py \
    --variant waveshift_edge --height "$H" --width "$W" --frames 1 \
    --deploy-reparam \
    | tee "$RUN/profile_waveshift_deploy_${W}x${H}.txt"
done
```

Do not describe the complete GSTS block as zero FLOPs. Only the indexing/shift
operation is zero MAC; fusion convolutions are counted.

## 9. Step E — select one GPU and run real native T=30 preflight

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
  --format=csv,noheader,nounits
```

Select one GPU with the largest free memory and export only that physical index:

```bash
export CUDA_VISIBLE_DEVICES=<selected_physical_gpu>
```

Record physical GPU index/name/total/free memory before the run. Inside PyTorch,
the selected device is `cuda:0`.

Run preflight:

```bash
cd "$REPO/nanovsr_deblur"
mkdir -p "$RUN/preflight"
set -o pipefail
python train_nanovnr_waveshift_pagf_fullframe.py \
  --variant waveshift_edge \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/preflight" \
  --short-frames 7 \
  --long-frames 30 \
  --amp \
  --grad-checkpoint \
  --preflight-only \
  2>&1 | tee "$RUN/preflight/preflight.log"
```

The script must execute real forward, Charbonnier, backward, gradient clipping,
and optimizer step for every discovered family/native-resolution pair.

If any native resolution OOMs, do not crop, resize, reduce T, change channels,
disable modules, CPU-offload, or modify the model. Stop and report:

```text
HUMAN_ACTION_REQUIRED: YES
REASON: NATIVE_FULLFRAME_T30_PREFLIGHT_OOM
DATASET: <family>
RESOLUTION: <WxH>
GPU: <name/index>
GPU_TOTAL_FREE_BEFORE: <...>
OOM_STAGE: <...>
```

## 10. Step F — full training or exact resume

Only start after all preflight checks PASS.

```bash
cd "$REPO/nanovsr_deblur"
mkdir -p "$RUN/train"

RESUME_ARGS=()
if [ -f "$RUN/train/latest.pth" ]; then
  RESUME_ARGS=(--resume "$RUN/train/latest.pth")
fi

set -o pipefail
python train_nanovnr_waveshift_pagf_fullframe.py \
  --variant waveshift_edge \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/train" \
  --short-frames 7 \
  --long-frames 30 \
  --switch-iter 50000 \
  --total-iterations 150000 \
  --workers 2 \
  --lr 3e-4 \
  --eta-min 1e-7 \
  --save-every 5000 \
  --amp \
  --grad-checkpoint \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$RUN/train/train.log"
```

Expected:

```text
RECIPE_ID=nanovnr_waveshift_pagf_edge_native_fullframe_bsd3ms24ms_v2
ARCHITECTURE=NanoVNRWaveShiftPAGF
VARIANT=waveshift_edge
```

Monitor loss, gradient norm, LR, source-family sampling, resolution, GPU memory,
and checkpoint creation. Non-finite loss/gradient, repeated explosive spikes,
missing checkpoint, wrong recipe, or a different model config is a failure.

## 11. Step G — checkpoint selection under one fixed protocol

Evaluate at least 50k, 75k, 100k, 125k, and 150k:

```bash
cd "$REPO/nanovsr_deblur"
mkdir -p "$RUN/eval"
for STEP in 0050000 0075000 0100000 0125000 0150000; do
  CKPT="$RUN/train/step_${STEP}.pth"
  test -f "$CKPT" || { echo "MISSING_CHECKPOINT=$CKPT"; exit 2; }
  python eval_gopro_nanovnr_waveshift_pagf.py \
    --gopro-root "$GOPRO" \
    --checkpoint "$CKPT" \
    --num-frames 15 \
    --max-clips 100 \
    --fp16 \
    --deploy-reparam \
    2>&1 | tee "$RUN/eval/step_${STEP}_T15.log"
done
```

Protocol is fixed:

- same first 100 GoPro test clips;
- native full frame;
- T=15;
- RGB PSNR;
- prediction clamped to `[0,1]`;
- FP16 inference;
- deployed RepConv.

Choose `BEST_T15_CHECKPOINT` by highest `OUTPUT_PSNR_RGB`. Do not choose by
training loss or business-video appearance.

Sanity gate: `OUTPUT_PSNR_RGB` must exceed `INPUT_PSNR_RGB`. If it does not,
treat the model/evaluation as failed and audit pairing/protocol before drawing a
model conclusion.

The historical `29.9600 dB` RepVGG result may only be used when every protocol
item is confirmed identical. Otherwise label the comparison `NOT_COMPARABLE`.

## 12. Step H — exact same-target context evaluation

Run the best checkpoint on the exact same `(sequence, absolute center index)`
targets across T=7, T=15, and T=30:

```bash
BEST=<absolute path to BEST_T15_CHECKPOINT>
python eval_gopro_context_matched.py \
  --gopro-root "$GOPRO" \
  --checkpoint "$BEST" \
  --contexts 7 15 30 \
  --max-targets 100 \
  --fp16 \
  --deploy-reparam \
  2>&1 | tee "$RUN/eval/best_context_matched.log"
```

Report:

- `CENTER_T7_OUTPUT_PSNR`;
- `CENTER_T15_OUTPUT_PSNR`;
- `CENTER_T30_OUTPUT_PSNR`;
- `CENTER_CONTEXT_GAIN_T15_VS_T7`;
- `CENTER_CONTEXT_GAIN_T30_VS_T15`.

Do not call separately selected first-100 T windows "matched".

## 13. Step I — same-protocol RGB NAFNet baseline comparison

Look for the existing baseline checkpoints only in the known baseline run:

```text
/mnt/ssd1/z00919662/motion_deblur/runs/
nanovnr_nafnet_rgb_fullframe_bsd_train_test_20260904/train
```

Do not search arbitrary `.pth` files and guess their identity. A baseline is
comparable only when its checkpoint says:

```text
architecture=NanoVNRNAFNetRGB
recipe_id=nanovnr_nafnet_rgb_native_fullframe_mix_bsd_train_test_v2
args.bsd_root=/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms
```

If comparable 50k/75k/100k/125k/150k baseline checkpoints exist, evaluate them
with `eval_gopro_nanovnr_nafnet_rgb.py` using the same T=15/first-100/native/RGB/
FP16 protocol and choose its best checkpoint. Report:

```text
GAIN_VS_COMPARABLE_RGB_NAFNET_T15 = improved_best - baseline_best
```

Strict effect success requires a positive gain; `>= +0.05 dB` is considered a
clear numerical improvement. If the gain is negative, do not claim the new
architecture is better, even if its output looks sharper.

If the comparable baseline is absent or unfinished, report:

```text
COMPARABLE_BASELINE_AVAILABLE: NO
EFFECT_VS_BASELINE: UNVERIFIED
```

Do not silently substitute RepVGG, Shift-Net, BSSTNet, or a different dataset
recipe as the architecture baseline.

## 14. Step J — native business-video inference with correct GSTS halo

Input:

`/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4`

First use core chunk 15. The script automatically uses halo 2 for two GSTS
blocks, processes only non-overlapping core frames recurrently, carries the LL
forward state, and resets backward state per core chunk.

```bash
mkdir -p "$RUN/business"
python infer_video_nanovnr_waveshift_pagf.py \
  --input "$INPUT" \
  --checkpoint "$BEST" \
  --output "$RUN/business/business_nanovnr_waveshift_pagf.mp4" \
  --side-by-side-output \
    "$RUN/business/input_vs_nanovnr_waveshift_pagf.mp4" \
  --chunk 15 \
  --fp16 \
  2>&1 | tee "$RUN/business/inference.log"
```

If inference OOMs, only reduce the inference core chunk to 9, then 7. Keep the
automatic GSTS halo and native resolution. Do not alter training or model config.

Verify output frame count, fps, and resolution equal the input. Then run:

```bash
python audit_video_output.py \
  --input "$INPUT" \
  --output "$RUN/business/business_nanovnr_waveshift_pagf.mp4" \
  --chunk 15 \
  --preview "$RUN/business/five_frame_preview.jpg" \
  2>&1 | tee "$RUN/business/video_output_audit.log"
```

If inference used chunk 9 or 7, pass that exact value to `audit_video_output.py`.

Automated red flags requiring explicit reporting:

- output frame count/resolution mismatch;
- near-zero output change, suggesting an identity model;
- absolute mean B/G/R shift greater than 3 gray levels;
- large new black/white clipping rate;
- chunk-boundary discontinuity ratio above 1.5;
- extreme Laplacian-variance increase suggesting oversharpen/ringing.

These checks do not replace visual inspection.

## 15. Required human quality check

The user must manually inspect:

- `input_vs_nanovnr_waveshift_pagf.mp4`;
- `five_frame_preview.jpg`.

Check face shape, moving text, edge recovery, trees/water/background motion,
ringing, oversharpening, ghosting, temporal flicker, chunk boundaries, and color
shift. CodeAgent must provide the exact paths. Do not mark subjective quality as
PASS before this inspection.

## 16. Stop conditions

Stop and return `HUMAN_ACTION_REQUIRED: YES` for:

- missing required code commit;
- dirty existing repository;
- missing/incompatible environment;
- architecture/unit/deploy-fusion test failure;
- BSD path-policy violation;
- blur/GT filename or shape mismatch;
- native T=30 preflight OOM;
- non-finite loss or gradient;
- missing/corrupt checkpoint;
- output frame/resolution mismatch.

Do not change code or launch unrequested ablation training after a stop.

## 17. Required final report

Return all fields below:

```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
HUMAN_ACTION: inspect <video> and <preview>, or exact blocker

GITHUB_BRANCH: agent/nanovnr-waveshift-pagf-fullframe-20260904
GITHUB_HEAD: <sha>
REQUIRED_CODE_COMMIT_PRESENT: YES / NO
SOURCE_CODE_MODIFIED_BY_CODEAGENT: NO

ARCHITECTURE: NanoVNRWaveShiftPAGF
VARIANT: waveshift_edge
RECIPE_ID: nanovnr_waveshift_pagf_edge_native_fullframe_bsd3ms24ms_v2
MODEL_CONFIG: <dict>
UNIT_TESTS: PASS / FAIL
HAAR_ROUNDTRIP_MAX_ABS_DIFF: <value>
DEPLOY_OUTPUT_MAX_ABS_DIFF: <value>
DEPLOY_STATE_MAX_ABS_DIFF: <value>

BSD_TRAIN_ONLY_FOR_TRAINING: YES / NO
BSD_NESTED_CONFIG_SPLITS_USED: NO / YES
GOPRO_T7_WINDOWS: <n>
GOPRO_T30_WINDOWS: <n>
DVD_T7_WINDOWS: <n>
DVD_T30_WINDOWS: <n>
BSD_TRAIN_T7_WINDOWS: <n>
BSD_TRAIN_T30_WINDOWS: <n>
BSD_TEST_T7_WINDOWS: <n>
BSD_TEST_T30_WINDOWS: <n>
GOPRO_NATIVE_RESOLUTIONS: <...>
DVD_NATIVE_RESOLUTIONS: <...>
BSD_TRAIN_NATIVE_RESOLUTIONS: <...>
BSD_TEST_NATIVE_RESOLUTIONS: <...>

GPU: <physical index and name>
GPU_TOTAL_FREE_BEFORE: <...>
PREFLIGHT_T30_NATIVE_FULLFRAME: PASS / FAIL
PREFLIGHT_PEAK_MEMORY_BY_RESOLUTION: <...>

PARAMS_BASELINE: <n>
PARAMS_WAVESHIFT_TRAIN: <n>
PARAMS_WAVESHIFT_DEPLOY: <n>
MACS_PER_FRAME_BASELINE_640x360: <...>
MACS_PER_FRAME_WAVESHIFT_640x360: <...>
MACS_PER_FRAME_WAVESHIFT_1280x720: <...>
MACS_PER_FRAME_WAVESHIFT_1920x1080: <...>

TRAIN_FINAL_STEP: <n>
CHECKPOINT_PSNR_T15: <50k/75k/100k/125k/150k table>
BEST_T15_CHECKPOINT: <path>
BEST_T15_INPUT_PSNR: <dB>
BEST_T15_OUTPUT_PSNR: <dB>
GAIN_VS_BLUR_INPUT: <dB>

MATCHED_CENTER_TARGETS: <n>
CENTER_T7_OUTPUT_PSNR: <dB>
CENTER_T15_OUTPUT_PSNR: <dB>
CENTER_T30_OUTPUT_PSNR: <dB>
CENTER_CONTEXT_GAIN_T15_VS_T7: <dB>
CENTER_CONTEXT_GAIN_T30_VS_T15: <dB>

COMPARABLE_BASELINE_AVAILABLE: YES / NO
COMPARABLE_BASELINE_CHECKPOINT: <path or N/A>
COMPARABLE_BASELINE_T15_PSNR: <dB or N/A>
GAIN_VS_COMPARABLE_RGB_NAFNET_T15: <dB or UNVERIFIED>
EFFECT_NUMERICAL_STATUS: CLEAR_GAIN / SMALL_GAIN / REGRESSION / UNVERIFIED

BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE_OUTPUT: <path>
FIVE_FRAME_PREVIEW: <path>
BUSINESS_FRAME_SIZE_COUNT_FPS: <...>
MEAN_CHANNEL_SHIFT_BGR: <...>
LAPLACIAN_VARIANCE_RATIO: <...>
CHUNK_BOUNDARY_DISCONTINUITY_RATIO: <...>
AUTOMATED_VIDEO_RED_FLAGS: <none or list>
SUBJECTIVE_QUALITY_STATUS: PENDING_USER_REVIEW
```

Training completion alone is not an effect PASS. Numerical gain requires the
same-protocol baseline; subjective quality remains pending until the user views
the generated comparison video and preview.
