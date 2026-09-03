# CODEAGENT TASK — NanoVSR-Deblur original-style curriculum on GoPro + DVD + BSD

## Goal
Train the existing NanoVSR-Deblur 48x12 model from scratch using a training recipe that stays as close as practical to the official NanoVSR recipe, adapted only where the task changes from x4 VSR to x1 video motion deblurring.

The main experiment must answer whether the NanoVSR architecture can become a strong small deblurring model WITHOUT custom temporal/edge/distillation losses.

Repository:
`https://github.com/hihiok/video_motion_deblur.git`

Branch:
`agent/nanovsr-deblur-original-recipe-mix-20260903`

Primary code:
- `nanovsr_deblur/train_original_recipe.py`
- `nanovsr_deblur/data/mixed_deblur.py`
- `nanovsr_deblur/audit_mixed_datasets.py`

## Non-negotiable recipe
Use the existing NanoVSR-Deblur architecture:
- num_feat = 48
- num_blocks = 12
- bidirectional additive recurrence
- RepVGG-style blocks
- x1 residual deblur head

Train FROM RANDOM INITIALIZATION. Do NOT resume any old Stage1/Stage2 checkpoint because those were trained with edge/temporal losses.

Loss for the entire run:
- Charbonnier ONLY
- no edge loss
- no temporal-delta loss
- no SSIM/perceptual/frequency loss
- no distillation

Optimizer/schedule for the entire run:
- Adam, betas=(0.9, 0.99)
- initial LR 3e-4
- one CosineAnnealingLR over the complete 150k optimizer steps
- eta_min 1e-7
- gradient clipping = 0.5
- optimizer and scheduler MUST NOT reset at the phase switch

Curriculum:
- Phase 1: steps 1..50000, T=7
- Phase 2: steps 50001..150000, T=30
- same model, optimizer, scheduler, patch size and batch size across both phases
- the only intended phase change is sequence length T=7 -> T=30

Training families in BOTH phases:
- GoPro
- DVD
- BSD

The loader uses family-balanced sampling so each family has equal total sampling weight; BSD's multiple exposure configurations must not dominate merely because it has more windows.

## Important difference from original NanoVSR patch size
Official NanoVSR uses a 256x256 GT crop for x4 SR, so the recurrent backbone actually processes about a 64x64 LR crop. Our x1 deblur model processes the crop at full size. Therefore do NOT blindly assume a 256x256 deblur patch will fit T=30 on a V100.

Run the provided T=30 backward-pass preflight and select the largest safe patch from:
256, 192, 160, 128.

Use the SAME selected patch size in Phase 1 and Phase 2.

## Server paths
Expected dataset roots:

```bash
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
```

If DVD capitalization/name differs, inspect only:
`/mnt/ssd1/z00919662/motion_deblur/datasets`
and resolve the existing DVD directory. Do not search the whole filesystem.

Run root:
```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovsr_deblur
RUN=$ROOT/runs/nanovsr_deblur_original_mix_20260903
mkdir -p "$RUN"
```

## Proxy / SSL
Do not print proxy credentials.

Before GitHub operations:
```bash
git config --global http.sslVerify false
```

If an existing conda proxy activation file exists, source it silently:
```bash
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
```

Do not commit proxy credentials.

## Step A — pull exact branch
```bash
cd "$ROOT"
if [ ! -d "$REPO/.git" ]; then
  git clone -b agent/nanovsr-deblur-original-recipe-mix-20260903 \
    https://github.com/hihiok/video_motion_deblur.git "$REPO"
else
  cd "$REPO"
  git fetch origin
  git checkout agent/nanovsr-deblur-original-recipe-mix-20260903
  git reset --hard origin/agent/nanovsr-deblur-original-recipe-mix-20260903
fi
cd "$REPO"
git rev-parse HEAD
```

Record the exact commit.

## Step B — environment
Prefer the already working environment used for the previous NanoVSR-Deblur training (`deblur_runtime`) if available.

CPU DataLoader workers MUST be <=2.

```bash
cd "$REPO/nanovsr_deblur"
python - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
PY
```

Do not upgrade/downgrade PyTorch unless there is a real blocker.

## Step C — dataset audit BEFORE training
First verify all three roots exist.

Then run:
```bash
python audit_mixed_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --split train \
  --patch-size 128 \
  | tee "$RUN/dataset_audit.txt"
```

The audit MUST successfully discover T=7 and T=30 windows for all three families.

Expected supported layouts include:
- GoPro/DVD: `train/blur/<seq>` + `train/gt|GT|sharp/<seq>`
- layout without explicit split: `blur/<seq>` + `gt/<seq>`
- official BSD: `BSD_<config>/train/<seq>/Blur/RGB/*.png` + `Sharp/RGB/*.png`

Important safety checks:
1. Do not rename/move dataset folders.
2. Do not create synthetic frame pairings.
3. Frame filenames between blur and GT must match exactly.
4. Randomly inspect at least 10 aligned blur/GT pairs per family and report dimensions and filenames.
5. Ensure test directories are NOT included in train discovery.

If any family cannot produce T=30 windows, STOP before training with:
`HUMAN_ACTION_REQUIRED: YES`
and print a compact directory tree (max depth 5) for that dataset only. Do not silently drop a family or shorten T.

## Step D — T=30 memory preflight
Use ONE idle V100 32GB GPU. Do not run multiple preflights in parallel.

Try patch sizes in descending order until the first safe configuration is found:
256 -> 192 -> 160 -> 128.

For each candidate P:
```bash
CUDA_VISIBLE_DEVICES=<GPU> python train_original_recipe.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/preflight_p${P}" \
  --num-feat 48 \
  --num-blocks 12 \
  --short-frames 7 \
  --long-frames 30 \
  --patch-size "$P" \
  --batch-size 1 \
  --workers 2 \
  --lr 3e-4 \
  --amp \
  --preflight-only \
  2>&1 | tee "$RUN/preflight_p${P}.log"
```

A patch is SAFE only if:
- preflight forward+backward succeeds
- no CUDA OOM
- peak allocated GPU memory <= 28 GiB

Select the largest SAFE patch and set:
```bash
PATCH=<selected>
```

The 28 GiB threshold leaves headroom for sequence/image variation and allocator fragmentation.

If even 128 fails, STOP with HUMAN_ACTION_REQUIRED. Do not change T=30, model size, or batch size without user approval.

## Step E — clean original-style training
Do NOT resume old checkpoints.

Start a new run:
```bash
mkdir -p "$RUN/train"

CUDA_VISIBLE_DEVICES=<GPU> python train_original_recipe.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/train" \
  --num-feat 48 \
  --num-blocks 12 \
  --short-frames 7 \
  --long-frames 30 \
  --switch-iter 50000 \
  --total-iterations 150000 \
  --patch-size "$PATCH" \
  --batch-size 1 \
  --workers 2 \
  --lr 3e-4 \
  --eta-min 1e-7 \
  --save-every 5000 \
  --amp \
  2>&1 | tee "$RUN/train/train.log"
```

Training invariants that MUST be verified from the log:
- step <= 50000: phase=short, T=7
- step >= 50001: phase=long, T=30
- same patch size throughout
- same batch size throughout
- loss is Charbonnier only
- optimizer is Adam, not AdamW
- no weight decay
- optimizer state is continuous across 50k
- scheduler state is continuous across 50k
- no LR reset at 50k

If training is interrupted for a non-code reason, resume ONLY from a checkpoint generated by `train_original_recipe.py`. The script intentionally rejects old foreign-recipe checkpoints.

## Step F — checkpoint evaluation on GoPro
The previous best clean-ish baseline to beat is:
- old Stage1@60k, T=15 old protocol: 29.9600 dB RGB

Use the same existing `eval_gopro.py` protocol for model selection so numbers are comparable.

After training, evaluate these checkpoints if they exist:
- 50000
- 75000
- 100000
- 125000
- 150000

First pass: T=15, first 100 clips.

Example:
```bash
python eval_gopro.py \
  --gopro-root "$GOPRO" \
  --checkpoint "$RUN/train/step_0050000.pth" \
  --num-frames 15 \
  --max-clips 100
```

Repeat for each candidate and produce:
`checkpoint | phase | GoPro PSNR RGB T15 (100 clips)`

Select the best checkpoint by this SAME T=15/100-clip metric.

For BEST_NEW, additionally evaluate:
- T=7, 200 clips
- T=15, 200 clips
- T=30, 100 clips if runtime permits

Do not compare T=7 vs T=15 vs T=30 as if context length were a training gain. The important model-selection comparison is checkpoint-to-checkpoint at the SAME T=15 protocol.

Report:
`GAIN_VS_OLD_STAGE1_T15 = BEST_NEW_T15_100 - 29.9600`

## Step G — business-video inference
Business input:
`/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4`

Run BEST_NEW with T/chunk=15 first for direct comparison with the previous baseline:
```bash
CUDA_VISIBLE_DEVICES=<GPU> python infer_video.py \
  --input /mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4 \
  --checkpoint <BEST_NEW> \
  --output "$RUN/business_best_new_t15.mp4" \
  --chunk 15 \
  --overlap 4 \
  --fp16
```

Generate side-by-side input vs new output with ffmpeg. Do not overwrite previous NanoVSR business outputs.

If old Stage1@60k business output still exists, also generate a 3-way comparison:
`Input | Old Stage1@60k | New Original-Recipe Best`

Human visual review remains required for sharpness, flicker, face deformation, color shift, and texture artifacts.

## Step H — profile final model
The architecture is still 48x12, but report it again for traceability:
```bash
python profile_model.py \
  --num-feat 48 --num-blocks 12 \
  --height 360 --width 640 --frames 7 \
  | tee "$RUN/profile_48x12.txt"
```

Report params and per-frame MAC/FLOP numbers using the profiler's convention.

## Do NOT do in this task
- no distillation
- no teacher cache
- no Shift-Net/BSSTNet training
- no edge/temporal/SSIM/perceptual/frequency loss
- no model-width/depth changes
- no old checkpoint resume
- no dataset modifications
- no automatic T reduction
- no extra random experiments after the requested run

## Final report
Return:

```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovsr-deblur-original-recipe-mix-20260903
GITHUB_COMMIT: <sha>
GPU: <name>
PYTORCH: <version>

GOPRO_ROOT: <path>
DVD_ROOT: <path>
BSD_ROOT: <path>
DATASET_AUDIT: <summary, including T7/T30 windows per family>
BSD_CONFIGS_DISCOVERED: <list>

MODEL: NanoVSR-Deblur 48x12
PARAMS: <count>
PATCH_SIZE_SELECTED: <P>
PREFLIGHT_T30_PEAK_GPU_GIB: <GiB>
BATCH_SIZE: 1

RECIPE:
  Phase1: T7 steps 1-50000
  Phase2: T30 steps 50001-150000
  Loss: Charbonnier only
  Optimizer: Adam beta 0.9/0.99
  Scheduler: single cosine 3e-4 -> 1e-7

CHECKPOINT_PSNR_T15_100:
  50000: <dB>
  75000: <dB>
  100000: <dB>
  125000: <dB>
  150000: <dB>

BEST_NEW_CHECKPOINT: <path>
BEST_NEW_T7_200: <dB>
BEST_NEW_T15_200: <dB>
BEST_NEW_T30_100: <dB or timeout>
GAIN_VS_OLD_STAGE1_T15_100: <dB>

MACS_PER_FRAME_640x360: <G>
FLOPS_PER_FRAME_640x360_2xMAC: <G>

BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE: <path>
THREE_WAY_COMPARISON: <path or unavailable>
VISUAL_REVIEW_REQUIRED: YES

KEY_ANSWER:
Did the clean original-style NanoVSR curriculum outperform the old 29.9600 dB Stage1 T15 baseline without custom temporal/edge losses?
```
