# CODEAGENT TASK — NanoVSR-style U-Net full-frame deblur, GoPro + DVD + BSD

## Goal
Train a new full-frame video motion deblurring model that keeps NanoVSR's useful temporal idea (bidirectional recurrent propagation and short-to-long temporal curriculum) but replaces the old full-resolution RepVGG propagation backbone with a lightweight U-Net.

This is a NEW architecture experiment. Do not resume or overwrite the old NanoVSR-Deblur RepVGG checkpoints.

Repository:
`https://github.com/hihiok/video_motion_deblur.git`

Branch:
`agent/nanovsr-deblur-unet-fullframe-20260903`

Primary code:
- `nanovsr_deblur/models/nanovsr_unet_deblur.py`
- `nanovsr_deblur/train_unet_fullframe.py`
- `nanovsr_deblur/audit_fullframe_datasets.py`
- `nanovsr_deblur/eval_gopro_unet_fullframe.py`
- `nanovsr_deblur/infer_video_unet_fullframe.py`
- `nanovsr_deblur/profile_unet_fullframe.py`

## Non-negotiable experiment rules
1. TRAIN ON NATIVE FULL FRAMES. No random crop.
2. Do not resize training frames.
3. Do not silently downscale 720p to make training fit.
4. Do not silently switch to patch training.
5. Batch size is exactly 1 because GoPro/DVD/BSD may have different native resolutions.
6. Use GoPro + DVD + BSD with family-balanced sampling.
7. Loss is Charbonnier ONLY for the whole run.
8. No edge loss, temporal-delta loss, SSIM loss, perceptual loss, frequency loss, optical-flow loss or distillation.
9. Phase 1: T=7 through step 50,000.
10. Phase 2: T=30 from step 50,001 through step 150,000.
11. Optimizer and LR scheduler MUST continue across the T=7 -> T=30 switch. Do not reset either.
12. Adam betas=(0.9, 0.99), initial LR=3e-4, one cosine schedule to eta_min=1e-7 over 150k total steps.
13. Gradient clip=0.5.
14. Use the new U-Net architecture only. Do not fall back to `NanoVSRDeblur` / RepVGG.
15. Do not resume any old RepVGG NanoVSR Stage1/Stage2 checkpoint.
16. Use gradient checkpointing for training (`--grad-checkpoint`) because T=30 full-frame is memory intensive.
17. If full-frame T=30 still OOMs on the largest-memory available GPU, STOP and report HUMAN_ACTION_REQUIRED. Do NOT crop, resize, shorten T, or shrink the model without explicit user approval.
18. CPU DataLoader workers <= 2.
19. Never delete existing datasets, checkpoints or runs.

## Architecture being tested
Default model config:
- U-Net channels: 32 -> 48 -> 64
- two spatial downsamples; temporal recurrent state lives at 1/4 resolution
- separate forward and backward temporal propagation modules
- 6 residual temporal blocks per direction at the 1/4-scale bottleneck
- full-resolution and half-resolution U-Net skip connections
- residual RGB output at the native input resolution
- no BatchNorm
- fully convolutional; input H/W are not fixed

The old `48x12` label does NOT apply to this U-Net architecture. Report actual parameter count and MACs from the new profiler.

## Dataset roots
Use exactly these roots unless audit proves the directory name differs only by case:

```bash
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
```

The loader already supports common GoPro/DVD blur-sharp layouts and official BSD-style paths such as:
`<BSD_config>/train/<sequence>/Blur/RGB/*.png`
paired with
`<BSD_config>/train/<sequence>/Sharp/RGB/*.png`.

Frame pairing is filename-exact. Never silently pair mismatched frame names by index.

## Proxy / SSL rules
The server is behind an internal HTTPS-inspection proxy.

Before GitHub operations:
```bash
git config --global http.sslVerify false
```

Use the existing server-local proxy environment. If a conda environment contains the approved proxy activation script, load it without printing credentials:
```bash
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
```

Do NOT echo proxy usernames/passwords and do NOT commit credentials.

## Workspace
```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovsr_unet_fullframe
RUN=$ROOT/runs/nanovsr_unet_fullframe_mix_20260903
mkdir -p "$RUN"
```

Clone/pull the branch exactly:
```bash
cd "$ROOT"
git config --global http.sslVerify false

if [ ! -d "$REPO/.git" ]; then
  git clone -b agent/nanovsr-deblur-unet-fullframe-20260903 \
    https://github.com/hihiok/video_motion_deblur.git "$REPO"
else
  cd "$REPO"
  git fetch origin
  git checkout agent/nanovsr-deblur-unet-fullframe-20260903
  git reset --hard origin/agent/nanovsr-deblur-unet-fullframe-20260903
fi
cd "$REPO"
git rev-parse HEAD
```

## Step A — environment and code sanity
Prefer the existing deblur training environment that already works on this server. Do not create a large new environment unless required.

Typical setup:
```bash
source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate deblur_runtime
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
cd "$REPO/nanovsr_deblur"
```

Check:
```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu_count', torch.cuda.device_count())
PY

python -m py_compile \
  models/nanovsr_unet_deblur.py \
  train_unet_fullframe.py \
  audit_fullframe_datasets.py \
  eval_gopro_unet_fullframe.py \
  infer_video_unet_fullframe.py \
  profile_unet_fullframe.py
```

Install only missing lightweight requirements if needed:
```bash
pip install -r requirements.txt
```
Do not upgrade/downgrade PyTorch unless absolutely necessary.

## Step B — select GPU
Inspect GPU free memory:
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
```

Choose ONE currently idle GPU with the largest free memory. Do not use a heavily occupied GPU.
Set:
```bash
export CUDA_VISIBLE_DEVICES=<SELECTED_GPU_INDEX>
```

Record physical GPU index, model name, total memory and free memory in the final report.

## Step C — full-frame dataset audit
First verify all three roots exist:
```bash
for p in "$GOPRO" "$DVD" "$BSD"; do
  test -d "$p" || { echo "MISSING_DATASET=$p"; exit 2; }
done
```

Run the native-resolution audit:
```bash
cd "$REPO/nanovsr_deblur"
python audit_fullframe_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  2>&1 | tee "$RUN/dataset_fullframe_audit.log"
```

Required audit conditions:
- GoPro, DVD and BSD all produce valid T=7 windows.
- GoPro, DVD and BSD all produce valid T=30 windows.
- blur/GT names are aligned exactly.
- blur and sharp resolution match.
- log native resolutions for each family.
- no image files are modified, renamed, moved or resized.

If a dataset layout is not discovered, inspect only that root and make the minimum loader fix needed to support the actual existing layout. Do NOT reorganize the dataset itself. Commit any genuine loader fix to the same branch and report it.

## Step D — profile the NEW U-Net architecture
Run both 640x360 and 1280x720 profiles:
```bash
python profile_unet_fullframe.py \
  --height 360 --width 640 --frames 3 \
  --base-channels 32 --mid-channels 48 --bottleneck-channels 64 \
  --num-temporal-blocks 6 \
  | tee "$RUN/profile_640x360.txt"

python profile_unet_fullframe.py \
  --height 720 --width 1280 --frames 1 \
  --base-channels 32 --mid-channels 48 --bottleneck-channels 64 \
  --num-temporal-blocks 6 \
  | tee "$RUN/profile_1280x720.txt"
```

Report:
- exact parameters
- MAC/frame @640x360
- FLOP/frame @640x360 using 2 FLOP/MAC convention
- MAC/frame @1280x720
- FLOP/frame @1280x720 using 2 FLOP/MAC convention

Do not reuse old RepVGG 0.588M / 137G numbers.

## Step E — mandatory T=30 FULL-FRAME backward preflight
This is mandatory before launching 150k steps.

Use AMP + gradient checkpointing:
```bash
python train_unet_fullframe.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/train" \
  --base-channels 32 \
  --mid-channels 48 \
  --bottleneck-channels 64 \
  --num-temporal-blocks 6 \
  --short-frames 7 \
  --long-frames 30 \
  --switch-iter 50000 \
  --total-iterations 150000 \
  --workers 0 \
  --lr 3e-4 \
  --eta-min 1e-7 \
  --amp \
  --grad-checkpoint \
  --preflight-only \
  2>&1 | tee "$RUN/t30_fullframe_preflight.log"
```

The preflight performs an actual forward + Charbonnier + backward for each observed family/native-resolution class.

PASS only if all native resolutions pass T=30 full-frame backward.

If OOM:
1. Confirm `--grad-checkpoint` and AMP were enabled.
2. Confirm you selected the largest-memory idle GPU.
3. You may retry on another available GPU with MORE memory.
4. Do NOT crop.
5. Do NOT resize.
6. Do NOT change T=30 to T=15.
7. Do NOT shrink channels/blocks.
8. If no available GPU can pass, stop:
   `HUMAN_ACTION_REQUIRED: YES — full-frame T=30 exceeds available GPU memory.`

## Step F — clean full-frame training from RANDOM INIT
Only run after preflight PASS.

Do NOT use `--resume` for the initial run.

```bash
mkdir -p "$RUN/train"
python train_unet_fullframe.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/train" \
  --base-channels 32 \
  --mid-channels 48 \
  --bottleneck-channels 64 \
  --num-temporal-blocks 6 \
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
  2>&1 | tee "$RUN/train/train.log"
```

Training invariants to verify from logs:
- FULL_FRAME=YES
- RANDOM_CROP=NO
- RESIZE=NO
- BATCH=1
- LOSS=CharbonnierOnly
- phase short uses T=7 through step 50000
- phase long begins at step 50001 and uses T=30
- optimizer state is NOT recreated at 50001
- scheduler is NOT recreated at 50001
- LR follows one continuous cosine curve
- source family varies among GoPro/DVD/BSD
- H/W in logs reflect native dataset resolution

If interrupted after a valid checkpoint, resume ONLY this same recipe checkpoint:
```bash
python train_unet_fullframe.py <same arguments> \
  --resume "$RUN/train/latest.pth"
```
The script rejects old RepVGG checkpoints by architecture/recipe ID.

## Step G — checkpoint evaluation on GoPro
Old reference from the previous RepVGG experiment:
`OLD_REPVGG_STAGE1_60K_T15_PSNR_RGB = 29.9600 dB`

This is a reference only. Do not assume the new evaluator is perfectly identical until you verify the same GoPro test split/order and first 100 clip convention.

Evaluate these U-Net checkpoints when present:
- 50k
- 75k
- 100k
- 125k
- 150k

Use full native GoPro frames, T=15, first 100 clips, all frames:
```bash
for STEP in 0050000 0075000 0100000 0125000 0150000; do
  CKPT="$RUN/train/step_${STEP}.pth"
  [ -f "$CKPT" ] || continue
  python eval_gopro_unet_fullframe.py \
    --gopro-root "$GOPRO" \
    --checkpoint "$CKPT" \
    --num-frames 15 \
    --max-clips 100 \
    --fp16 \
    2>&1 | tee "$RUN/eval_t15_step_${STEP}.txt"
done
```

Select `BEST_T15` by the highest T=15 PSNR from the SAME protocol.

Then evaluate BEST_T15 with T=7, T=15 and T=30:
```bash
BEST=<best checkpoint path>
for T in 7 15 30; do
  python eval_gopro_unet_fullframe.py \
    --gopro-root "$GOPRO" \
    --checkpoint "$BEST" \
    --num-frames "$T" \
    --max-clips 100 \
    --fp16 \
    2>&1 | tee "$RUN/best_eval_t${T}.txt"
done
```

Also run center-only T=7/T=15/T=30 on the same first 100 windows for context diagnostics:
```bash
for T in 7 15 30; do
  python eval_gopro_unet_fullframe.py \
    --gopro-root "$GOPRO" \
    --checkpoint "$BEST" \
    --num-frames "$T" \
    --max-clips 100 \
    --center-only \
    --fp16 \
    2>&1 | tee "$RUN/best_eval_center_t${T}.txt"
done
```

Do not mix T=7 and T=30 numbers as evidence of training gain. They have different inference context. Training gain is assessed across checkpoints at the SAME T=15 protocol.

## Step H — business video inference
Input:
```bash
INPUT=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Run BEST_T15 at native 1280x720; do NOT resize input:
```bash
python infer_video_unet_fullframe.py \
  --input "$INPUT" \
  --checkpoint "$BEST" \
  --output "$RUN/nanovsr_unet_fullframe_business_t15.mp4" \
  --chunk 15 \
  --overlap 4 \
  --fp16 \
  2>&1 | tee "$RUN/business_infer.log"
```

Generate side-by-side with ffmpeg:
```bash
ffmpeg -y \
  -i "$INPUT" \
  -i "$RUN/nanovsr_unet_fullframe_business_t15.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" -an \
  "$RUN/input_vs_nanovsr_unet_fullframe_t15.mp4"
```

Do not claim subjective success automatically. Human visual review is required for:
- deblur strength
- facial/texture artifacts
- ringing/oversharpening
- color shift
- temporal flicker

## Step I — Git rules
Do not commit checkpoints, dataset files, MP4 outputs or logs.
Only commit genuine code fixes needed for this task.
Never commit credentials.

If code changes are required due to a real runtime/data-layout bug:
- make the smallest fix
- run `py_compile` again
- commit to `agent/nanovsr-deblur-unet-fullframe-20260903`
- report the exact commit and diff summary

## Required final report
Return this structure:

```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovsr-deblur-unet-fullframe-20260903
GITHUB_COMMIT: <actual pulled/final sha>
GPU_PHYSICAL_INDEX: <index>
GPU: <name>
GPU_TOTAL_MEMORY_GIB: <GiB>
PYTORCH: <version>

ARCHITECTURE: NanoVSRUNetDeblur
MODEL_CONFIG: base=32 mid=48 bottleneck=64 temporal_blocks_per_direction=6
PARAMS: <count and M>
MACS_PER_FRAME_640x360: <G>
FLOPS_PER_FRAME_640x360_2xMAC: <G>
MACS_PER_FRAME_1280x720: <G>
FLOPS_PER_FRAME_1280x720_2xMAC: <G>

TRAINING_INPUT: NATIVE_FULL_FRAME
RANDOM_CROP: NO
RESIZE: NO
BATCH_SIZE: 1
DATASETS: GoPro + DVD + BSD
DATASET_NATIVE_RESOLUTIONS: <report>
DATASET_FAMILY_WINDOWS_T7: <report>
DATASET_FAMILY_WINDOWS_T30: <report>
LOSS: Charbonnier only
CURRICULUM: T7 0-50k -> T30 50k-150k

T30_FULLFRAME_PREFLIGHT: PASS / FAIL
T30_PREFLIGHT_PEAK_GPU_GIB_BY_RESOLUTION: <report>

PSNR_T15_STEP_50K: <dB>
PSNR_T15_STEP_75K: <dB>
PSNR_T15_STEP_100K: <dB>
PSNR_T15_STEP_125K: <dB>
PSNR_T15_STEP_150K: <dB>
BEST_T15_CHECKPOINT: <path>
BEST_T15_PSNR: <dB>
GAIN_VS_OLD_REPVGG_29P9600: <dB; note protocol compatibility>

BEST_PSNR_T7: <dB>
BEST_PSNR_T15: <dB>
BEST_PSNR_T30: <dB>
BEST_CENTER_PSNR_T7: <dB>
BEST_CENTER_PSNR_T15: <dB>
BEST_CENTER_PSNR_T30: <dB>

FINAL_CHECKPOINT_SHA256: <sha256>
BUSINESS_INPUT: /mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE: <path>
BUSINESS_INFERENCE_PEAK_GPU_GIB: <GiB>
VISUAL_REVIEW_REQUIRED: YES

NOTES: <concise factual notes only>
```

Do not automatically claim this U-Net model is better than Shift-Net-s or BSSTNet based only on GoPro PSNR. Business-video visual comparison is required.
