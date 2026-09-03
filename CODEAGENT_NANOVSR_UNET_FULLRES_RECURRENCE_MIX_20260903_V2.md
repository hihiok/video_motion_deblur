# CODEAGENT TASK V2 — NanoVSR-Deblur Full-Resolution Recurrent U-Net

Use the latest HEAD of branch:
`agent/nanovsr-deblur-unet-fullframe-20260903`

Repository:
`https://github.com/hihiok/video_motion_deblur.git`

This V2 guide supersedes all older U-Net guides in this branch.

## Goal
Establish a quality-first deblurring upper baseline with:
- native full-frame training
- NO random crop
- NO resize/downscale
- GoPro + DVD + BSD family-balanced mixture
- Charbonnier-only loss
- NanoVSR-style bidirectional recurrence
- recurrent hidden state kept at FULL HxW resolution
- each recurrent update implemented by a complete U-Net

The U-Net may internally use 1/2 and 1/4 spatial branches to enlarge receptive field, but the recurrent input state and recurrent output state must always remain full resolution. Do NOT move the hidden state itself to 1/2 or 1/4 scale.

## Required code
Use only these current files:
- `nanovsr_deblur/models/nanovsr_unet_fullres_deblur.py`
- `nanovsr_deblur/train_unet_fullres.py`
- `nanovsr_deblur/profile_unet_fullres.py`
- `nanovsr_deblur/eval_gopro_unet_fullres.py`
- `nanovsr_deblur/infer_video_unet_fullres.py`
- `nanovsr_deblur/audit_fullframe_datasets.py`

Do NOT train the archived low-resolution recurrent model `nanovsr_unet_deblur.py`.

## Fixed model configuration
```text
ARCHITECTURE = NanoVSRFullResUNetDeblur
base_channels = 48
mid_channels = 64
bottleneck_channels = 96
fullres_blocks = 2
mid_blocks = 2
bottleneck_blocks = 4
forward recurrent U-Net and backward recurrent U-Net have separate parameters
BatchNorm = NONE
```

## Dataset paths
```bash
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
INPUT=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Do not modify, rename, move, or regenerate any dataset image.

BSD official layouts such as:
`<config>/train/<seq>/Blur/RGB/*.png`
and
`<config>/train/<seq>/Sharp/RGB/*.png`
are supported.

## Git / proxy
```bash
git config --global http.sslVerify false
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
```
Do not print or commit proxy credentials.

## Workspace
```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovsr_fullres_unet
RUN=$ROOT/runs/nanovsr_unet_fullres_mix_20260903_v2
mkdir -p "$RUN"

cd "$ROOT"
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
Report the actual latest `GITHUB_COMMIT`.

## Step 1 — syntax / environment audit
```bash
cd "$REPO/nanovsr_deblur"
python -m py_compile \
  models/nanovsr_unet_fullres_deblur.py \
  train_unet_fullres.py \
  profile_unet_fullres.py \
  eval_gopro_unet_fullres.py \
  infer_video_unet_fullres.py \
  audit_fullframe_datasets.py

python - <<'PY'
import torch
print('torch=', torch.__version__)
print('cuda=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i))
PY
```
Prefer an existing working NanoVSR/PyTorch CUDA environment. Do not create a huge new environment unless needed.

## Step 2 — dataset audit
```bash
python audit_fullframe_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  2>&1 | tee "$RUN/dataset_audit.log"
```
Must verify:
- exact blur/GT filename alignment
- T=7 windows
- T=30 windows
- native resolutions
- no silent index pairing

If any dataset cannot produce aligned T=30 windows, STOP and report the exact reason.

## Step 3 — profile the NEW full-resolution recurrent model
```bash
python profile_unet_fullres.py \
  --height 360 --width 640 --frames 3 \
  --base-channels 48 --mid-channels 64 --bottleneck-channels 96 \
  --fullres-blocks 2 --mid-blocks 2 --bottleneck-blocks 4 \
  | tee "$RUN/profile_640x360.txt"

python profile_unet_fullres.py \
  --height 720 --width 1280 --frames 3 \
  --base-channels 48 --mid-channels 64 --bottleneck-channels 96 \
  --fullres-blocks 2 --mid-blocks 2 --bottleneck-blocks 4 \
  | tee "$RUN/profile_1280x720.txt"
```
Report parameters, MAC/frame and 2xMAC FLOPs/frame. Do not reuse any old RepVGG or low-resolution recurrent numbers.

## Step 4 — choose GPU
Use one GPU with the largest free VRAM. Record GPU name, total memory, and free memory. Then set:
```bash
export CUDA_VISIBLE_DEVICES=<selected_gpu>
```

## Step 5 — mandatory T=30 native full-frame training preflight
```bash
python train_unet_fullres.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/preflight" \
  --base-channels 48 \
  --mid-channels 64 \
  --bottleneck-channels 96 \
  --fullres-blocks 2 \
  --mid-blocks 2 \
  --bottleneck-blocks 4 \
  --short-frames 7 \
  --long-frames 30 \
  --amp \
  --grad-checkpoint \
  --preflight-only \
  2>&1 | tee "$RUN/preflight.log"
```
This must perform real T=30 native full-frame forward + Charbonnier + backward + optimizer step for each representative family/resolution.

If OOM occurs, DO NOT:
- crop
- resize
- lower T
- reduce channels/blocks
- move recurrence to lower resolution
- CPU offload silently

STOP with:
`HUMAN_ACTION_REQUIRED: YES — full-resolution recurrent T=30 native full-frame training does not fit available GPU memory.`

## Step 6 — training recipe
Only if preflight PASS:
```bash
python train_unet_fullres.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  --output-dir "$RUN/train" \
  --base-channels 48 \
  --mid-channels 64 \
  --bottleneck-channels 96 \
  --fullres-blocks 2 \
  --mid-blocks 2 \
  --bottleneck-blocks 4 \
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
  2>&1 | tee "$RUN/train.log"
```

Fixed recipe:
```text
step 1-50000: T=7
step 50001-150000: T=30
input: native full frame
batch: 1
loss: Charbonnier only
optimizer: Adam beta=(0.9,0.99)
LR: one continuous cosine 3e-4 -> 1e-7
clip_grad_norm: 0.5
optimizer/scheduler reset at 50k: NO
```

No edge/temporal/SSIM/perceptual/frequency/GAN/distillation losses.
Start from random initialization. Do not resume any prior RepVGG or low-res U-Net checkpoint.

## Step 7 — checkpoint comparison
Evaluate 50k / 75k / 100k / 125k / 150k using the same GoPro T=15 first-100-clips protocol:
```bash
for STEP in 0050000 0075000 0100000 0125000 0150000; do
  CKPT="$RUN/train/step_${STEP}.pth"
  test -f "$CKPT" || continue
  python eval_gopro_unet_fullres.py \
    --gopro-root "$GOPRO" \
    --checkpoint "$CKPT" \
    --num-frames 15 \
    --max-clips 100 \
    --fp16 \
    | tee "$RUN/eval_t15_${STEP}.txt"
done
```
Choose the best checkpoint by this same-protocol T15 PSNR.

Old reference: RepVGG Stage1@60k T15 = 29.9600 dB. Only report numerical gain if evaluation protocol is identical; otherwise `NOT_COMPARABLE`.

## Step 8 — matched-center context test
For the best checkpoint evaluate T7/T15/T30 with `--center-only`, same first 100 clips/anchors where possible. Report:
- CENTER_T7_PSNR
- CENTER_T15_PSNR
- CENTER_T30_PSNR
- T15 minus T7
- T30 minus T15

Do not infer context gains from unmatched target frames.

## Step 9 — business video
```bash
BEST=<best checkpoint>
python infer_video_unet_fullres.py \
  --input "$INPUT" \
  --checkpoint "$BEST" \
  --output "$RUN/business_fullres_unet.mp4" \
  --chunk 15 \
  --overlap 4 \
  --fp16 \
  2>&1 | tee "$RUN/business_infer.log"
```
If inference-only chunk=15 OOMs, reduce chunk to 9 then 7. Do not resize 1280x720 input.

```bash
ffmpeg -y -i "$INPUT" -i "$RUN/business_fullres_unet.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" -an "$RUN/input_vs_fullres_unet.mp4"
```

Human visual review is required.

## No extra experiments
Do not automatically add distillation, optical flow, new losses, quantization, pruning, model shrinking, or low-resolution recurrence. First establish this quality-first baseline.

## Final report
```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovsr-deblur-unet-fullframe-20260903
GITHUB_COMMIT: <actual latest sha>
RECIPE_ID: nanovsr_unet_fullres_recurrence_charbonnier_mix_v2
ARCHITECTURE: NanoVSRFullResUNetDeblur
RECURRENT_STATE: FULL_RESOLUTION
MODEL_CONFIG: C=48/64/96 blocks=2/2/4
GPU: <name>
PYTORCH: <version>
GOPRO_ROOT: <path>
DVD_ROOT: <path>
BSD_ROOT: <path>
PARAMS: <count / M>
MACS_PER_FRAME_640x360: <G>
MACS_PER_FRAME_1280x720: <G>
FLOPS_PER_FRAME_640x360_IF_2X_MAC: <G>
FLOPS_PER_FRAME_1280x720_IF_2X_MAC: <G>
PREFLIGHT_T30_FULLFRAME: PASS / FAIL
PREFLIGHT_PEAK_GPU_MEMORY: <details>
CHECKPOINT_T15_TABLE: <50k/75k/100k/125k/150k>
BEST_CHECKPOINT: <path>
BEST_CHECKPOINT_SHA256: <sha>
BEST_T15_PSNR: <dB>
CENTER_T7_PSNR: <dB>
CENTER_T15_PSNR: <dB>
CENTER_T30_PSNR: <dB>
GAIN_VS_OLD_REPVGG_T15: <dB or NOT_COMPARABLE>
BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE: <path>
BUSINESS_INFER_CHUNK: <N>
VISUAL_REVIEW_REQUIRED: YES
NOTES: <concise>
```
