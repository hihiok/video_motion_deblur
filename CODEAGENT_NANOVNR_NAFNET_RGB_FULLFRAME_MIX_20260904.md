# CODEAGENT TASK — NanoVNR NAFNet RGB, Exact Structure, Native Full-Frame Mixed Training

## Goal
Train a video motion-deblurring model whose network structure matches the supplied `network_nanovnr_nafnet_featprop.py` model, with exactly one requested architecture change:

- supplied model input feature extractor: `Conv2d(4, 12, 3, 1, 1)`
- this experiment: `Conv2d(3, 12, 3, 1, 1)` for RGB input

Do NOT add the previously explored deeper outer encoder/decoder, 48-channel hidden state, extra concat-fusion CNN, or another U-Net design.

The intended model is `NanoVNRNAFNetRGB` in:
`nanovsr_deblur/models/network_nanovnr_nafnet_rgb.py`

## Repository
Repository:
`https://github.com/hihiok/video_motion_deblur.git`

Branch:
`agent/nanovnr-nafnet-rgb-fullframe-20260904`

Primary files:
- `nanovsr_deblur/models/network_nanovnr_nafnet_rgb.py`
- `nanovsr_deblur/train_nanovnr_nafnet_rgb_fullframe.py`
- `nanovsr_deblur/profile_nanovnr_nafnet_rgb.py`
- `nanovsr_deblur/eval_gopro_nanovnr_nafnet_rgb.py`
- `nanovsr_deblur/infer_video_nanovnr_nafnet_rgb.py`
- `nanovsr_deblur/audit_fullframe_datasets.py`

## Non-negotiable model structure
The model must match these properties exactly.

### RGB feature extraction
```text
RGB input frame [B,3,H,W]
  -> Conv2d(3,12,3,1,1)
  -> cur_feat [B,12,H,W]
```

The only architecture difference from the supplied Python model is input channels 4 -> 3.

### Bidirectional temporal propagation
Forward and backward propagation use separate `NAFUNetPropagationDefineChannel` instances.

At every time step:
```text
cur_feat  [B,12,H,W]
prop_feat [B,12,H,W]
      |
      v
concat along channel
      |
      v
[B,24,H,W]
```

The concat MUST occur at native full spatial resolution.
Do NOT replace concat with addition.

### NAF U-Net propagation
Fixed propagation channels:
```text
24 -> 32 -> 48 -> 72
```

Fixed blocks:
```text
enc_blk_nums = [1,1,1]
middle_blk_num = 1
dec_blk_nums = [1,1,1]
```

Each NAF block must retain the supplied implementation:
- `ChannelRowLayerNorm`
- 3x3 Conv
- PReLU
- 1x1 Conv
- beta residual scale
- second ChannelRowLayerNorm
- 1x1 Conv
- PReLU
- 1x1 Conv
- gamma residual scale
- no SCA

Downsampling must remain:
```text
Conv2d(kernel=2, stride=2)
```

There are three downsampling levels, so the U-Net internally reaches 1/8 spatial scale.

Upsampling must remain:
```text
1x1 Conv -> PixelShuffle(2)
```

Encoder/decoder skip fusion must remain ADDITION:
```text
x = x + enc_skip
```

Final propagation output must remain:
```text
feature_out: Conv2d(..., 12, 3,1,1)
prop hidden: [B,12,H,W]
```

### Final output fusion
At each frame:
```text
forward hidden  [B,12,H,W]
backward hidden [B,12,H,W]
      |
      v
concat -> [B,24,H,W]
      |
      v
1x1 Conv 24 -> 12
      |
      v
3x3 Conv 12 -> 3
      |
      v
RGB residual
      |
      v
output = input RGB + residual
```

Do NOT add an extra image-feature branch to the output decoder.
Do NOT add a deeper outer decoder.
Do NOT change hidden width from 12.

### Forward-state carry
Preserve the supplied model interface:
```python
forward(x, prev_forward_feat=None)
```

The returned `next_forward_feat` is the last forward propagation hidden state and may be used by the next non-overlapping inference chunk.
Backward propagation resets per chunk, matching the supplied model behavior.

### Gradient checkpointing
`grad_checkpoint` is allowed only as an execution/memory optimization during training. It must not change layers, parameters, tensor math, or inference outputs.

## Training resolution policy — native full frame
This experiment must train on the complete native frame.

Rules:
- no random crop
- no center crop
- no patch training
- no resize
- no forced 720p
- no forced 1080p
- no aspect-ratio conversion

Use each sequence at its dataset-native resolution.
Examples:
- if GoPro/DVD sequence is 1280x720: train full 1280x720
- if a sequence is 1920x1080: train full 1920x1080
- if BSD sequence is 640x480: train full 640x480

The actual resolutions MUST be discovered from the server dataset audit before training. Do not assume them.

Because mixed datasets can have different native resolutions, batch size is fixed to 1.

## Datasets
```bash
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
```

Use family-balanced sampling across:
```text
GoPro : DVD : BSD ~= 1 : 1 : 1
```
regardless of the raw number of sliding windows in each family.

Do not modify, rename, move, or regenerate existing dataset images.

BSD official layouts such as:
```text
<config>/train/<seq>/Blur/RGB/*.png
<config>/train/<seq>/Sharp/RGB/*.png
```
are supported by the existing mixed dataset loader.

## Training recipe
Keep the clean NanoVSR-style temporal curriculum already selected for this project:

Phase 1:
```text
T=7
steps 1-50000
```

Phase 2:
```text
T=30
steps 50001-150000
```

Loss for both phases:
```text
Charbonnier only
```

Do NOT use:
- edge loss
- temporal loss
- SSIM loss
- perceptual loss
- frequency loss
- GAN loss
- distillation

Optimizer:
```text
Adam betas=(0.9,0.99)
```

Schedule:
```text
one CosineAnnealing schedule across all 150000 steps
lr 3e-4 -> 1e-7
```

At step 50000 -> 50001:
- switch T=7 -> T=30
- do NOT reset optimizer
- do NOT reset scheduler

Gradient clip:
```text
0.5
```

Training execution:
- batch=1
- AMP enabled
- gradient checkpointing enabled
- workers <=2
- random initialization

Do NOT resume any previous RepVGG/U-Net checkpoint.

## Proxy / SSL
The server is behind an internal HTTPS inspection proxy.

Before Git operations:
```bash
git config --global http.sslVerify false
```

Use any existing proxy environment without printing credentials:
```bash
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
```

Never print or commit proxy usernames/passwords.

## Paths
```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovnr_nafnet_rgb
RUN=$ROOT/runs/nanovnr_nafnet_rgb_fullframe_mix_20260904

GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
INPUT=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4

mkdir -p "$RUN"
```

## Step A — clone/sync exact branch
```bash
cd "$ROOT"
if [ ! -d "$REPO/.git" ]; then
  git clone -b agent/nanovnr-nafnet-rgb-fullframe-20260904 \
    https://github.com/hihiok/video_motion_deblur.git "$REPO"
else
  cd "$REPO"
  git fetch origin
  git checkout agent/nanovnr-nafnet-rgb-fullframe-20260904
  git reset --hard origin/agent/nanovnr-nafnet-rgb-fullframe-20260904
fi
cd "$REPO"
git rev-parse HEAD
```

Record `GITHUB_COMMIT`.

## Step B — environment and syntax audit
Prefer an existing PyTorch/CUDA environment already proven to run NanoVSR-Deblur on this server.

```bash
cd "$REPO/nanovsr_deblur"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free,total = torch.cuda.mem_get_info(i)
        print(i, torch.cuda.get_device_name(i), 'free_GiB', free/2**30, 'total_GiB', total/2**30)
PY

python -m py_compile \
  models/network_nanovnr_nafnet_rgb.py \
  train_nanovnr_nafnet_rgb_fullframe.py \
  profile_nanovnr_nafnet_rgb.py \
  eval_gopro_nanovnr_nafnet_rgb.py \
  infer_video_nanovnr_nafnet_rgb.py \
  audit_fullframe_datasets.py
```

If syntax/import fails, only make the minimum execution fix needed. Do NOT redesign the architecture.

## Step C — dataset audit and actual native resolutions
Verify roots exist:
```bash
for p in "$GOPRO" "$DVD" "$BSD"; do
  test -d "$p" || { echo "MISSING_DATASET=$p"; exit 2; }
done
```

Run:
```bash
python audit_fullframe_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  2>&1 | tee "$RUN/dataset_audit.log"
```

Required report:
- GoPro native resolution(s)
- DVD native resolution(s)
- BSD native resolution(s)
- T=7 window counts by family
- T=30 window counts by family
- exact blur/GT shape match
- no crop
- no resize

If any family cannot produce aligned T=30 windows, STOP and report the exact path/reason.

## Step D — architecture verification
Run explicit Python assertions before any training.

Required assertions:
```text
feat_extract.in_channels == 3
feat_extract.out_channels == 12
forward_net is NAFUNetPropagationDefineChannel
backward_net is NAFUNetPropagationDefineChannel
forward_net.prop_channels == [24,32,48,72]
backward_net.prop_channels == [24,32,48,72]
fusion.in_channels == 24
fusion.out_channels == 12
conv_last.in_channels == 12
conv_last.out_channels == 3
```

Run a small RGB tensor through the model and confirm:
```text
input  = [1,T,3,H,W]
output = [1,T,3,H,W]
next_forward_feat = [1,12,H,W]
```

Also inspect source and verify propagation uses:
```python
torch.cat([cur_feat, prop_feat], dim=1)
```
at full resolution.

Verify U-Net skip uses addition, not concat:
```python
x = x + enc_skip
```

If any assertion fails, STOP. Do not substitute another architecture.

## Step E — parameter and MAC profile
Profile at three standard sizes for reference:
```bash
python profile_nanovnr_nafnet_rgb.py --height 360 --width 640 --frames 3 \
  | tee "$RUN/profile_640x360.txt"

python profile_nanovnr_nafnet_rgb.py --height 720 --width 1280 --frames 3 \
  | tee "$RUN/profile_1280x720.txt"

python profile_nanovnr_nafnet_rgb.py --height 1080 --width 1920 --frames 3 \
  | tee "$RUN/profile_1920x1080.txt"
```

Report:
- PARAMS
- PARAMS_M
- MACS_PER_FRAME_G at 640x360
- MACS_PER_FRAME_G at 1280x720
- MACS_PER_FRAME_G at 1920x1080
- FLOPS/frame if using 2 FLOP/MAC convention

Do not alter the network based on compute results. This experiment prioritizes effect first.

## Step F — choose GPU
Use ONE GPU with the largest currently available free memory.
Record:
- GPU name
- total memory
- free memory before preflight

```bash
export CUDA_VISIBLE_DEVICES=<selected_gpu>
```
Inside Python use `cuda:0`.

## Step G — mandatory native full-frame T=30 training preflight
This is mandatory before full training.

```bash
mkdir -p "$RUN/preflight"
python train_nanovnr_nafnet_rgb_fullframe.py \
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

For every observed family/native-resolution representative, preflight must execute real:
- native full-frame T=30 load
- forward
- Charbonnier loss
- backward
- optimizer step

If any native resolution OOMs:
- do NOT crop
- do NOT resize
- do NOT change 1080p to 720p
- do NOT lower T automatically
- do NOT change num_feat=12
- do NOT alter propagation channels
- do NOT replace NAFBlock
- do NOT change full-resolution hidden concat

STOP and return:
```text
HUMAN_ACTION_REQUIRED: YES
NATIVE_FULLFRAME_T30_OOM: <family resolution GPU peak-memory>
```

## Step H — full training
Only execute after Step G PASS.

```bash
mkdir -p "$RUN/train"
python train_nanovnr_nafnet_rgb_fullframe.py \
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
  2>&1 | tee "$RUN/train/train.log"
```

Training must remain:
```text
steps 1-50000: T=7
steps 50001-150000: T=30
native full frame
batch=1
Charbonnier only
Adam betas=(0.9,0.99)
one cosine schedule
no optimizer/scheduler reset at phase switch
```

## Step I — checkpoint evaluation
At minimum evaluate:
- 50k
- 75k
- 100k
- 125k
- 150k

Use same GoPro protocol for all checkpoints:
```text
T=15
first 100 clips
native full frame
RGB PSNR
FP16 inference
```

Commands:
```bash
for STEP in 0050000 0075000 0100000 0125000 0150000; do
  CKPT="$RUN/train/step_${STEP}.pth"
  test -f "$CKPT" || continue
  python eval_gopro_nanovnr_nafnet_rgb.py \
    --gopro-root "$GOPRO" \
    --checkpoint "$CKPT" \
    --num-frames 15 \
    --max-clips 100 \
    --fp16 \
    | tee "$RUN/train/eval_t15_${STEP}.txt"
done
```

Select `BEST_T15` by highest same-protocol RGB PSNR.

Reference only:
```text
old RepVGG NanoVSR-Deblur Stage1@60k + T15 = 29.9600 dB
```
Only calculate gain if the evaluation protocol is confirmed identical.

## Step J — temporal-context evaluation
For BEST_T15, evaluate T=7, T=15, T=30.
Also run center-only mode for matched target-frame context comparison.

Report:
```text
BEST_T7_PSNR
BEST_T15_PSNR
BEST_T30_PSNR
CENTER_T7_PSNR
CENTER_T15_PSNR
CENTER_T30_PSNR
CENTER_CONTEXT_GAIN_T15_VS_T7
CENTER_CONTEXT_GAIN_T30_VS_T15
```

## Step K — business video inference
Input:
```text
/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Use BEST checkpoint at native 1280x720. Start with non-overlapping chunk=15:
```bash
BEST=<best checkpoint path>
python infer_video_nanovnr_nafnet_rgb.py \
  --input "$INPUT" \
  --checkpoint "$BEST" \
  --output "$RUN/business_nanovnr_nafnet_rgb.mp4" \
  --chunk 15 \
  --fp16 \
  2>&1 | tee "$RUN/business_infer.log"
```

The inference script must carry `next_forward_feat` into the next non-overlapping chunk as `prev_forward_feat`, matching the supplied model interface.
Backward state resets inside every chunk.

If chunk=15 inference OOMs, reducing inference chunk only is allowed, e.g. 9 or 7. Do not resize the video.

Create side-by-side:
```bash
ffmpeg -y \
  -i "$INPUT" \
  -i "$RUN/business_nanovnr_nafnet_rgb.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" -an \
  "$RUN/input_vs_nanovnr_nafnet_rgb.mp4"
```

Visually inspect:
- deblur strength
- face deformation
- text detail
- ringing/oversharpening
- ghosting
- temporal flicker
- chunk-boundary discontinuity
- color shift

Do not claim subjective success without visual inspection.

## Step L — no extra experiments
Do NOT automatically:
- add extra CNN encoder/decoder
- increase hidden width
- modify NAFBlock
- add SCA
- add optical flow
- add temporal loss
- distill
- prune
- quantize
- crop or resize training data

This task establishes the exact supplied model's RGB/full-frame deblur baseline first.

## Required final report
Return:
```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovnr-nafnet-rgb-fullframe-20260904
GITHUB_COMMIT: <sha>
RECIPE_ID: nanovnr_nafnet_rgb_native_fullframe_mix_v1
ARCHITECTURE: NanoVNRNAFNetRGB
MODEL_DIFF_VS_SUPPLIED: input channels 4->3 only
NUM_FEAT: 12
PROP_CHANNELS: 24,32,48,72
GOPRO_NATIVE_RESOLUTIONS: <actual audit>
DVD_NATIVE_RESOLUTIONS: <actual audit>
BSD_NATIVE_RESOLUTIONS: <actual audit>
TRAINING_RESOLUTION_POLICY: NATIVE_FULL_FRAME_NO_CROP_NO_RESIZE
GPU: <name/memory>
PYTORCH: <version>
PARAMS: <count/M>
MACS_PER_FRAME_640x360: <G>
MACS_PER_FRAME_1280x720: <G>
MACS_PER_FRAME_1920x1080: <G>
PREFLIGHT_T30_NATIVE_FULLFRAME: PASS / FAIL
PREFLIGHT_PEAK_MEMORY_BY_RESOLUTION: <values>
TRAIN_PHASE1: T7 steps 1-50000
TRAIN_PHASE2: T30 steps 50001-150000
LOSS: Charbonnier only
PSNR_50K_T15: <dB>
PSNR_75K_T15: <dB>
PSNR_100K_T15: <dB>
PSNR_125K_T15: <dB>
PSNR_150K_T15: <dB>
BEST_CHECKPOINT: <path>
BEST_T15_PSNR: <dB>
BEST_T7_PSNR: <dB>
BEST_T30_PSNR: <dB>
CENTER_T7_PSNR: <dB>
CENTER_T15_PSNR: <dB>
CENTER_T30_PSNR: <dB>
GAIN_VS_OLD_REPVGG_T15: <dB or NOT_COMPARABLE>
BUSINESS_OUTPUT: <path>
BUSINESS_SIDE_BY_SIDE: <path>
BUSINESS_INFERENCE_CHUNK: <N>
FORWARD_STATE_CARRY_ACROSS_CHUNKS: YES
VISUAL_REVIEW: <brief factual observations>
```
