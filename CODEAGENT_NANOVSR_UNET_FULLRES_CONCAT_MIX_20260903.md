# CODEAGENT TASK — NanoVSR-Deblur Full-Resolution CONCAT Recurrent U-Net, Native Full-Frame Mixed Training

## Goal
Train the quality-first video motion deblurring architecture `NanoVSRFullResConcatUNetDeblur`.

This is the updated architecture requested by the user. It is NOT the earlier low-resolution recurrence model and NOT the earlier full-resolution additive-state model.

The key design is:

```text
RGB frame x_t
   |
   v
Outer Image Encoder (full resolution CNN)
   |
   +--> image feature F_t: B x 48 x H x W

Forward direction:
   concat(F_t, H^f_{t-1}) at FULL H x W
          |
          v
   direction-specific fusion CNN (96 -> 48)
          |
          v
   full-resolution recurrent U-Net
      48@1x -> 64@1/2 -> 96@1/4 -> 64@1/2 -> 48@1x
          |
          v
   H^f_t: B x 48 x H x W

Backward direction mirrors the same pipeline with independent parameters.

Output:
   concat(F_t, H^f_t, H^b_t) at FULL H x W
          |
          v
   Outer CNN Decoder
          |
          v
   RGB residual
          |
          v
   output = input + residual
```

## Repository / branch
Repository:
`https://github.com/hihiok/video_motion_deblur.git`

Branch:
`agent/nanovsr-deblur-unet-fullframe-20260903`

Required files:
- `nanovsr_deblur/models/nanovsr_unet_fullres_concat_deblur.py`
- `nanovsr_deblur/train_unet_fullres_concat.py`
- `nanovsr_deblur/profile_unet_fullres_concat.py`
- `nanovsr_deblur/eval_gopro_unet_fullres_concat.py`
- `nanovsr_deblur/infer_video_unet_fullres_concat.py`
- `nanovsr_deblur/audit_fullframe_datasets.py`

## Non-negotiable architecture definition
1. Architecture must be `NanoVSRFullResConcatUNetDeblur`.
2. Outer image encoder is mandatory and runs before temporal recurrence.
3. Default image encoder is full-resolution `3->48 Conv + 2 ResidualConvBlocks + 48->48 Conv`.
4. Therefore image feature `F_t` is full-resolution `B x 48 x H x W`.
5. Recurrent hidden state is full-resolution `B x 48 x H x W` for both directions.
6. Image feature and previous/next hidden state MUST be concatenated at full resolution. Do NOT replace concat with addition.
7. Each direction has its own state-fusion CNN after concat.
8. Each direction then runs an independent recurrent U-Net whose input and output are full resolution.
9. U-Net internal downsampling to 1/2 and 1/4 resolution is allowed only inside the U-Net; the recurrent state itself remains full resolution.
10. Output stage MUST concat current image feature + forward hidden + backward hidden at full resolution.
11. Outer CNN decoder is mandatory after this 3-way concat.
12. No BatchNorm.
13. Default channels: encoder/recurrent full-res=48, U-Net half=64, bottleneck=96, output decoder hidden=64.
14. Default U-Net residual blocks: full-res=2, half-res=2, bottleneck=4.
15. Outer encoder blocks=2, state-fusion blocks=1, outer decoder blocks=2.

## Training definition
- Native full-frame input and target.
- No random crop.
- No resize/downscale.
- Batch size=1.
- Data: GoPro + DVD + BSD with family-balanced sampling.
- Loss: Charbonnier only.
- No edge loss, temporal loss, SSIM, perceptual, frequency, GAN, or distillation loss.
- Phase 1: T=7, steps 1-50000.
- Phase 2: T=30, steps 50001-150000.
- One Adam optimizer for all 150k steps, betas=(0.9,0.99).
- One cosine scheduler for all 150k steps, LR=3e-4 -> 1e-7.
- Gradient clipping=0.5.
- AMP + gradient checkpointing.
- Start from random initialization. Do not resume any previous RepVGG, additive-state U-Net, or low-resolution recurrence checkpoint.

## Proxy / SSL
Before Git operations:
```bash
git config --global http.sslVerify false
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
```
Do not print or commit proxy credentials.

## Paths
```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovsr_fullres_concat
RUN=$ROOT/runs/nanovsr_unet_fullres_concat_mix_20260903
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
INPUT=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
mkdir -p "$RUN"
```

## Step A — clone / sync
```bash
cd "$ROOT"
if [ ! -d "$REPO/.git" ]; then
  git clone -b agent/nanovsr-deblur-unet-fullframe-20260903 https://github.com/hihiok/video_motion_deblur.git "$REPO"
else
  cd "$REPO"
  git fetch origin
  git checkout agent/nanovsr-deblur-unet-fullframe-20260903
  git reset --hard origin/agent/nanovsr-deblur-unet-fullframe-20260903
fi
cd "$REPO"
git rev-parse HEAD
```
Record `GITHUB_COMMIT`.

## Step B — syntax/import audit
```bash
cd "$REPO/nanovsr_deblur"
python -m py_compile \
  models/nanovsr_unet_fullres_concat_deblur.py \
  train_unet_fullres_concat.py \
  profile_unet_fullres_concat.py \
  eval_gopro_unet_fullres_concat.py \
  infer_video_unet_fullres_concat.py \
  audit_fullframe_datasets.py
```
If this fails, only make minimal execution fixes. Do not redesign the architecture.

## Step C — dataset audit
Verify all roots exist, then run:
```bash
python audit_fullframe_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  2>&1 | tee "$RUN/dataset_audit.log"
```
Required:
- exact blur/GT filename alignment
- T=7 and T=30 windows
- native resolution report
- no dataset modification

## Step D — architecture assertions
Run a small Python sanity test and explicitly assert:
- image encoder output shape is `B,48,H,W`
- forward hidden shape is `B,48,H,W`
- backward hidden shape is `B,48,H,W`
- state fusion uses `torch.cat([image_feat, hidden], dim=1)` and therefore receives 96 channels before reduction
- output decoder receives `torch.cat([image_feat, fwd_hidden, bwd_hidden], dim=1)` and therefore receives 144 channels before reduction
- final RGB output H/W equals input H/W

If any of these are false, STOP and report. Do not silently substitute addition or reduced-resolution state.

## Step E — profile
```bash
python profile_unet_fullres_concat.py --height 360 --width 640 --frames 3 | tee "$RUN/profile_640x360.txt"
python profile_unet_fullres_concat.py --height 720 --width 1280 --frames 3 | tee "$RUN/profile_1280x720.txt"
```
Report params, MAC/frame, and 2xMAC FLOPs/frame at both resolutions.

Quality is the priority in this experiment. Do not shrink the model merely because these numbers are large.

## Step F — GPU selection
Use ONE GPU with the largest currently available free memory. Record model, total memory, free memory.

## Step G — mandatory T=30 native-full-frame forward+backward preflight
```bash
mkdir -p "$RUN/preflight"
python train_unet_fullres_concat.py \
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

For each native family/resolution representative, this must execute real T=30 forward + Charbonnier + backward + optimizer step.

If any resolution OOMs:
- do NOT crop
- do NOT resize
- do NOT lower T
- do NOT reduce channels/blocks
- do NOT change concat back to addition
- do NOT move hidden state to low resolution
- do NOT CPU offload silently

STOP with `HUMAN_ACTION_REQUIRED: YES` and report the failing family/resolution.

## Step H — full training
Only after preflight PASS:
```bash
mkdir -p "$RUN/train"
python train_unet_fullres_concat.py \
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

## Step I — checkpoint evaluation
Evaluate at least 50k, 75k, 100k, 125k, 150k using identical GoPro T=15, first 100 clips:
```bash
for STEP in 0050000 0075000 0100000 0125000 0150000; do
  CKPT="$RUN/train/step_${STEP}.pth"
  test -f "$CKPT" || continue
  python eval_gopro_unet_fullres_concat.py \
    --gopro-root "$GOPRO" \
    --checkpoint "$CKPT" \
    --num-frames 15 \
    --max-clips 100 \
    --fp16 \
    | tee "$RUN/train/eval_t15_${STEP}.txt"
done
```
Choose BEST_T15 by same-protocol PSNR.

Reference only: old RepVGG Stage1@60k + T15 = 29.9600 dB. Only compute gain if protocol is truly identical.

## Step J — context evaluation
For BEST_T15 evaluate T=7,15,30 and center-only T=7,15,30 on matched targets where possible.
Report context gains only from center-frame matched-target evaluation.

## Step K — business video inference
```bash
BEST=<best checkpoint>
python infer_video_unet_fullres_concat.py \
  --input "$INPUT" \
  --checkpoint "$BEST" \
  --output "$RUN/business_fullres_concat_unet.mp4" \
  --chunk 15 \
  --overlap 4 \
  --fp16 \
  2>&1 | tee "$RUN/business_infer.log"
```
If inference chunk=15 OOMs, reducing inference chunk only is allowed. Do not resize the 1280x720 input.

Create side-by-side with ffmpeg:
```bash
ffmpeg -y -i "$INPUT" -i "$RUN/business_fullres_concat_unet.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -an \
  "$RUN/input_vs_fullres_concat_unet.mp4"
```

## Do not perform extra experiments
Do not automatically distill, add losses, prune, quantize, reduce resolution, reduce width/depth, or change dataset mixture.

## Required final report
```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovsr-deblur-unet-fullframe-20260903
GITHUB_COMMIT: <sha>
RECIPE_ID: nanovsr_unet_fullres_concat_charbonnier_mix_v3
ARCHITECTURE: NanoVSRFullResConcatUNetDeblur
IMAGE_ENCODER: full-res CNN, 3->48 + 2 ResBlocks + 48->48
RECURRENT_STATE: FULL_RESOLUTION Bx48xHxW
STATE_FUSION: CONCAT(image_feature, hidden_state) at full resolution
FORWARD_BACKWARD_PARAMS_SHARED: NO
RECURRENT_UNET: 48@1x -> 64@1/2 -> 96@1/4 -> 64@1/2 -> 48@1x
OUTPUT_FUSION: CONCAT(image_feature, forward_hidden, backward_hidden) at full resolution
OUTPUT_DECODER: full-res CNN decoder
FULL_FRAME_TRAINING: YES
RANDOM_CROP: NO
RESIZE: NO
LOSS: Charbonnier only
PARAMS: <count/M>
MACS_PER_FRAME_640x360: <G>
MACS_PER_FRAME_1280x720: <G>
PREFLIGHT_T30_FULLFRAME: PASS / FAIL
PREFLIGHT_PEAK_GPU_MEMORY: <per family/resolution>
BEST_CHECKPOINT: <path>
BEST_T15_PSNR: <dB>
CENTER_T7_PSNR: <dB>
CENTER_T15_PSNR: <dB>
CENTER_T30_PSNR: <dB>
BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE: <path>
VISUAL_REVIEW_REQUIRED: YES
```
