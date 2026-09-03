# CODEAGENT TASK — NanoVSR-Deblur Full-Resolution Recurrent U-Net, Native Full-Frame Mixed Training

## Goal
Train a quality-first video motion deblurring model derived from NanoVSR's bidirectional recurrent idea, but replace the old RepVGG propagation with a U-Net recurrent update and keep the recurrent hidden state at FULL spatial resolution.

This experiment prioritizes restoration quality first. Do NOT move recurrence to 1/2 or 1/4 resolution to save compute.

Repository:
`https://github.com/hihiok/video_motion_deblur.git`

Branch:
`agent/nanovsr-deblur-unet-fullframe-20260903`

Expected branch HEAD at task creation:
`0cd0776887f9a85eb658a3c025b116700e8cf5ab`

Primary files:
- `nanovsr_deblur/models/nanovsr_unet_fullres_deblur.py`
- `nanovsr_deblur/train_unet_fullres.py`
- `nanovsr_deblur/profile_unet_fullres.py`
- `nanovsr_deblur/eval_gopro_unet_fullres.py`
- `nanovsr_deblur/infer_video_unet_fullres.py`
- `nanovsr_deblur/audit_fullframe_datasets.py`

## Non-negotiable experiment definition
1. Architecture must be `NanoVSRFullResUNetDeblur`.
2. Recurrent state must remain full resolution: `B x C x H x W` for every forward/backward time step.
3. Each recurrent update is a complete U-Net whose INPUT and OUTPUT are full-resolution feature maps.
4. Internal U-Net down/up-sampling is allowed only as a spatial multi-scale branch. The recurrent state itself must NOT live at reduced resolution.
5. Default channels: `48 -> 64 -> 96`.
6. Default U-Net residual blocks: full-res=2, half-res=2, bottleneck=4.
7. Forward and backward recurrent U-Nets use separate parameters.
8. No BatchNorm.
9. Native full-frame training only.
10. No random crop.
11. No resize/downscale.
12. Batch size = 1.
13. Dataset mixture: GoPro + DVD + BSD with family-balanced sampling.
14. Loss: Charbonnier only.
15. No edge loss, temporal loss, SSIM loss, perceptual loss, frequency loss, GAN loss, or distillation.
16. Phase 1: T=7, steps 1-50000.
17. Phase 2: T=30, steps 50001-150000.
18. One continuous Adam optimizer and one continuous cosine LR schedule across both phases.
19. Start from random initialization. Do NOT resume previous RepVGG or low-resolution-recurrence checkpoints.
20. Use AMP and gradient checkpointing for training.

## Proxy / SSL
The server is behind an internal HTTPS-inspection proxy.

Before Git operations:
```bash
git config --global http.sslVerify false
```

Load the existing proxy environment if present, without printing credentials:
```bash
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
```

Do NOT print, commit, or expose proxy usernames/passwords.

## Paths
```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovsr_fullres_unet
RUN=$ROOT/runs/nanovsr_unet_fullres_mix_20260903

GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD

INPUT=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
mkdir -p "$RUN"
```

## Step A — clone / sync exact branch
```bash
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

Report `GITHUB_COMMIT`.

## Step B — environment and syntax audit
Prefer an existing CUDA/PyTorch environment already working for the previous NanoVSR-Deblur experiments. Do not rebuild a huge environment unless necessary.

```bash
cd "$REPO/nanovsr_deblur"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i))
PY

python -m py_compile \
  models/nanovsr_unet_fullres_deblur.py \
  train_unet_fullres.py \
  profile_unet_fullres.py \
  eval_gopro_unet_fullres.py \
  infer_video_unet_fullres.py \
  audit_fullframe_datasets.py
```

If syntax/import fails, report the exact error. Only make a minimal bug fix if required to execute the committed design. Do not redesign the model.

## Step C — dataset audit, no modification
Verify all roots exist:
```bash
for p in "$GOPRO" "$DVD" "$BSD"; do
  test -d "$p" || { echo "MISSING_DATASET=$p"; exit 2; }
done
```

Run the full-frame dataset audit:
```bash
python audit_fullframe_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  2>&1 | tee "$RUN/dataset_audit.log"
```

Required checks:
- exact blur/GT filename pairing
- no index-only silent matching
- T=7 windows available
- T=30 windows available
- report native resolutions by family/config
- do not rename/move/modify any dataset file

BSD official layout such as:
`<config>/train/<seq>/Blur/RGB/*.png`
paired with
`<config>/train/<seq>/Sharp/RGB/*.png`
is expected and supported.

If any family cannot produce T=30 aligned windows, STOP and report the exact path/problem.

## Step D — architecture sanity
The intended model is:
```text
RGB frame
  -> full-resolution 48ch feature
  -> add previous full-resolution recurrent state
  -> full-resolution recurrent U-Net update
       full-res 48ch
       1/2-res 64ch
       1/4-res 96ch
       decode back to full-res 48ch
  -> new full-resolution recurrent state
```

This is run independently in forward and backward directions.

Assert with a small tensor that returned recurrent/output spatial shape equals input HxW. Do not alter the architecture to move hidden state to low resolution.

## Step E — parameter and compute profile
Run at both 640x360 and 1280x720. Use T=3 only for profiler practicality; divide total MACs by T as the script does.

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

Report:
- PARAMS
- PARAMS_M
- MACS_PER_FRAME_G at 640x360
- MACS_PER_FRAME_G at 1280x720
- FLOPS_PER_FRAME_G_IF_2X_MAC at both resolutions

Do NOT compare these values directly to old RepVGG numbers unless units are identical.

## Step F — choose GPU
Use ONE GPU with the largest currently available free memory. Do not occupy multiple GPUs just to increase batch size.

Record:
- GPU model
- total memory
- free memory before test

Set:
```bash
export CUDA_VISIBLE_DEVICES=<selected_gpu>
```
Then inside Python use `cuda:0`.

## Step G — mandatory native full-frame T=30 forward+backward preflight
This is required before any long training.

```bash
mkdir -p "$RUN/preflight"
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
  2>&1 | tee "$RUN/preflight/preflight.log"
```

The preflight must execute real:
- native full-frame load
- T=30
- forward
- Charbonnier loss
- backward
- optimizer step
for every observed family/resolution representative.

If any native resolution OOMs:
- DO NOT crop
- DO NOT resize
- DO NOT use T=15 instead
- DO NOT lower channels
- DO NOT lower blocks
- DO NOT move recurrence to 1/2 or 1/4 scale
- DO NOT silently use CPU offload

STOP and return:
`HUMAN_ACTION_REQUIRED: YES — native full-frame T=30 full-resolution recurrent U-Net does not fit available GPU memory.`

Also report which family/resolution OOMed and peak memory reached if available.

## Step H — full training
Only execute if Step G PASS.

```bash
mkdir -p "$RUN/train"
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
  2>&1 | tee "$RUN/train/train.log"
```

Training recipe must remain:
- steps 1-50000: T=7
- steps 50001-150000: T=30
- native full frame
- batch=1
- Charbonnier only
- Adam betas=(0.9,0.99)
- one cosine schedule 3e-4 -> 1e-7 across full 150k
- grad clip=0.5
- optimizer/scheduler NOT reset at 50k

## Step I — checkpoint evaluation
At minimum evaluate:
- 50k
- 75k
- 100k
- 125k
- 150k

Use GoPro test, T=15, same first 100 clips, same script/settings for all checkpoints:
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
    | tee "$RUN/train/eval_t15_${STEP}.txt"
done
```

Choose `BEST_T15` by the highest same-protocol GoPro RGB PSNR.

Reference only:
Old RepVGG Stage1@60k + T15 = 29.9600 dB.

Only compute `GAIN_VS_OLD_REPVGG_T15` if you verify evaluation protocol is truly the same. Otherwise label it `NOT_COMPARABLE`.

## Step J — context evaluation of best checkpoint
For BEST_T15 run:
- T=7
- T=15
- T=30

Use same GoPro evaluation protocol and same first 100 clips where possible.
Also run center-only mode for T7/T15/T30 so context comparison targets the center frame.

Report:
- BEST_T7_PSNR
- BEST_T15_PSNR
- BEST_T30_PSNR
- CENTER_T7_PSNR
- CENTER_T15_PSNR
- CENTER_T30_PSNR
- CENTER_CONTEXT_GAIN_T15_VS_T7
- CENTER_CONTEXT_GAIN_T30_VS_T15

Do not claim context gain from non-matched target frames.

## Step K — business video inference
Input:
`/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4`

Run native 1280x720 inference with BEST checkpoint. Start with chunk=15:
```bash
BEST=<best checkpoint path>
python infer_video_unet_fullres.py \
  --input "$INPUT" \
  --checkpoint "$BEST" \
  --output "$RUN/business_fullres_unet.mp4" \
  --chunk 15 \
  --overlap 4 \
  --fp16 \
  2>&1 | tee "$RUN/business_infer.log"
```

If inference chunk=15 OOMs, it is allowed to reduce inference chunk length only (e.g. 9, then 7) because this does not alter trained weights. Report the final chunk used. Do not resize the 1280x720 business video.

Create side-by-side:
```bash
ffmpeg -y -i "$INPUT" -i "$RUN/business_fullres_unet.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" -an "$RUN/input_vs_fullres_unet.mp4"
```

Visual review is required. Do not claim subjective success automatically.

## Step L — do not perform extra experiments
Do NOT automatically:
- distill from BSSTNet/Shift-Net
- add new losses
- train a lower-resolution recurrent variant
- prune/quantize
- reduce model width/depth
- add optical flow
- change to another dataset mixture

This task is specifically to establish the quality upper baseline of the full-resolution recurrent U-Net.

## Required final report
Return:
```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovsr-deblur-unet-fullframe-20260903
GITHUB_COMMIT: <sha>
RECIPE_ID: nanovsr_unet_fullres_recurrence_charbonnier_mix_v2
ARCHITECTURE: NanoVSRFullResUNetDeblur
RECURRENT_STATE: FULL_RESOLUTION
MODEL_CONFIG: C=48/64/96 blocks=2/2/4
GOPRO_ROOT: /mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD_ROOT: /mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD_ROOT: /mnt/ssd1/z00919662/datasets/BSD
GPU: <name>
PYTORCH: <version>
PARAMS: <count and M>
MACS_PER_FRAME_640x360: <G>
MACS_PER_FRAME_1280x720: <G>
FLOPS_PER_FRAME_640x360_IF_2X_MAC: <G>
FLOPS_PER_FRAME_1280x720_IF_2X_MAC: <G>
PREFLIGHT_T30_FULLFRAME: PASS / FAIL
PREFLIGHT_PEAK_GPU_MEMORY: <GiB by family/resolution>
TRAIN_PHASE1: T7 steps 1-50000
TRAIN_PHASE2: T30 steps 50001-150000
LOSS: Charbonnier only
CHECKPOINT_T15_TABLE:
  50k: <dB>
  75k: <dB>
  100k: <dB>
  125k: <dB>
  150k: <dB>
BEST_CHECKPOINT: <path>
BEST_CHECKPOINT_SHA256: <sha>
BEST_T7_PSNR: <dB>
BEST_T15_PSNR: <dB>
BEST_T30_PSNR: <dB>
CENTER_T7_PSNR: <dB>
CENTER_T15_PSNR: <dB>
CENTER_T30_PSNR: <dB>
GAIN_VS_OLD_REPVGG_T15: <dB or NOT_COMPARABLE>
BUSINESS_INPUT: <path>
BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE: <path>
BUSINESS_INFER_CHUNK: <N>
VISUAL_REVIEW_REQUIRED: YES
NOTES: <concise>
```
