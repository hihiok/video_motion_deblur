# CODEAGENT TASK — NanoVSR-Deblur on GoPro with optional distillation

## Goal
Train a very small video motion deblurring model derived from NanoVSR's core architecture, but for 1x restoration instead of 4x VSR. Final deployment model must remain small and self-contained: bidirectional additive recurrent propagation + RepVGG-style conv blocks + 1x residual RGB head. No PixelShuffle in the final deblur model.

Branch to use:
`agent/nanovsr-deblur-gopro-distill-20260902`

Repository:
`https://github.com/hihiok/video_motion_deblur.git`

Primary code directory:
`nanovsr_deblur/`

## Important execution rules
1. DO NOT rewrite or redesign the model code unless a real runtime bug is found.
2. If you must modify code, stop and report the exact diff first. Prefer reporting the problem back to the user rather than creating a new implementation.
3. Keep CPU use conservative: DataLoader workers <= 2. Do not launch many background jobs.
4. Use one GPU unless explicitly instructed otherwise.
5. Never delete existing motion_deblur checkpoints/runs.
6. Final model must be the NanoVSR-Deblur student only. Teacher models are training-only and must not be needed for inference.
7. Record git commit, dataset path, dataset layout, checkpoint SHA256, parameter count, MACs/FLOPs, PSNR, runtime, GPU memory, and all commands.

## Proxy / SSL
This server is behind an internal HTTPS-inspection proxy. Before `git clone`, `git pull`, pip/wget/curl, load the user's existing proxy environment or set the proxy locally from the server's approved credentials. Do NOT commit proxy usernames/passwords to GitHub or logs.

At minimum run before GitHub operations:
```bash
git config --global http.sslVerify false
```

If proxy variables are already configured in shell/conda activation, preserve them:
```bash
env | grep -i proxy || true
```

If Git proxy configuration is required, use the server-local approved proxy URL from the user's environment. Do not invent or print credentials.

## Expected workspace
Use:
```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovsr_deblur
RUN=$ROOT/runs/nanovsr_deblur_gopro_20260902
mkdir -p "$RUN"
```

Clone/pull exactly this branch:
```bash
cd "$ROOT"
if [ ! -d "$REPO/.git" ]; then
  git clone -b agent/nanovsr-deblur-gopro-distill-20260902 https://github.com/hihiok/video_motion_deblur.git "$REPO"
else
  cd "$REPO"
  git fetch origin
  git checkout agent/nanovsr-deblur-gopro-distill-20260902
  git reset --hard origin/agent/nanovsr-deblur-gopro-distill-20260902
fi
cd "$REPO"
git rev-parse HEAD
```

## Phase A — environment and dataset audit
Prefer an existing PyTorch CUDA environment that already works on this server. Do not create a huge new environment if unnecessary.

```bash
cd "$REPO/nanovsr_deblur"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu', torch.cuda.get_device_name(0))
PY
pip install -r requirements.txt
```

Find the GoPro deblurring dataset. Accept any of these layouts:
- `<GOPRO>/train/blur` + `<GOPRO>/train/sharp`
- `<GOPRO>/train/blur` + `<GOPRO>/train/GT`
- same under `test/`

Search likely roots without scanning the whole filesystem:
```bash
for p in \
  /mnt/ssd1/z00919662/motion_deblur/datasets/GoPro \
  /mnt/ssd1/z00919662/motion_deblur/dataset/GoPro \
  /mnt/ssd1/z00919662/datasets/GoPro \
  /data/pub1/z00919662/motion_deblur/datasets/GoPro \
  /data/pub1/z00919662/dataset/GoPro; do
  [ -d "$p" ] && echo "FOUND_GOPRO=$p"
done
```

If GoPro is not present, stop with:
`HUMAN_ACTION_REQUIRED: YES — GoPro dataset path is missing; user must provide or upload it.`
Do not silently download hundreds of GB.

Set:
```bash
GOPRO=<FOUND_PATH>
```

Audit counts and image resolution before training.

## Phase B — model sanity/profile
The intended baseline is `num_feat=48`, `num_blocks=12`.

Run:
```bash
cd "$REPO/nanovsr_deblur"
python profile_model.py --num-feat 48 --num-blocks 12 --height 360 --width 640 --frames 7 | tee "$RUN/profile_48x12.txt"
```

Also profile two smaller variants for comparison only:
```bash
python profile_model.py --num-feat 32 --num-blocks 8 --height 360 --width 640 --frames 7 | tee "$RUN/profile_32x8.txt"
python profile_model.py --num-feat 40 --num-blocks 10 --height 360 --width 640 --frames 7 | tee "$RUN/profile_40x10.txt"
```

Do NOT choose a larger final model than 48x12 unless the user explicitly approves it.

## Phase C — Stage 1 supervised GoPro training
Purpose: first learn strong blur->sharp reconstruction without teacher bias.
Use short clips so training is stable and fast.

```bash
mkdir -p "$RUN/stage1"
CUDA_VISIBLE_DEVICES=0 python train_gopro.py \
  --gopro-root "$GOPRO" \
  --output-dir "$RUN/stage1" \
  --num-feat 48 \
  --num-blocks 12 \
  --num-frames 7 \
  --patch-size 256 \
  --batch-size 2 \
  --workers 2 \
  --steps 60000 \
  --lr 3e-4 \
  --lambda-edge 0.05 \
  --lambda-temp 0.05 \
  --lambda-distill 0 \
  --amp \
  --save-every 5000 \
  2>&1 | tee "$RUN/stage1/train.log"
```

If OOM occurs, first reduce batch size to 1. Do not reduce model size before trying batch size 1.

Evaluate:
```bash
python eval_gopro.py --gopro-root "$GOPRO" --checkpoint "$RUN/stage1/latest.pth" --num-frames 7 | tee "$RUN/stage1/eval.txt"
```

## Phase D — Stage 2 long-clip temporal refinement
Purpose: improve true video consistency and exploit the recurrent structure.
Resume Stage 1, use T=15 and stronger temporal-delta supervision.

Important: `train_gopro.py` stores optimizer state. Resume from Stage 1 but train to absolute step 120000.

```bash
mkdir -p "$RUN/stage2"
CUDA_VISIBLE_DEVICES=0 python train_gopro.py \
  --gopro-root "$GOPRO" \
  --output-dir "$RUN/stage2" \
  --resume "$RUN/stage1/latest.pth" \
  --num-feat 48 \
  --num-blocks 12 \
  --num-frames 15 \
  --patch-size 256 \
  --batch-size 1 \
  --workers 2 \
  --steps 120000 \
  --lr 1.5e-4 \
  --lambda-edge 0.08 \
  --lambda-temp 0.15 \
  --lambda-distill 0 \
  --amp \
  --save-every 5000 \
  2>&1 | tee "$RUN/stage2/train.log"
```

Evaluate both T=7 and T=15:
```bash
python eval_gopro.py --gopro-root "$GOPRO" --checkpoint "$RUN/stage2/latest.pth" --num-frames 7  | tee "$RUN/stage2/eval_t7.txt"
python eval_gopro.py --gopro-root "$GOPRO" --checkpoint "$RUN/stage2/latest.pth" --num-frames 15 | tee "$RUN/stage2/eval_t15.txt"
```

## Phase E — optional teacher distillation
Only do this if a stronger teacher output cache already exists or can be generated using an EXISTING, already-validated inference script in this repository/workspace. Preferred teacher order:
1. BSSTNet if its existing validated GoPro inference is available and clearly stronger.
2. Shift-Net+ if validated.
3. Shift-Net Ours-s only if no stronger teacher is available.

Do NOT let CodeAgent invent a new teacher implementation.

Teacher cache layout required by our dataset loader:
`<TEACHER_ROOT>/train/<sequence>/<same frame filename>`

Before using a teacher cache, verify all three:
- same frame count as GoPro train blur/GT
- same spatial resolution
- random 20-frame filename alignment check

If no validated teacher cache exists, SKIP Phase E. Stage 2 is still a valid final model.

If a validated teacher cache exists:
```bash
TEACHER_ROOT=<validated_cache_root>
mkdir -p "$RUN/stage3_distill"
CUDA_VISIBLE_DEVICES=0 python train_gopro.py \
  --gopro-root "$GOPRO" \
  --teacher-root "$TEACHER_ROOT" \
  --output-dir "$RUN/stage3_distill" \
  --resume "$RUN/stage2/latest.pth" \
  --num-feat 48 \
  --num-blocks 12 \
  --num-frames 15 \
  --patch-size 256 \
  --batch-size 1 \
  --workers 2 \
  --steps 150000 \
  --lr 7.5e-5 \
  --lambda-edge 0.08 \
  --lambda-temp 0.15 \
  --lambda-distill 0.10 \
  --amp \
  --save-every 5000 \
  2>&1 | tee "$RUN/stage3_distill/train.log"
```

Do not assume distilled is automatically better. Compare Stage 2 and Stage 3 PSNR and business-video visual quality. If Stage 3 introduces teacher artifacts or loses PSNR materially, keep Stage 2 as final.

## Phase F — business-video zero-shot test
Use the known business video if it exists:
```bash
INPUT=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Pick candidate checkpoint:
- Stage 2 by default
- Stage 3 only if it is measurably/visually better

```bash
CKPT="$RUN/stage2/latest.pth"
[ -f "$RUN/stage3_distill/latest.pth" ] && echo 'Stage3 exists; compare before selecting.'

CUDA_VISIBLE_DEVICES=0 python infer_video.py \
  --input "$INPUT" \
  --checkpoint "$CKPT" \
  --output "$RUN/nanovsr_deblur_business.mp4" \
  --chunk 15 \
  --overlap 4 \
  --fp16
```

Also generate a side-by-side MP4 with ffmpeg WITHOUT writing new Python code:
```bash
ffmpeg -y -i "$INPUT" -i "$RUN/nanovsr_deblur_business.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -an \
  "$RUN/input_vs_nanovsr_deblur.mp4"
```

Human review is required here. Stop and report the two output paths. Do not claim visual success without user review.

## Phase G — final model selection
Final student selection rules:
1. Prefer 48x12 Stage 2 unless Stage 3 clearly improves business video without artifacts.
2. If 48x12 is clearly over budget, train 40x10 using the exact same Stage 1/2 curriculum. Do not jump straight to 32x8 unless required.
3. Report GoPro PSNR for all actually trained variants.
4. The final deployment checkpoint must not depend on a teacher.

Expected success criteria:
- Params roughly sub-1M to around 1M; exact result must come from `profile_model.py`, not estimation.
- 640x360 per-frame MACs/FLOPs must be reported from profiler.
- GoPro deblur must be clearly stronger than blurry input baseline.
- Business-video output should improve motion sharpness without obvious color shift, face warping, texture hallucination, or temporal flicker.

## Final report format
Return exactly these fields plus concise notes:

```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovsr-deblur-gopro-distill-20260902
GITHUB_COMMIT: <sha>
GOPRO_ROOT: <path>
GPU: <name>
PYTORCH: <version>
FINAL_VARIANT: <e.g. 48x12-stage2>
FINAL_CHECKPOINT: <path>
CHECKPOINT_SHA256: <sha256>
PARAMS: <count and M>
MACS_PER_FRAME_640x360: <G>
FLOPS_PER_FRAME_640x360_2xMAC: <G>
GOPRO_PSNR_RGB_T7: <dB>
GOPRO_PSNR_RGB_T15: <dB>
TEACHER_USED: YES/NO + identity
BUSINESS_INPUT: <path>
BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE: <path>
PEAK_GPU_MEMORY: <GiB>
TRAINING_NOTES: <short>
VISUAL_REVIEW_REQUIRED: YES
```

Do not rank or declare this model better than Shift-Net-s/BSSTNet until the user has visually reviewed the business output under the same input video.
