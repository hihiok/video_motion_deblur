# CODEAGENT TASK — NanoVNR NAFNet RGB Full-Frame, BSD train/test only

## Goal
Run the NanoVNR NAFNet RGB experiment using the exact RGB-adapted model already committed on this branch, with one dataset-policy correction:

**BSD may only use**
- `/mnt/ssd1/z00919662/datasets/BSD/train`
- `/mnt/ssd1/z00919662/datasets/BSD/test`

Do not discover or sample any other directory below `/mnt/ssd1/z00919662/datasets/BSD`, including any `BSD/<config>/train` or `BSD/<config>/test` tree.

## Repository
Repository:
`https://github.com/hihiok/video_motion_deblur.git`

Branch:
`agent/nanovnr-nafnet-rgb-fullframe-20260904`

Primary files:
- `nanovsr_deblur/models/network_nanovnr_nafnet_rgb.py`
- `nanovsr_deblur/data/mixed_deblur.py`
- `nanovsr_deblur/audit_fullframe_datasets.py`
- `nanovsr_deblur/train_nanovnr_nafnet_rgb_fullframe_bsd_splits.py`
- `nanovsr_deblur/profile_nanovnr_nafnet_rgb.py`
- `nanovsr_deblur/eval_gopro_nanovnr_nafnet_rgb.py`
- `nanovsr_deblur/infer_video_nanovnr_nafnet_rgb.py`

## Model definition
Use `NanoVNRNAFNetRGB` exactly as committed.

Only architecture difference from the user-supplied Python model:
- supplied: `Conv2d(4,12,3,1,1)`
- current RGB version: `Conv2d(3,12,3,1,1)`

Everything else must remain unchanged:
- `num_feat=12`
- `prop_channels=[24,32,48,72]`
- `enc_blk_nums=[1,1,1]`
- `middle_blk_num=1`
- `dec_blk_nums=[1,1,1]`
- ChannelRowLayerNorm
- PReLU
- beta/gamma residual scaling
- no SCA
- current feature + hidden concat at full resolution
- 3-level U-Net, down to 1/8 internally
- decoder skip uses addition
- PixelShuffle upsampling
- independent forward/backward propagation
- final forward/backward concat -> 1x1 24->12 -> 3x3 12->3
- `prev_forward_feat` / `next_forward_feat` interface preserved

## Dataset roots
```bash
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
```

### BSD strict policy
Allowed BSD roots are exactly:
```text
/mnt/ssd1/z00919662/datasets/BSD/train
/mnt/ssd1/z00919662/datasets/BSD/test
```

Forbidden examples:
```text
/mnt/ssd1/z00919662/datasets/BSD/BSD_*/train
/mnt/ssd1/z00919662/datasets/BSD/BSD_*/test
/mnt/ssd1/z00919662/datasets/BSD/<anything-else>/train
/mnt/ssd1/z00919662/datasets/BSD/<anything-else>/test
```

Do not copy, merge, rename, or regenerate BSD files.

Training uses BSD `train` only. BSD `test` is audit/evaluation-eligible only and must never enter training sampling.

## Resolution policy
Native full-frame only.

- no random crop
- no patch training
- no resize
- no forced 720p
- no forced 1080p
- batch size 1

Use each sequence at its native resolution.

## Training recipe
Phase 1:
- T=7
- steps 1-50000

Phase 2:
- T=30
- steps 50001-150000

Loss:
- Charbonnier only

Optimizer:
- Adam, betas=(0.9,0.99)

Schedule:
- one cosine schedule across all 150000 steps
- LR 3e-4 -> 1e-7
- no optimizer/scheduler reset at 50k

Other:
- grad clip 0.5
- AMP
- gradient checkpointing
- workers <=2
- random initialization

## Step A — sync branch
```bash
git config --global http.sslVerify false
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true

ROOT=/mnt/ssd1/z00919662/motion_deblur
REPO=$ROOT/video_motion_deblur_nanovnr_nafnet_rgb
RUN=$ROOT/runs/nanovnr_nafnet_rgb_fullframe_bsd_train_test_20260904
GOPRO=/mnt/ssd1/z00919662/motion_deblur/datasets/GoPro
DVD=/mnt/ssd1/z00919662/motion_deblur/datasets/DVD
BSD=/mnt/ssd1/z00919662/datasets/BSD
INPUT=$ROOT/input/xiaobieli38_trimmed.mp4

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
mkdir -p "$RUN"
```

Do not print proxy credentials.

## Step B — syntax audit
```bash
cd "$REPO/nanovsr_deblur"
python -m py_compile \
  models/network_nanovnr_nafnet_rgb.py \
  data/mixed_deblur.py \
  audit_fullframe_datasets.py \
  train_nanovnr_nafnet_rgb_fullframe.py \
  train_nanovnr_nafnet_rgb_fullframe_bsd_splits.py \
  profile_nanovnr_nafnet_rgb.py \
  eval_gopro_nanovnr_nafnet_rgb.py \
  infer_video_nanovnr_nafnet_rgb.py
```

## Step C — mandatory BSD split audit
Verify exact directories exist:
```bash
test -d "$BSD/train" || { echo "MISSING=$BSD/train"; exit 2; }
test -d "$BSD/test"  || { echo "MISSING=$BSD/test"; exit 2; }
```

Run:
```bash
python audit_fullframe_datasets.py \
  --gopro-root "$GOPRO" \
  --dvd-root "$DVD" \
  --bsd-root "$BSD" \
  2>&1 | tee "$RUN/dataset_audit.log"
```

Required output must contain:
```text
BSD_ROOT_SPLIT_ONLY=YES
BSD_ALLOWED_SPLITS=train,test
BSD_NESTED_CONFIG_SPLITS=FORBIDDEN
```

Audit every printed BSD blur/GT path. Every BSD training path must be below:
```text
$BSD/train
```
and every BSD test path must be below:
```text
$BSD/test
```

If any BSD path is outside those roots, STOP immediately:
```text
HUMAN_ACTION_REQUIRED: YES
BSD_SPLIT_POLICY_VIOLATION: <path>
```

Report T=7 and T=30 window counts for both BSD train and BSD test.

## Step D — architecture assertions
Confirm:
```text
feat_extract.in_channels=3
feat_extract.out_channels=12
prop_channels=[24,32,48,72]
fusion: 24->12
conv_last: 12->3
```

Confirm source uses:
```python
torch.cat([cur_feat, prop_feat], dim=1)
```
and U-Net skip uses:
```python
x = x + enc_skip
```

## Step E — profile
```bash
python profile_nanovnr_nafnet_rgb.py --height 360 --width 640 --frames 3 \
  | tee "$RUN/profile_640x360.txt"
python profile_nanovnr_nafnet_rgb.py --height 720 --width 1280 --frames 3 \
  | tee "$RUN/profile_1280x720.txt"
python profile_nanovnr_nafnet_rgb.py --height 1080 --width 1920 --frames 3 \
  | tee "$RUN/profile_1920x1080.txt"
```

## Step F — GPU selection
Use one GPU with the largest free memory. Record GPU name, total memory and free memory.

## Step G — T=30 native full-frame preflight
Use the strict recipe entrypoint:
```bash
mkdir -p "$RUN/preflight"
python train_nanovnr_nafnet_rgb_fullframe_bsd_splits.py \
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

For BSD, preflight samples must come only from `$BSD/train`.

If OOM, do not crop/resize/reduce T/change model. Stop and report the failing family/resolution.

## Step H — full training
Only after preflight PASS:
```bash
mkdir -p "$RUN/train"
python train_nanovnr_nafnet_rgb_fullframe_bsd_splits.py \
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

Expected recipe id:
```text
nanovnr_nafnet_rgb_native_fullframe_mix_bsd_train_test_v2
```

## Step I — checkpoint evaluation
Evaluate 50k/75k/100k/125k/150k on GoPro test, same first 100 clips, T=15, native full frame, RGB PSNR.
Choose BEST_T15.

Then evaluate BEST at T=7/T=15/T=30 and center-frame matched-target metrics.

## Step J — business video
Run native 1280x720 inference on:
```text
/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Use `infer_video_nanovnr_nafnet_rgb.py`, preserving non-overlap chunk forward-state carry via `prev_forward_feat`.

Generate output and side-by-side MP4.

## Required final report
Return at least:
```text
STATUS: PASS / PARTIAL / FAIL
HUMAN_ACTION_REQUIRED: YES / NO
GITHUB_BRANCH: agent/nanovnr-nafnet-rgb-fullframe-20260904
GITHUB_COMMIT: <sha>
RECIPE_ID: nanovnr_nafnet_rgb_native_fullframe_mix_bsd_train_test_v2
ARCHITECTURE: NanoVNRNAFNetRGB
BSD_ROOT: /mnt/ssd1/z00919662/datasets/BSD
BSD_TRAIN_ONLY_FOR_TRAINING: YES
BSD_ALLOWED_SPLITS: train,test
BSD_NESTED_CONFIG_SPLITS_USED: NO
BSD_TRAIN_T7_WINDOWS: <n>
BSD_TRAIN_T30_WINDOWS: <n>
BSD_TEST_T7_WINDOWS: <n>
BSD_TEST_T30_WINDOWS: <n>
GOPRO_NATIVE_RESOLUTIONS: <...>
DVD_NATIVE_RESOLUTIONS: <...>
BSD_NATIVE_RESOLUTIONS: <...>
PARAMS: <...>
MACS_PER_FRAME_640x360: <...>
MACS_PER_FRAME_1280x720: <...>
MACS_PER_FRAME_1920x1080: <...>
PREFLIGHT_T30_FULLFRAME: PASS / FAIL
BEST_T15_CHECKPOINT: <path>
BEST_T15_PSNR: <dB>
BUSINESS_OUTPUT: <path>
SIDE_BY_SIDE_OUTPUT: <path>
```
