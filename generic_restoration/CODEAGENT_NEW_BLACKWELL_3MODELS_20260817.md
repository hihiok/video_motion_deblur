# CodeAgent task: FlashVSR v1.1, SeedVR2-3B, and DOVE Final on Blackwell

## Objective

Run three released generic-restoration checkpoints on the exact business MP4 used by RealViformer on the old server:

1. FlashVSR v1.1 Tiny Long, experimental direct 1x mode.
2. SeedVR2-3B, target size equal to input size.
3. DOVE Stage-2 Final, official `--upscale 1` with the padding-crop bug fixed by the committed patch.

Use separate conda environments. Do not train or fine-tune. Do not write or modify model code yourself. All required adapters and deterministic patches are already in `hihiok/video_motion_deblur`. If another change appears necessary, stop and send the exact error, command, official commit, and affected file to the user.

## Proxy and SSL requirements

The authenticated proxy must be loaded only from the server's private conda activation file. Never print, paste, or commit the proxy username, password, URL, token, or cookie.

```bash
if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]]; then
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi

git config --global http.sslVerify false
git config --global http.version HTTP/1.1
conda config --set ssl_verify false
export GIT_SSL_NO_VERIFY=true
export PYTHONHTTPSVERIFY=0
export SSL_NO_VERIFY=1
export HF_HUB_DISABLE_XET=1
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org huggingface.co cdn-lfs.huggingface.co cas-bridge.xethub.hf.co github.com raw.githubusercontent.com objects.githubusercontent.com"
```

The committed setup script also injects `generic_restoration/no_ssl/sitecustomize.py` only for Python download commands, preventing `requests`/`httpx` from turning certificate verification back on behind the inspection proxy.

Do not display proxy variables. If the private proxy activation file is missing or an HTTP 407 occurs, stop and ask the user to configure it privately. Do not place credentials in this Markdown file, shell history, logs, scripts, or Git.

## Fixed locations

```text
Project root: /data/pub1/z00919662/motion_deblur/generic_restoration
Benchmark code: /data/pub1/z00919662/motion_deblur/generic_restoration/benchmark_code
Default input: /data/pub1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
Conda base: /data/pub1/z00919662/anaconda3
Blackwell source env: StereoPilot
GPU: RTX PRO 6000 Blackwell, compute capability 12.0
```

The setup script refuses a non-sm_120 GPU. It keeps independent environments:

```text
flashvsr_blackwell
seedvr2_blackwell
dove_blackwell
```

## Phase 1: clone benchmark code

```bash
set -euo pipefail
ROOT=/data/pub1/z00919662/motion_deblur/generic_restoration
mkdir -p "$ROOT"
cd "$ROOT"

git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true

if [[ ! -d benchmark_code/.git ]]; then
  git clone https://github.com/hihiok/video_motion_deblur.git benchmark_code
fi
cd benchmark_code
git fetch origin
git checkout agent/generic-restoration-four-models
git status -sb
```

Stop if the benchmark checkout contains unrelated local changes.

## Phase 2: set up all three models

```bash
cd /data/pub1/z00919662/motion_deblur/generic_restoration/benchmark_code

ROOT=/data/pub1/z00919662/motion_deblur/generic_restoration \
CODE=$PWD \
BLACKWELL_SOURCE_ENV=StereoPilot \
CUDA_VISIBLE_DEVICES=0 \
bash generic_restoration/setup_new_models.sh all
```

Expected markers:

```text
FLASHVSR_SETUP_PASS
SEEDVR2_SETUP_PASS
DOVE_SETUP_PASS
```

`all` continues to the next model if one setup fails. Report each failure separately; never mark a skipped model as zero quality.

Manual action may be required for DOVE: if Google Drive fails through the proxy, download the official Stage-2 Final archive from the DOVE repository link and copy it to:

```text
/data/pub1/z00919662/motion_deblur/generic_restoration/weights/DOVE_Final/dove_final_download
```

Then rerun only:

```bash
bash generic_restoration/setup_new_models.sh dove
```

FlashVSR's block-sparse CUDA extension is compiled from the pinned upstream commit with `sm_120` and CUDA 12.8. SeedVR2 first attempts Blackwell-compatible FlashAttention and Apex builds. If either build fails, the committed inference-only PyTorch fallback is applied and recorded; do not invent another replacement.

## Phase 3: 25-frame smoke inference

```bash
cd /data/pub1/z00919662/motion_deblur/generic_restoration/benchmark_code

ROOT=/data/pub1/z00919662/motion_deblur/generic_restoration \
CODE=$PWD \
INPUT_VIDEO=/data/pub1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4 \
GPU=0 \
bash generic_restoration/run_new_models.sh smoke all
```

The script continues to the remaining model if one fails. Individual retries are:

```bash
bash generic_restoration/run_new_models.sh smoke flashvsr
bash generic_restoration/run_new_models.sh smoke seedvr2
bash generic_restoration/run_new_models.sh smoke dove
```

Report these items and stop:

```text
Source MP4 SHA256: canonical/manifest.json -> source_sha256
Frame count and resolution
Each official repo commit
Checkpoint paths and byte sizes
Torch/CUDA/GPU per environment
Attention/norm backend used by SeedVR2
Peak memory and elapsed time when available
Each check_report.json status
Combined preview path
```

Combined preview:

```text
/data/pub1/z00919662/motion_deblur/generic_restoration/runs/new_models_smoke_combined_preview.jpg
```

The user must manually compare frame 0, middle, and last for faces, subtitles, logos, thin lines, color, hallucinated texture, ringing, borders, residual blur, and temporal consistency.

## Phase 4: user approval, then full inference

Do not create the marker until the user explicitly approves the combined smoke preview.

```bash
ROOT=/data/pub1/z00919662/motion_deblur/generic_restoration
touch "$ROOT/APPROVE_NEW_MODELS_FULL"

cd "$ROOT/benchmark_code"
ROOT="$ROOT" \
CODE=$PWD \
INPUT_VIDEO=/data/pub1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4 \
GPU=0 \
bash generic_restoration/run_new_models.sh full all
```

The full outputs are under:

```text
runs/flashvsr_full/
runs/seedvr2_full/
runs/dove_full/
runs/new_models_full_combined_preview.jpg
```

Every model directory must contain:

```text
frames/
output_1x.mp4
run_metadata.json
check_report.json
preview_input_output.jpg
```

## OOM recovery without code changes

Run one model at a time. Do not reduce source resolution or discard frames.

For SeedVR2, reduce bounded clip cores:

```bash
SEEDVR2_CORE_FRAMES=33 bash generic_restoration/run_new_models.sh full seedvr2
```

If needed, use 25, then 17.

For DOVE, first reduce temporal chunks:

```bash
DOVE_CHUNK_LEN_FULL=25 bash generic_restoration/run_new_models.sh full dove
```

If it still OOMs, enable committed spatial tiling:

```bash
DOVE_CHUNK_LEN_FULL=25 \
DOVE_TILE_H=512 DOVE_TILE_W=768 \
DOVE_OVERLAP_H=64 DOVE_OVERLAP_W=64 \
bash generic_restoration/run_new_models.sh full dove
```

FlashVSR direct1x should fit the 97GB card. If it fails inside block-sparse attention, report the build/import/runtime error exactly; do not switch to a third-party ComfyUI implementation or silently remove locality-constrained sparse attention.

## Acceptance rules

- The source MP4 SHA256 must exactly match the old-server RealViformer run.
- Output frame count and geometry must equal input; the FlashVSR tail-frame bug must not recur.
- No missing tail frames, crop, black/constant/NaN output, RGB/BGR reversal, chunk seam, tile seam, or checkpoint HTML/LFS pointer.
- All output MP4 files must reuse original relative PTS and audio.
- Do not rank a failed model as zero, and do not describe pipeline corruption as domain mismatch.
- Do not compute PSNR/SSIM without a sharp ground truth.
- FlashVSR direct1x must be labeled experimental; do not claim it is the model's official recommended 4x configuration.
