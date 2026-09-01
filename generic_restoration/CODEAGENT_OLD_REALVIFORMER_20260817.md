# CodeAgent task: RealViformer on the old V100 server

## Objective

Run the released RealViformer checkpoint on the same business MP4 used by the Blackwell-server benchmark. Do not train or fine-tune. RealViformer is fixed to 4x internally; the provided adapter runs the official 4x network in spatial/temporal chunks and normalizes each result back to the original resolution.

Do not write or modify model code yourself. Use only the code already committed in `hihiok/video_motion_deblur`. If you believe a code change is required, stop and send the exact error, command, repository commit, and affected file to the user.

## Proxy and SSL requirements

The server is behind an authenticated corporate proxy. The proxy values, including credentials, must come from the existing private conda activation file. Never print, copy, or commit the proxy URL, username, password, token, or cookie.

After activating the source/runtime conda environment, source its private proxy file if present:

```bash
if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]]; then
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi
```

Disable certificate verification on this trusted internal server before cloning or downloading:

```bash
git config --global http.sslVerify false
git config --global http.version HTTP/1.1
conda config --set ssl_verify false
export GIT_SSL_NO_VERIFY=true
export PYTHONHTTPSVERIFY=0
export SSL_NO_VERIFY=1
export HF_HUB_DISABLE_XET=1
```

The committed setup script also injects `generic_restoration/no_ssl/sitecustomize.py` only for Python download commands, so `requests`/`httpx` do not re-enable certificate checks behind the inspection proxy.

Do not display proxy environment variables. If the proxy activation file is missing or the proxy returns HTTP 407, stop and ask the user to configure the proxy privately; do not paste credentials into a command, Markdown file, log, or Git commit.

## Fixed locations

```text
Project root: /mnt/ssd1/z00919662/motion_deblur/generic_restoration
Benchmark code: /mnt/ssd1/z00919662/motion_deblur/generic_restoration/benchmark_code
Default input: /mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
Official repo: /mnt/ssd1/z00919662/motion_deblur/generic_restoration/envs/RealViformer
Runtime env: realviformer_rwvsr (cloned from RVRT)
```

If the default MP4 is absent, locate the exact business MP4 but do not guess based only on a similar filename. Set `INPUT_VIDEO=/absolute/path/file.mp4` when running the commands below.

## Phase 1: clone benchmark code

```bash
set -euo pipefail
ROOT=/mnt/ssd1/z00919662/motion_deblur/generic_restoration
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

## Phase 2: environment, official code, and checkpoint

```bash
cd /mnt/ssd1/z00919662/motion_deblur/generic_restoration/benchmark_code

ROOT=/mnt/ssd1/z00919662/motion_deblur/generic_restoration \
CODE=$PWD \
SOURCE_ENV=RVRT \
bash generic_restoration/setup_old_realviformer.sh
```

Expected final marker:

```text
REALVIFORMER_SETUP_PASS
```

If Google Drive cannot be downloaded through the proxy, manual action is required: the user must download the official RealViformer `weights.pth` and copy it to:

```text
/mnt/ssd1/z00919662/motion_deblur/generic_restoration/weights/realviformer/weights.pth
```

Do not substitute a BasicVSR, task-specific denoise/deblur, or unofficial checkpoint.

## Phase 3: 25-frame smoke test

```bash
cd /mnt/ssd1/z00919662/motion_deblur/generic_restoration/benchmark_code

ROOT=/mnt/ssd1/z00919662/motion_deblur/generic_restoration \
CODE=$PWD \
INPUT_VIDEO=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4 \
GPU=0 \
bash generic_restoration/run_old_realviformer.sh smoke
```

Report these items and then stop:

```text
Source MP4 SHA256: canonical/manifest.json -> source_sha256
Frame count and resolution
Official repo commit
Checkpoint filename, byte size, SHA256
Torch/CUDA/GPU
Peak GPU memory and elapsed time
check_report.json status
Preview path
```

Required preview:

```text
/mnt/ssd1/z00919662/motion_deblur/generic_restoration/runs/realviformer_smoke/preview_input_output.jpg
```

The user must manually inspect frame 0, middle, and last for color correctness, borders, face/text distortion, hallucinated texture, and residual blur.

## Phase 4: user approval, then full run

Do not create the approval marker until the user explicitly confirms the smoke preview is acceptable.

After approval:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur/generic_restoration
touch "$ROOT/APPROVE_REALVIFORMER_FULL"

cd "$ROOT/benchmark_code"
ROOT="$ROOT" \
CODE=$PWD \
INPUT_VIDEO=/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4 \
GPU=0 \
bash generic_restoration/run_old_realviformer.sh full
```

Expected outputs:

```text
runs/realviformer_full/frames/
runs/realviformer_full/output_1x.mp4
runs/realviformer_full/run_metadata.json
runs/realviformer_full/check_report.json
runs/realviformer_full/preview_input_output.jpg
```

If CUDA OOM occurs, retry without changing code by reducing `REALVIFORMER_TILE` from 256 to 192, then 160. Keep overlap at least 32 and report the final values. Do not silently reduce input resolution or drop frames.

## Acceptance rules

- Old and new servers must report the same source MP4 SHA256.
- Output frame count and size must equal the input.
- No black/constant/NaN output, RGB/BGR reversal, tile seam, border loss, or missing tail frames.
- The checkpoint must load with no missing/unexpected keys.
- Do not call a failed or corrupt output a domain gap.
- No PSNR/SSIM claim is allowed because the business MP4 has no sharp ground truth.
