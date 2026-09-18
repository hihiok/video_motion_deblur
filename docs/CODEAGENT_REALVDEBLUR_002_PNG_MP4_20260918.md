# CodeAgent: RealVDeblur 002/Blur/RGB -> BF16 PNG + MP4

## Scope

Already inside the target server. Do not SSH, rebuild torch, change the official network, change existing results, run other models, or continue the edge-audit experiments.

Input: `/data/pub1/h00306136/motion_deblur/input/002/Blur/RGB`.
Output: `/data/pub1/h00306136/motion_deblur/benchmark/outputs/realvdeblur_002_BF16/frames/` and `output.mp4` in the parent directory. If that output directory exists, preserve it and choose a timestamp-suffixed new directory. Never delete old FP16/BF16 outputs.

Use the existing, successfully tested Blackwell Python environment. The model repository remains `/data/pub1/h00306136/motion_deblur/envs/realvdeblur_repo`, pinned official commit `63a2eb0ed4a22cb76b796a65d5d3052616352573`. This task's wrapper directly launches its unmodified `inference.py`, with `bfloat16`, seed 0, stride 1, one step, TWM enabled, window 21. No FP32 VAE experiment, filtering, sharpening, independent temporal clips, or resizing.

Actual input frame count and resolution must be measured, not copied from the previous 452-frame video. Non-multiple-of-16 dimensions follow the official loader's small center crop, recorded in the manifest; never stretch the result back or claim that cropped pixels were retained.

## Proxy / SSL

No credentials are stored in this public repository. Use the user's private proxy configuration supplied in the companion local MD, or an already configured environment. Do not print credentials or upload private logs, diffs, frames, or weights. Before clone/fetch, for the user's explicitly requested HTTPS-inspection environment only:

```bash
set +x
: "${https_proxy:?Load the user's private proxy configuration first}"
export http_proxy="$https_proxy"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
export PIP_TRUSTED_HOST='pypi.org files.pythonhosted.org'
```

Disabling TLS verification weakens server authentication. Do not change system certificates or drivers. Inference uses existing local weights and does not need network access.

## Code and environment

Repository: `https://github.com/hihiok/video_motion_deblur.git`.
Branch: `agent/realvdeblur-002-png-mp4-20260918-v1`.
Immutable runner/test commit: `5b3037ee935c783a446a47e0d67264675e89a885`.
Path: `scripts/realvdeblur_sequence/`.

Clone into a separate tools checkout, not the model repo. Use a clean checkout of the immutable runner commit. Offline use of the supplied code ZIP is allowed with the following SHA256 checks:

```text
b6ed913c532b1f1e05e0d9da8a582241a61be6c0b53e20b8ffa377068af5513c  run_realvdeblur_sequence.py
e2d7a66568f5e63a83e196e9574fdd47a1b45162e85cb8234099bc7420306672  test_sequence_runner.py
```

Set `PY` to the actual Python recorded in the last successful RealVDeblur report. Candidate: `/data/pub1/h00306136/anaconda3/envs/realvdeblur_blackwell/bin/python`. If it does not work, locate the already working StereoPilot-derived environment in prior local reports; do not upgrade or install torch. Explicitly use `$PY`, not an unspecified shell Python.

```bash
ROOT=/data/pub1/h00306136/motion_deblur
MODULE_DIR="$ROOT/tools/realvdeblur_002_runner_20260918/scripts/realvdeblur_sequence"
PY=/data/pub1/h00306136/anaconda3/envs/realvdeblur_blackwell/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import sys,torch,PIL; print(sys.executable,torch.__version__,torch.version.cuda)'
```

## MP4 is required: check ffmpeg first

```bash
"$PY" -c 'from run_realvdeblur_sequence import find_ffmpeg; print(find_ffmpeg())'
```

This checks PATH, the current environment, explicit FFMPEG_EXE/IMAGEIO_FFMPEG_EXE and installed imageio_ffmpeg. An existing user-owned ffmpeg binary with libx264 can be supplied using `--ffmpeg /absolute/path`. No ffprobe is required.

If no encoder is found, install only the isolated video tool, not into the model environment:

```bash
FFTOOL="$ROOT/tools/imageio_ffmpeg_0_6_0"
mkdir -p "$FFTOOL"
"$PY" -m pip install --no-deps --only-binary=:all: --no-cache-dir --retries 2 --timeout 30 \
  --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  --target "$FFTOOL" 'imageio-ffmpeg==0.6.0'
export PYTHONPATH="$FFTOOL:$MODULE_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'from run_realvdeblur_sequence import find_ffmpeg; print(find_ffmpeg())'
```

If network/cache access fails, request this offline Linux wheel in `$ROOT/tools/wheelhouse/`:
`imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl`, SHA256 `c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282`, official source `https://pypi.org/project/imageio-ffmpeg/0.6.0/#files`. Install with `$PY -m pip install --no-index --no-deps --target "$FFTOOL" "$WHEEL"` after hash verification. Do not use the Windows wheel or source tarball. Do not mark completion while MP4 is skipped.

## GPU and timing

The runner selects the numerically largest free-MiB GPU just before inference, fixes it by UUID and uses logical cuda:0. It respects inherited CUDA_VISIBLE_DEVICES and optional --allowed-gpus. If an inherited single-GPU setting is merely a stale manual setting in an unrestricted login shell, clear that stale setting before running. Never clear actual scheduler/container/administrator allocation restrictions. Do not kill another process. GPU selection is not a reservation or a guarantee against later contention.

The runner reads fps.txt or fps/frame_rate/framerate from metadata.json within RGB, Blur, and this scene's 002 folder. Conflicts stop for review. If no timing metadata exists, use a clearly labeled 30fps preview, not a claimed original frame rate. An authoritative rate can be supplied with --fps 25 or --fps 30000/1001. Never borrow xiaobieli38's frame rate. FPS only affects MP4 playback; inference uses all frames, stride 1. Image-sequence input has no audio.

## Execute

Run the 11 CPU tests; the actual MP4 encode/decode test must not skip:

```bash
cd "$MODULE_DIR"
"$PY" -m unittest -v test_sequence_runner.py
INPUT="$ROOT/input/002/Blur/RGB"
OUT="$ROOT/benchmark/outputs/realvdeblur_002_BF16"
if [ -e "$OUT" ]; then OUT="${OUT}_$(date +%Y%m%d_%H%M%S)"; fi
"$PY" -u "$MODULE_DIR/run_realvdeblur_sequence.py" \
  --repo "$ROOT/envs/realvdeblur_repo" --input "$INPUT" --out "$OUT" \
  --wan-dir "$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B" \
  --checkpoint "$ROOT/benchmark/weights/realvdeblur/realvdeblur_dmd.safetensors"
```

Do not mkdir OUT beforehand: the runner refuses existing output directories. PNG preparation preserves decoded RGB pixels and uses sequential staging names only to guarantee temporal ordering/PNG output. Final names keep source stems. It does not alter source files. The runner performs a real BF16 CUDA gate, not a version-string-only check, then runs the complete new sequence once.

The official repository must be unchanged. A diff/new Python file in model-loading paths stops execution. Do not reset or repair it; retain the diff and changed files for the user to send back. OOM must retain the exact traceback and GPU/input geometry. No resize/clip fallback.

The video uses libx264 CRF10, yuv420p, no audio, no spatial resize, and the model output dimensions. PNG remains the quality reference; MP4 is not lossless RGB. The runner decodes every MP4 frame to verify frame count, dimensions, and frame rate.

If PNG is valid but only MP4 fails, repair the external encoder and use `--encode-only --input "$INPUT" --out "$OUT"`; no repeat diffusion. Preserve/rename any failed output.partial.mp4 before retry, never overwrite a successful output.mp4.

## Required report / human actions

Report actual input/output counts and dimensions, crop borders, Python/torch/CUDA, selected GPU index and UUID, free MiB, official commit/diff, weights SHA256, nonzero LoRA update count (not proof of complete coverage), actual peak memory from official logs, inference seconds, PNG path, MP4 path, playback FPS and its source, and FINAL_REPORT.md.

Only claim REALVDEBLUR_002_PNG_MP4_COMPLETE when both PNG and MP4 validations pass. Current local tests cover CPU orchestration and real MP4 round trip, not the target server's GPU/model. Do not infer that successful decoding fixes edge artifacts.

Manual actions only if needed: upload missing offline ffmpeg wheel; return local model code modifications/diff; coordinate unavailable GPU resources. Ordinary execution needs no SSH or environment rebuild.
