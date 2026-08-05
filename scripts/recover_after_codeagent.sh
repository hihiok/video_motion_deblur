#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
CODE="${CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
SOURCE_ENV="${SOURCE_ENV:-turtle_joint_py222}"
RUNTIME_ENV="${RUNTIME_ENV:-deblur_runtime}"
GPU="${GPU:-0}"
BENCH="$ROOT/benchmark"
SMOKE_SRC="$BENCH/smoke/source_24"
SMOKE_INPUT="$BENCH/smoke/input_frames"
mkdir -p "$SMOKE_SRC" "$SMOKE_INPUT" "$BENCH/logs"

conda_env_exists() {
  command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$1"
}

if ! conda_env_exists "$RUNTIME_ENV"; then
  if conda_env_exists "$SOURCE_ENV"; then
    echo "Cloning existing local environment $SOURCE_ENV -> $RUNTIME_ENV (no package download)..."
    conda create -n "$RUNTIME_ENV" --clone "$SOURCE_ENV" -y || {
      echo "Clone failed; using $SOURCE_ENV directly." >&2
      RUNTIME_ENV="$SOURCE_ENV"
    }
  else
    echo "Neither $RUNTIME_ENV nor source environment $SOURCE_ENV exists." >&2
    echo "Set RUNTIME_ENV to an existing Torch 2.4/CUDA 11.8 environment." >&2
    exit 1
  fi
fi

PY=(conda run -n "$RUNTIME_ENV" python)
PIP=(conda run -n "$RUNTIME_ENV" python -m pip)

echo "Runtime environment: $RUNTIME_ENV"
"${PY[@]}" -c 'import sys, torch; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'

missing=$("${PY[@]}" - <<'PY'
import importlib.util
checks = {
    "numpy": "numpy",
    "Pillow": "PIL",
    "opencv-python-headless": "cv2",
    "einops": "einops",
    "tqdm": "tqdm",
    "safetensors": "safetensors",
}
print(" ".join(pkg for pkg, module in checks.items() if importlib.util.find_spec(module) is None))
PY
)

if [[ -n "$missing" ]]; then
  echo "Missing Python packages: $missing" >&2
  if [[ "${INSTALL_MISSING:-0}" == "1" ]]; then
    pip_args=(install $missing)
    if [[ "${ALLOW_INSECURE_SSL:-0}" == "1" ]]; then
      echo "WARNING: pip TLS certificate verification is being bypassed for the configured proxy." >&2
      pip_args+=(--trusted-host pypi.org --trusted-host files.pythonhosted.org)
    fi
    "${PIP[@]}" "${pip_args[@]}" || {
      echo "Package installation failed. DSTNet/Shift-Net may still run if their imports are present." >&2
    }
  else
    echo "Re-run with INSTALL_MISSING=1 to install only these small runtime packages." >&2
  fi
fi

# Make a deterministic 24-frame smoke source using symlinks.
rm -f "$SMOKE_SRC"/*
mapfile -t first_frames < <(find "$ROOT/input/xiaobieli38_trimmed" -maxdepth 1 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | sort | head -n 24)
if (( ${#first_frames[@]} == 0 )); then
  echo "No source frames found." >&2
  exit 1
fi
for frame in "${first_frames[@]}"; do
  ln -s "$frame" "$SMOKE_SRC/$(basename "$frame")"
done

echo "Smoke source contains ${#first_frames[@]} frames."

# Recover the one missing RealVDeblur file. Failure does not block DSTNet/Shift-Net.
ROOT="$ROOT" ALLOW_INSECURE_SSL="${ALLOW_INSECURE_SSL:-0}" \
  HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-}" \
  bash "$CODE/scripts/download_realvdeblur_vae.sh" || true

common_run=(
  env ROOT="$ROOT" CODE="$CODE" GPU="$GPU"
  COMMON_ENV="$RUNTIME_ENV" REAL_ENV="$RUNTIME_ENV" DST_ENV="$RUNTIME_ENV" SHIFT_ENV="$RUNTIME_ENV"
  INPUT_SRC="$SMOKE_SRC" INPUT="$SMOKE_INPUT"
  bash "$CODE/run_all.sh"
)

# These two no longer require fresh conda environments, mmcv or CuPy.
"${common_run[@]}" --model=shiftnet
"${common_run[@]}" --model=dstnet

# RealVDeblur is attempted only when all three official files are present and
# the official CLI imports successfully in the cloned runtime.
REAL_REPO="$ROOT/envs/realvdeblur_repo"
REAL_DIR="$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B"
if [[ -s "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" && \
      -s "$REAL_DIR/diffusion_pytorch_model.safetensors" && \
      -s "$REAL_DIR/Wan2.1_VAE.pth" ]]; then
  if "${PY[@]}" "$REAL_REPO/inference.py" --help >/dev/null 2>&1; then
    "${common_run[@]}" --model=realvdeblur
  else
    echo "RealVDeblur official CLI does not import in $RUNTIME_ENV." >&2
    echo "Run the following to see the exact missing package or Python-version error:" >&2
    echo "conda run -n $RUNTIME_ENV python $REAL_REPO/inference.py --help" >&2
  fi
else
  echo "Skipping RealVDeblur: one or more official weights are still missing." >&2
fi

if [[ ! -s "$BENCH/weights/bsstnet/BSST_gopro.pth" || \
      ! -s "$BENCH/weights/bsstnet/BSST_dvd.pth" ]]; then
  echo "BSSTNet remains blocked only by its official Google Drive checkpoints." >&2
  echo "Expected paths:" >&2
  echo "  $BENCH/weights/bsstnet/BSST_gopro.pth" >&2
  echo "  $BENCH/weights/bsstnet/BSST_dvd.pth" >&2
  echo "  $BENCH/weights/bsstnet/raft-things.pth" >&2
fi

echo "Recovery smoke run finished. Inspect: $BENCH/outputs and $BENCH/logs"
