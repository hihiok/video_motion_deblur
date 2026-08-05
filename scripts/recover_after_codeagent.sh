#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
CODE="${CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
SOURCE_ENV="${SOURCE_ENV:-turtle_joint_py222}"
RUNTIME_ENV="${RUNTIME_ENV:-deblur_runtime}"
REAL_ENV="${REAL_ENV:-$RUNTIME_ENV}"
SHIFT_ENV="${SHIFT_ENV:-$RUNTIME_ENV}"
DST_ENV="${DST_ENV:-$RUNTIME_ENV}"
BSST_ENV="${BSST_ENV:-bsstnet}"
GPU="${GPU:-0}"
SKIP_REALVDEBLUR="${SKIP_REALVDEBLUR:-0}"
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
      REAL_ENV="$SOURCE_ENV"
      SHIFT_ENV="$SOURCE_ENV"
      DST_ENV="$SOURCE_ENV"
    }
  else
    echo "Neither $RUNTIME_ENV nor source environment $SOURCE_ENV exists." >&2
    exit 1
  fi
fi

# Refuse to run a stale checkout that still imports all of BasicSR.
grep -q "spec_from_file_location" "$CODE/adapters/shiftnet_infer.py" || {
  echo "Stale Shift-Net adapter. Run git pull in $CODE." >&2
  exit 1
}
grep -q "get_root_logger" "$CODE/adapters/dstnet_compat.py" || {
  echo "Stale DSTNet compatibility loader. Run git pull in $CODE." >&2
  exit 1
}

PY=(conda run -n "$RUNTIME_ENV" python)
echo "Runtime environment: $RUNTIME_ENV"
"${PY[@]}" -c 'import sys, torch, numpy; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("numpy", numpy.__version__)'

# Deterministic 24-frame smoke source.
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

run_model() {
  local model="$1"
  echo "===== smoke: $model ====="
  if env ROOT="$ROOT" CODE="$CODE" GPU="$GPU" \
      COMMON_ENV="$RUNTIME_ENV" REAL_ENV="$REAL_ENV" \
      SHIFT_ENV="$SHIFT_ENV" DST_ENV="$DST_ENV" BSST_ENV="$BSST_ENV" \
      INPUT_SRC="$SMOKE_SRC" INPUT="$SMOKE_INPUT" \
      bash "$CODE/run_all.sh" "--model=$model"; then
    echo "$model: PASSED"
    return 0
  fi
  echo "$model: FAILED; continuing." >&2
  return 1
}

failures=0

# Run independently. A failure in one architecture never suppresses the rest.
run_model shiftnet || failures=$((failures + 1))
run_model dstnet || failures=$((failures + 1))

if [[ -s "$BENCH/weights/bsstnet/BSST_gopro.pth" && \
      -s "$BENCH/weights/bsstnet/BSST_dvd.pth" && \
      -s "$BENCH/weights/bsstnet/raft-things.pth" ]]; then
  run_model bsstnet || failures=$((failures + 1))
else
  echo "BSSTNet skipped: missing one or more official checkpoints:" >&2
  echo "  $BENCH/weights/bsstnet/BSST_gopro.pth" >&2
  echo "  $BENCH/weights/bsstnet/BSST_dvd.pth" >&2
  echo "  $BENCH/weights/bsstnet/raft-things.pth" >&2
fi

if [[ "$SKIP_REALVDEBLUR" == "1" ]]; then
  echo "RealVDeblur skipped because SKIP_REALVDEBLUR=1."
else
  real_dir="$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B"
  if [[ -s "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" && \
        -s "$real_dir/diffusion_pytorch_model.safetensors" && \
        -s "$real_dir/Wan2.1_VAE.pth" ]]; then
    run_model realvdeblur || failures=$((failures + 1))
  else
    echo "RealVDeblur requested but one or more checkpoint files are missing." >&2
    failures=$((failures + 1))
  fi
fi

echo "Recovery smoke run finished. failures=$failures"
echo "Inspect: $BENCH/outputs and $BENCH/logs"
(( failures == 0 ))
