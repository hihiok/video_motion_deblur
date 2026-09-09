#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/ssd1/z00919662/motion_deblur}
CODE=${CODE:-$ROOT/benchmark_code}
CONFIG=${CONFIG:-$CODE/configs/rtf_t6_gopro_bsd_dvd_fullframe.yaml}
ENV_NAME=${ENV_NAME:-deblur_runtime}
GPU=${GPU:-0}
MODE=${1:-formal}
RUN=${RUN:-$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_fullframe_v2}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export GIT_SSL_NO_VERIFY=${GIT_SSL_NO_VERIFY:-true}
export CONDA_SSL_VERIFY=${CONDA_SSL_VERIFY:-false}

find_pretrained() {
  local candidates=(
    "$ROOT/envs/RT-Focuser/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth"
    "$ROOT/envs/rt_focuser_repo/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth"
    "$ROOT/benchmark/weights/rt_focuser/GoPro_RT_Focuser_Standard_256.pth"
  )
  local item
  for item in "${candidates[@]}"; do
    if [[ -s "$item" ]]; then
      printf '%s\n' "$item"
      return 0
    fi
  done
  return 1
}

PRETRAINED=${PRETRAINED:-$(find_pretrained || true)}
if [[ -z "$PRETRAINED" || ! -s "$PRETRAINED" ]]; then
  echo "Missing official GoPro_RT_Focuser_Standard_256.pth" >&2
  echo "Set PRETRAINED=/absolute/path/to/GoPro_RT_Focuser_Standard_256.pth" >&2
  exit 2
fi

mkdir -p "$RUN/audit"

if [[ "$MODE" == "preflight" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python \
    "$CODE/tools/preflight_rtf_t6_fullframe.py" \
    --config "$CONFIG" \
    --pretrained "$PRETRAINED" \
    --output "$RUN/audit/fullframe_memory_preflight.json"
elif [[ "$MODE" == "formal" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python "$CODE/tools/train_rtf_t6.py" \
    --config "$CONFIG" \
    --pretrained "$PRETRAINED"
else
  echo "Usage: $0 [preflight|formal]" >&2
  exit 2
fi
