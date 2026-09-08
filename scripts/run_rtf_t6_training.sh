#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/ssd1/z00919662/motion_deblur}
CODE=${CODE:-$ROOT/benchmark_code}
CONFIG=${CONFIG:-$CODE/configs/rtf_t6_gopro_bsd_dvd.yaml}
ENV_NAME=${ENV_NAME:-deblur_runtime}
GPU=${GPU:-0}
MODE=${1:-formal}

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

mkdir -p "$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_v1/audit"

conda run --no-capture-output -n "$ENV_NAME" python "$CODE/tools/audit_rtf_t6_data.py" \
  --config "$CONFIG" \
  --output "$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_v1/audit/dataset_audit.json"

conda run --no-capture-output -n "$ENV_NAME" python "$CODE/tools/check_rtf_t6_complexity.py" \
  --config "$CONFIG" \
  --pretrained "$PRETRAINED" \
  --output "$ROOT/runs/rtfocuser_shift_dst_t6_gopro_bsd_dvd_v1/audit/complexity.json"

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT="$ROOT/runs/rtfocuser_shift_dst_t6_smoke"
  conda run --no-capture-output -n "$ENV_NAME" python "$CODE/tools/train_rtf_t6.py" \
    --config "$CONFIG" \
    --pretrained "$PRETRAINED" \
    --output "$OUTPUT" \
    --max-iters 4 \
    --samples-per-epoch 8 \
    --crop-size 64 \
    --validate-every 4
  echo "RTF_T6_SMOKE_PASS"
elif [[ "$MODE" == "formal" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python "$CODE/tools/train_rtf_t6.py" \
    --config "$CONFIG" \
    --pretrained "$PRETRAINED"
else
  echo "Usage: $0 [smoke|formal]" >&2
  exit 2
fi
