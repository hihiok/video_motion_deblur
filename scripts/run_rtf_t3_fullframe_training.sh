#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/ssd1/z00919662/motion_deblur}
CODE=${CODE:-$ROOT/benchmark_code_t3_fullframe_v3}
CONFIG=${CONFIG:-$CODE/configs/rtf_t3_gopro_bsd_dvd_fullframe.yaml}
RUN=${RUN:-$ROOT/runs/rtfocuser_shift_dst_t3_gopro_bsd_dvd_fullframe_v3}
ENV_NAME=${ENV_NAME:-deblur_runtime}
GPU=${GPU:?Set GPU to a verified free physical GPU index}
PRETRAINED=${PRETRAINED:-$ROOT/envs/RT-Focuser/Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth}
MODE=${1:-preflight}
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
# Proxy credentials belong in $ROOT/proxy.md, never in this script or GitHub.
# The run itself is offline. For Git downloads use the SSL settings in the task MD.
cd "$CODE"
test -s "$PRETRAINED"
EXPECTED_SHA=6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb
ACTUAL_SHA=$(sha256sum "$PRETRAINED" | cut -d ' ' -f 1)
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || { echo 'Official checkpoint SHA256 mismatch' >&2; exit 2; }
test -z "$(git status --short)"
mkdir -p "$RUN/audit"
exec 9>"$RUN/.training.lock"
flock -n 9 || { echo 'This run already has a launcher running' >&2; exit 2; }
run_python() { conda run --no-capture-output -n "$ENV_NAME" python "$@"; }
REPORT=$RUN/audit/fullframe_memory_preflight.json
if [[ "$MODE" == preflight ]]; then
  run_python tools/preflight_rtf_t6_fullframe.py --config "$CONFIG" \
    --pretrained "$PRETRAINED" --output "$REPORT" --rounds 2
elif [[ "$MODE" == smoke || "$MODE" == formal || "$MODE" == resume ]]; then
  run_python tools/verify_rtf_fullframe_gate.py --config "$CONFIG" \
    --pretrained "$PRETRAINED" --report "$REPORT"
  if [[ "$MODE" == smoke ]]; then
    if [[ -e "$RUN/smoke/resolved_config.yaml" ]]; then
      echo 'Smoke output already exists; preserve it and report rather than overwrite' >&2; exit 2
    fi
    run_python tools/train_rtf_t6.py --config "$CONFIG" --pretrained "$PRETRAINED" \
      --output "$RUN/smoke" --max-iters 8 --samples-per-epoch 8 --validate-every 4
  elif [[ "$MODE" == formal ]]; then
    if [[ -e "$RUN/resolved_config.yaml" || -e "$RUN/checkpoints/latest.pth" ]]; then
      echo 'Formal output already exists; inspect it and use resume if valid' >&2; exit 2
    fi
    test -s "$RUN/smoke/checkpoints/latest.pth"
    run_python tools/train_rtf_t6.py --config "$CONFIG" --pretrained "$PRETRAINED" --output "$RUN"
  else
    test -s "$RUN/checkpoints/latest.pth"
    run_python tools/train_rtf_t6.py --config "$CONFIG" --resume "$RUN/checkpoints/latest.pth" --output "$RUN"
  fi
else
  echo "Usage: $0 [preflight|smoke|formal|resume]" >&2; exit 2
fi
