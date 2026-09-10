#!/usr/bin/env bash
set -euo pipefail
set +x
# Invoke after activating deblur_runtime. Never change system proxy here.
CONFIG=${1:?Usage: bash scripts/run_rtf_temporal.sh CONFIG preflight-or-train [extra flags]}
MODE=${2:?Pass preflight or train}
shift 2
: "${CUDA_VISIBLE_DEVICES:?Choose one or two idle A100 GPU indices first}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
IFS=',' read -r -a SELECTED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
COUNT=${#SELECTED_GPUS[@]}
if [[ "$COUNT" != 1 && "$COUNT" != 2 ]]; then
  echo 'Select exactly one or two GPUs' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$COUNT" == 2 ]]; then
  exec python -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/train_rtf_temporal.py --config "$CONFIG" --mode "$MODE" "$@"
else
  exec python tools/train_rtf_temporal.py --config "$CONFIG" --mode "$MODE" "$@"
fi
