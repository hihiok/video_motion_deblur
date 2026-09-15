#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SHIFT500_CONFIG:?Set SHIFT500_CONFIG to prepared config.json}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to 1/2/4/8 verified idle GPUs}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
world=${#gpu_ids[@]}
case "$world" in 1|2|4|8) ;; *) echo 'Use 1, 2, 4 or 8 GPUs'; exit 1;; esac
train_args=(--activation-checkpointing "${SHIFT500_CHECKPOINTING:-on}" --prefetch-factor "${SHIFT500_PREFETCH:-2}")
if [[ -n "${SHIFT500_WORKERS:-}" ]]; then train_args+=(--workers "$SHIFT500_WORKERS"); fi
run=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$SHIFT500_CONFIG")
# Preflight trains a shared 256 crop from the largest source frame per domain.
# Those disposable weights are never reused by training.
for variant in quality compact; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shift500.train --config "$SHIFT500_CONFIG" --variant "$variant" --preflight "${train_args[@]}"
 done
for variant in quality compact; do
    python -m torch.distributed.run --standalone --nproc_per_node="$world" -m shift500.train --config "$SHIFT500_CONFIG" --variant "$variant" "${train_args[@]}" 2>&1 | tee -a "$run/$variant/console.log"
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shift500.evaluate --config "$SHIFT500_CONFIG" --variant "$variant" --checkpoint "$run/$variant/latest.pth" --output "$run/test_$variant.json"
done
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shift500.evaluate --config "$SHIFT500_CONFIG" --variant teacher --output "$run/test_teacher.json"
python -m shift500.report --run "$run"
