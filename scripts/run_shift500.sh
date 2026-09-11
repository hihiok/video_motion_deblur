#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SHIFT500_CONFIG:?Set SHIFT500_CONFIG to prepared config.json}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one or two verified idle GPUs}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
world=${#gpu_ids[@]}
if (( world < 1 || world > 2 )); then echo 'Use 1 or 2 GPUs'; exit 1; fi
run=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$SHIFT500_CONFIG")
# Preflight does an actual optimizer step on largest native frame per domain.
# Those disposable weights are never reused by training.
for variant in quality compact; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shift500.train --config "$SHIFT500_CONFIG" --variant "$variant" --preflight
 done
# Fixed holdout reference; test data are not used for checkpoint selection.
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shift500.evaluate --config "$SHIFT500_CONFIG" --variant teacher --split val --max-windows 3 --output "$run/val_teacher.json"
for variant in quality compact; do
    python -m torch.distributed.run --standalone --nproc_per_node="$world" -m shift500.train --config "$SHIFT500_CONFIG" --variant "$variant" 2>&1 | tee -a "$run/$variant/console.log"
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shift500.evaluate --config "$SHIFT500_CONFIG" --variant "$variant" --checkpoint "$run/$variant/best_gopro.pth" --output "$run/test_$variant.json"
done
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shift500.evaluate --config "$SHIFT500_CONFIG" --variant teacher --output "$run/test_teacher.json"
python -m shift500.report --run "$run"
