#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SHIFTWAVE_CONFIG:?Set SHIFTWAVE_CONFIG to the new prepared config.json}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
run=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$SHIFTWAVE_CONFIG")
exec 9>"$run/launcher.lock"
flock -n 9 || { echo 'This run already has an active launcher'; exit 1; }
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(python -m shiftwave.gpus)
fi
python -m shiftwave.gpus --validate "$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#gpu_ids[@]} -eq 2 && "${gpu_ids[0]}" != "${gpu_ids[1]}" ]]
upstream=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream"])' "$SHIFTWAVE_CONFIG")
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$run/gpu_ids.txt"
# Synthetic DDP test: disposable weights, exactly two ranks; no training data used.
python -m torch.distributed.run --standalone --nproc_per_node=2 -m shiftwave.ddp_check \
    --upstream "$upstream" --output "$run/ddp_check.json"
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shiftwave.train \
    --config "$SHIFTWAVE_CONFIG" --preflight --activation-checkpointing on
# Confirm a real two-rank training update/resume before the uninterrupted full run.
python -m torch.distributed.run --standalone --nproc_per_node=2 -m shiftwave.train \
    --config "$SHIFTWAVE_CONFIG" --stop-after 2 --activation-checkpointing on 2>&1 | tee -a "$run/student/console.log"
python -m torch.distributed.run --standalone --nproc_per_node=2 -m shiftwave.train \
    --config "$SHIFTWAVE_CONFIG" --activation-checkpointing on 2>&1 | tee -a "$run/student/console.log"
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shiftwave.evaluate \
    --config "$SHIFTWAVE_CONFIG" --checkpoint "$run/student/latest.pth" --output "$run/test_student.json" \
    2>&1 | tee "$run/evaluate_student.log"
# Baseline failure must not erase a successfully measured student result.
if ! CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" python -m shiftwave.evaluate \
    --config "$SHIFTWAVE_CONFIG" --model teacher --domains gopro --output "$run/test_teacher_gopro.json" \
    > "$run/evaluate_teacher.log" 2>&1; then
    echo 'TEACHER_EVALUATION_FAILED; see evaluate_teacher.log; student results preserved'
fi
python -m shiftwave.report --run "$run" | tee "$run/final_console.log"
