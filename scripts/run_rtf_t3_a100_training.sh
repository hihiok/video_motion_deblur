#!/usr/bin/env bash
set -euo pipefail
export ROOT=${ROOT:-/data/pub/z00919662/motion_deblur}
export CODE=${CODE:-$ROOT/benchmark_code_t3_fullframe_v3}
export ENV_NAME=${ENV_NAME:-deblur_runtime}
GPUS=${GPUS:-${GPU:-}}
[[ -n "$GPUS" ]] || { echo 'Set GPUS to one or two verified free A100 indices' >&2; exit 2; }
IFS=, read -r -a DEVICES <<< "$GPUS"
NGPUS=${#DEVICES[@]}
[[ "$NGPUS" == 1 || "$NGPUS" == 2 ]] || { echo 'Only 1 or 2 GPUs supported' >&2; exit 2; }
for GPU_ID in "${DEVICES[@]}"; do
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'Use numeric physical GPU indices' >&2; exit 2; }
done
if [[ "$NGPUS" == 2 && "${DEVICES[0]}" == "${DEVICES[1]}" ]]; then
  echo 'GPU indices must be distinct' >&2; exit 2
fi
if [[ ${WORLD_SIZE:-1} != 1 || ${RANK:-0} != 0 || ${LOCAL_RANK:-0} != 0 ]]; then
  echo 'Run launcher outside an existing distributed job' >&2; exit 2
fi
export RUN=${RUN:-$ROOT/runs/rtfocuser_shift_dst_t3_gopro_bsd_dvd_a100_v4_${NGPUS}gpu}
export CONFIG=${CONFIG:-$RUN/runtime_config.yaml}
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
MODE=${1:-preflight}
cd "$CODE"
test -z "$(git status --short)"
test -s "$CONFIG"
run_python() { conda run --no-capture-output -n "$ENV_NAME" python "$@"; }
if [[ -z ${PRETRAINED:-} ]]; then
  PRETRAINED=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pretrained"])' "$RUN/a100_setup.json")
fi
export PRETRAINED
run_python -c '
import sys,torch,yaml
from rtf_t6.protocol import sha256_file
c=yaml.safe_load(open(sys.argv[1]))
assert c["train"]["expected_world_size"] == int(sys.argv[3]), "Config/GPU count mismatch"
assert sha256_file(sys.argv[2]) == "6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb", "Wrong official checkpoint"
for i in range(int(sys.argv[3])):
    p=torch.cuda.get_device_properties(i)
    assert "A100" in p.name and p.total_memory >= 70*2**30, "Requires full A100 80GB devices"
' "$CONFIG" "$PRETRAINED" "$NGPUS"
mkdir -p "$RUN/audit"
exec 9>"$RUN/.training.lock"
flock -n 9 || { echo 'This run already has an active launcher' >&2; exit 2; }
run_training() {
  if [[ "$NGPUS" == 1 ]]; then run_python "$@";
  else run_python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "$@"; fi
}
REPORT=$RUN/audit/fullframe_memory_preflight.json
if [[ "$MODE" == preflight ]]; then
  run_training tools/preflight_rtf_t6_fullframe.py --config "$CONFIG" --pretrained "$PRETRAINED" --output "$REPORT" --rounds 2
elif [[ "$MODE" == benchmark || "$MODE" == smoke || "$MODE" == formal || "$MODE" == resume ]]; then
  run_training tools/verify_rtf_fullframe_gate.py --config "$CONFIG" --pretrained "$PRETRAINED" --report "$REPORT"
  if [[ "$MODE" == benchmark ]]; then
    BENCH_RUN=${BENCH_RUN:-$RUN/benchmark}
    test ! -e "$BENCH_RUN/resolved_config.yaml" || { echo 'Benchmark output exists; set a new BENCH_RUN' >&2; exit 2; }
    run_training tools/train_rtf_t6.py --config "$CONFIG" --pretrained "$PRETRAINED" --output "$BENCH_RUN" \
      --benchmark-updates 16 --benchmark-warmup-updates 4
  elif [[ "$MODE" == smoke ]]; then
    test ! -e "$RUN/smoke/resolved_config.yaml" || { echo 'Preserve existing smoke output; do not overwrite' >&2; exit 2; }
    run_training tools/train_rtf_t6.py --config "$CONFIG" --pretrained "$PRETRAINED" --output "$RUN/smoke" \
      --max-iters "$((8 / NGPUS))" --samples-per-epoch 8 --validate-every "$((4 / NGPUS))"
  elif [[ "$MODE" == formal ]]; then
    test ! -e "$RUN/resolved_config.yaml" || { echo 'Formal output exists; inspect and use resume' >&2; exit 2; }
    test -s "$RUN/smoke/checkpoints/latest.pth"
    run_training tools/train_rtf_t6.py --config "$CONFIG" --pretrained "$PRETRAINED" --output "$RUN"
  else
    test -s "$RUN/checkpoints/latest.pth"
    run_training tools/train_rtf_t6.py --config "$CONFIG" --resume "$RUN/checkpoints/latest.pth" --output "$RUN"
  fi
else
  echo "Usage: $0 [preflight|benchmark|smoke|formal|resume]" >&2; exit 2
fi
