#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
BENCH="$ROOT/benchmark"
INPUT_SRC="$ROOT/input/xiaobieli38_trimmed"
INPUT_MP4="$ROOT/input/xiaobieli38_trimmed.mp4"
INPUT="$BENCH/input_frames"
GPU="${GPU:-0}"
MODE="${1:---all}"

REAL_ENV="${REAL_ENV:-realvdeblur}"
BSST_ENV="${BSST_ENV:-bsstnet}"
DST_ENV="${DST_ENV:-dstnet}"
SHIFT_ENV="${SHIFT_ENV:-shiftnet}"

mkdir -p "$BENCH"/{outputs,logs,manifests,videos}
python3 "$CODE/scripts/prepare_input.py" --source-frames "$INPUT_SRC" --source-mp4 "$INPUT_MP4" --output "$INPUT"
FPS=$(python3 -c "import json; print(json.load(open('$BENCH/manifests/input.json'))['fps'])")

run_logged() {
  local name="$1"; shift
  echo "===== $name ====="
  if "$@" 2>&1 | tee "$BENCH/logs/${name}.log"; then
    echo "$name: complete"
  else
    echo "$name: FAILED; see $BENCH/logs/${name}.log" >&2
    return 1
  fi
}

if [[ "$MODE" == "--download-only" ]]; then
  ROOT="$ROOT" bash "$CODE/scripts/download_weights.sh"
  exit
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=realvdeblur" ]]; then
  run_logged realvdeblur \
    env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$REAL_ENV" python "$CODE/adapters/realvdeblur_infer.py" \
      --repo "$ROOT/envs/realvdeblur_repo" --input "$INPUT" \
      --output "$BENCH/outputs/realvdeblur_dmd/frames" \
      --wan-model-dir "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B" \
      --checkpoint "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" \
      --device cuda:0 --dtype float16 --temporal-window-size 21 || true
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=dstnet" ]]; then
  for domain in GOPRO DVD BSD; do
    lname=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
    run_logged "dstnet_${lname}" \
      env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$DST_ENV" python "$CODE/adapters/dstnet_infer.py" \
        --repo "$ROOT/envs/dstnet_repo" --input "$INPUT" \
        --output "$BENCH/outputs/dstnet_${lname}/frames" \
        --checkpoint "$BENCH/weights/dstnet/${domain}.pth" \
        --clip-len 30 --overlap 10 --device cuda:0 --amp || true
  done
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=shiftnet" ]]; then
  for domain in gopro dvd; do
    run_logged "shiftnet_${domain}_plus" \
      env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$SHIFT_ENV" python "$CODE/adapters/shiftnet_infer.py" \
        --repo "$ROOT/envs/shiftnet_repo" --input "$INPUT" \
        --output "$BENCH/outputs/shiftnet_${domain}_plus/frames" \
        --checkpoint "$BENCH/weights/shiftnet/net_${domain}_deblur.pth" \
        --one-len 48 --device cuda:0 --fp16 || true
  done
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=bsstnet" ]]; then
  for domain in gopro dvd; do
    run_logged "bsstnet_${domain}" \
      env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$BSST_ENV" python "$CODE/adapters/bsstnet_infer.py" \
        --repo "$ROOT/envs/bsstnet_repo" --input "$INPUT" \
        --output "$BENCH/outputs/bsstnet_${domain}/frames" \
        --checkpoint "$BENCH/weights/bsstnet/BSST_${domain}.pth" \
        --raft-checkpoint "$BENCH/weights/bsstnet/raft-things.pth" \
        --clip-len 48 --temporal-overlap 16 --patch-size 256 --patch-overlap 64 --device cuda:0 || true
  done
fi

# Validate and encode all completed outputs.
for frames in "$BENCH"/outputs/*/frames; do
  [[ -d "$frames" ]] || continue
  model_dir=$(dirname "$frames")
  python3 "$CODE/scripts/check_output.py" --input "$INPUT" --output "$frames" --report "$model_dir/check_report.json" || continue
  python3 "$CODE/scripts/frames_to_video.py" --frames "$frames" --output "$model_dir/output.mp4" --fps "$FPS"
done

echo "Finished. Outputs: $BENCH/outputs"
