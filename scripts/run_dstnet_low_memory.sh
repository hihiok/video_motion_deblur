#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
CODE="${CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${DST_ENV:-deblur_runtime}"
GPU="${GPU:-0}"
BENCH="$ROOT/benchmark"
INPUT_SRC="${INPUT_SRC:-$ROOT/input/xiaobieli38_trimmed}"
INPUT_MP4="${INPUT_MP4:-$ROOT/input/xiaobieli38_trimmed.mp4}"
INPUT="${INPUT:-$BENCH/input_frames}"
CLIP_LEN="${DST_CLIP_LEN:-4}"
TEMPORAL_OVERLAP="${DST_TEMPORAL_OVERLAP:-1}"
TILE_SIZE="${DST_TILE_SIZE:-512}"
TILE_OVERLAP="${DST_TILE_OVERLAP:-64}"
MIN_TILE_SIZE="${DST_MIN_TILE_SIZE:-256}"

mkdir -p "$BENCH"/{outputs,logs,manifests}

python3 "$CODE/scripts/prepare_input.py" \
  --source-frames "$INPUT_SRC" \
  --source-mp4 "$INPUT_MP4" \
  --output "$INPUT"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "DSTNet conda environment not found: $ENV_NAME" >&2
  exit 1
fi

status=0
for domain in GOPRO DVD BSD; do
  lname=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
  checkpoint="$BENCH/weights/dstnet/${domain}.pth"
  output="$BENCH/outputs/dstnet_${lname}/frames"
  log="$BENCH/logs/dstnet_${lname}_low_memory.log"

  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing DSTNet checkpoint: $checkpoint" >&2
    status=1
    continue
  fi

  rm -rf "$output"
  mkdir -p "$output"

  echo "===== DSTNet $domain: clip=$CLIP_LEN tile=$TILE_SIZE ====="
  if ! env CUDA_VISIBLE_DEVICES="$GPU" \
      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" \
      conda run -n "$ENV_NAME" python "$CODE/adapters/dstnet_infer.py" \
        --repo "$ROOT/envs/dstnet_repo" \
        --input "$INPUT" \
        --output "$output" \
        --checkpoint "$checkpoint" \
        --clip-len "$CLIP_LEN" \
        --overlap "$TEMPORAL_OVERLAP" \
        --tile-size "$TILE_SIZE" \
        --tile-overlap "$TILE_OVERLAP" \
        --min-tile-size "$MIN_TILE_SIZE" \
        --device cuda:0 \
        --amp 2>&1 | tee "$log"; then
    echo "DSTNet $domain failed; see $log" >&2
    status=1
    continue
  fi

  python3 "$CODE/scripts/check_output.py" \
    --input "$INPUT" \
    --output "$output" \
    --report "$(dirname "$output")/check_report.json" || status=1
done

exit "$status"
