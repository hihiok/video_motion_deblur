#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
CODE="${CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${DSTNETPLUS_ENV:-deblur_runtime}"
GPU="${GPU:-0}"
INPUT_FRAMES="${INPUT_FRAMES:-$ROOT/input/xiaobieli38_trimmed}"
INPUT_MP4="${INPUT_MP4:-$ROOT/input/xiaobieli38_trimmed.mp4}"
REPO="${DSTNETPLUS_REPO:-$ROOT/envs/dstnetplus_repo}"
REPO_COMMIT="54363c15d8b924aa1ae56b8f835c1f0289954e95"
WEIGHT_DIR="$ROOT/benchmark/weights/dstnetplus_2025"
CHECKPOINT="$WEIGHT_DIR/DSTNetPlus_base_gopro.pth"
OUTPUT_DIR="$ROOT/benchmark/outputs/dstnetplus_base_gopro_2025"
FRAMES_OUT="$OUTPUT_DIR/frames"
LOG_DIR="$ROOT/benchmark/logs"
LOG="$LOG_DIR/dstnetplus_base_gopro_2025_full.log"

CLIP_LEN="${DSTNETPLUS_CLIP_LEN:-8}"
TEMPORAL_OVERLAP="${DSTNETPLUS_TEMPORAL_OVERLAP:-2}"
TILE_SIZE="${DSTNETPLUS_TILE_SIZE:-384}"
TILE_OVERLAP="${DSTNETPLUS_TILE_OVERLAP:-64}"
MIN_TILE_SIZE="${DSTNETPLUS_MIN_TILE_SIZE:-192}"

mkdir -p "$WEIGHT_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$ROOT/envs"

# Trusted internal TLS inspection environment: do not fail on certificate checks.
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
export PYTHONHTTPSVERIFY=0

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Missing conda env: $ENV_NAME" >&2
  exit 1
fi

if [[ ! -d "$REPO/.git" ]]; then
  rm -rf "$REPO"
  GIT_SSL_NO_VERIFY=true git clone https://github.com/sunny2109/DSTNet-plus.git "$REPO"
fi

cd "$REPO"
git config http.sslVerify false
git fetch --all --tags || true
git checkout "$REPO_COMMIT"
cd "$CODE"

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Downloading official DSTNet+ Base GoPro checkpoint..."
  curl -k -L --retry 5 --retry-delay 3 -C - \
    -o "$CHECKPOINT" \
    https://github.com/sunny2109/DSTNet-plus/releases/download/v0.1.0/DSTNetPlus_base_gopro.pth
fi

# Official release asset is 15,704,138 bytes. Reject obviously incomplete/error files.
SIZE=$(stat -c%s "$CHECKPOINT")
if (( SIZE < 15000000 )); then
  echo "Checkpoint too small: $CHECKPOINT ($SIZE bytes)" >&2
  exit 1
fi

# Validate checkpoint can be parsed before spending GPU time.
conda run -n "$ENV_NAME" python - <<PY
import torch
p = r"$CHECKPOINT"
x = torch.load(p, map_location="cpu")
if isinstance(x, dict):
    print("checkpoint keys:", list(x.keys())[:10])
else:
    print("checkpoint type:", type(x))
print("checkpoint load: PASS")
PY

COUNT=$(find "$INPUT_FRAMES" -maxdepth 1 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | wc -l)
echo "Input frames: $COUNT"
if [[ "$COUNT" -ne 452 ]]; then
  echo "Expected exactly 452 business frames, got $COUNT" >&2
  exit 1
fi

rm -rf "$FRAMES_OUT"
mkdir -p "$FRAMES_OUT"

nvidia-smi || true

echo "Running DSTNet+ Base 2025 (GoPro checkpoint)..."
echo "clip=$CLIP_LEN overlap=$TEMPORAL_OVERLAP tile=$TILE_SIZE tile_overlap=$TILE_OVERLAP min_tile=$MIN_TILE_SIZE"

set +e
env CUDA_VISIBLE_DEVICES="$GPU" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64" \
  conda run -n "$ENV_NAME" python "$CODE/adapters/dstnetplus_infer.py" \
    --repo "$REPO" \
    --input "$INPUT_FRAMES" \
    --output "$FRAMES_OUT" \
    --checkpoint "$CHECKPOINT" \
    --clip-len "$CLIP_LEN" \
    --overlap "$TEMPORAL_OVERLAP" \
    --tile-size "$TILE_SIZE" \
    --tile-overlap "$TILE_OVERLAP" \
    --min-tile-size "$MIN_TILE_SIZE" \
    --device cuda:0 \
    --amp 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
if [[ "$STATUS" -ne 0 ]]; then
  echo "DSTNet+ inference failed; see $LOG" >&2
  exit "$STATUS"
fi

python3 "$CODE/scripts/check_output.py" \
  --input "$INPUT_FRAMES" \
  --output "$FRAMES_OUT" \
  --report "$OUTPUT_DIR/check_report.json"

python3 - <<PY
import json
p = r"$OUTPUT_DIR/check_report.json"
d = json.load(open(p))
if not d.get("passed") or d.get("output_count") != 452:
    raise SystemExit(f"Output validation failed: {d}")
print("452-frame validation: PASS")
PY

FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
  -of default=noprint_wrappers=1:nokey=1 "$INPUT_MP4")
FPS_NUM=$(python3 - <<PY
from fractions import Fraction
print(float(Fraction("$FPS")))
PY
)
echo "Input FPS: $FPS_NUM"

python3 "$CODE/scripts/frames_to_video.py" \
  --frames "$FRAMES_OUT" \
  --output "$OUTPUT_DIR/output.mp4" \
  --fps "$FPS_NUM"

ls -lh "$OUTPUT_DIR/output.mp4"
sha256sum "$CHECKPOINT" > "$OUTPUT_DIR/checkpoint.sha256"

echo "DSTNet+ Base 2025 COMPLETE"
echo "Frames: $FRAMES_OUT"
echo "Video:  $OUTPUT_DIR/output.mp4"
echo "Report: $OUTPUT_DIR/check_report.json"
echo "Meta:   $OUTPUT_DIR/run_metadata.json"
