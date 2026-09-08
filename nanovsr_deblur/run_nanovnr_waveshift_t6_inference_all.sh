#!/usr/bin/env bash
set -euo pipefail

source /mnt/ssd1/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate RVRT
source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" 2>/dev/null || true
git config --global http.sslVerify false

ROOT=/mnt/ssd1/z00919662/motion_deblur
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT=${CHECKPOINT:-$ROOT/runs/nanovnr_waveshift_pagf_fullframe_t6_bsd3ms24ms_20260907/amp_recovery_20260907/train/step_0150000.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runs/nanovnr_waveshift_t6_center_inference_20260908}
DATASET_MAX_FRAMES=${DATASET_MAX_FRAMES:-120}
GOPRO=${GOPRO:-$ROOT/datasets/GoPro}
DVD=${DVD:-$ROOT/datasets/DVD}
BSD=${BSD:-/mnt/ssd1/z00919662/datasets/BSD/BSD_3ms24ms}
BUSINESS_VIDEO=${BUSINESS_VIDEO:-$ROOT/input/xiaobieli38_trimmed.mp4}

test -s "$CHECKPOINT"
test -d "$GOPRO"
test -d "$DVD"
test -d "$BSD/test"
test -s "$BUSINESS_VIDEO"
if [ -e "$OUTPUT_DIR" ]; then
  echo "Refusing existing OUTPUT_DIR: $OUTPUT_DIR"
  exit 2
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(python - <<'PY'
import csv
import subprocess

def query(kind, fields):
    return subprocess.check_output(
        ['nvidia-smi', '--query-' + kind + '=' + fields,
         '--format=csv,noheader,nounits'], text=True)

busy = {line.strip() for line in query('compute-apps', 'gpu_uuid').splitlines()
        if line.strip()}
rows = csv.reader(query('gpu', 'index,uuid,memory.free').splitlines())
candidates = [(int(free.strip()), uuid.strip()) for index, uuid, free in rows
              if index.strip() != '6' and uuid.strip() not in busy]
if not candidates:
    raise SystemExit('No free GPU other than physical GPU 6.')
print(max(candidates)[1])
PY
  )
  export CUDA_VISIBLE_DEVICES
fi

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"
COMMON=(--checkpoint "$CHECKPOINT")

python infer_nanovnr_waveshift_t6_center.py "${COMMON[@]}" \
  --dataset-root "$GOPRO" --family GoPro --split test --sequence-index 0 \
  --max-frames "$DATASET_MAX_FRAMES" --output-dir "$OUTPUT_DIR/gopro" \
  2>&1 | tee "$OUTPUT_DIR/gopro.log"

python infer_nanovnr_waveshift_t6_center.py "${COMMON[@]}" \
  --dataset-root "$BSD" --family BSD --split test --sequence-index 0 \
  --max-frames "$DATASET_MAX_FRAMES" --output-dir "$OUTPUT_DIR/bsd" \
  2>&1 | tee "$OUTPUT_DIR/bsd.log"

python infer_nanovnr_waveshift_t6_center.py "${COMMON[@]}" \
  --dataset-root "$DVD" --family DVD --split test --sequence-index 0 \
  --max-frames "$DATASET_MAX_FRAMES" --output-dir "$OUTPUT_DIR/dvd" \
  2>&1 | tee "$OUTPUT_DIR/dvd.log"

python infer_nanovnr_waveshift_t6_center.py "${COMMON[@]}" \
  --input-video "$BUSINESS_VIDEO" --output-dir "$OUTPUT_DIR/business" \
  2>&1 | tee "$OUTPUT_DIR/business.log"

export WAVESHIFT_INFERENCE_OUTPUT="$OUTPUT_DIR"
python - <<'PY' | tee "$OUTPUT_DIR/FINAL_INFERENCE_REPORT.txt"
import json
import os
from pathlib import Path

root = Path(os.environ['WAVESHIFT_INFERENCE_OUTPUT'])
print('WAVESHIFT_T6_CENTER_INFERENCE_COMPLETE')
print('STATE_POLICY=RESET_PER_TARGET_WINDOW')
print('WINDOW=6 SELECTED_POSITION=3 PAST=3 FUTURE=2 PRECISION=FP32')
any_red_flag = False
for name in ('gopro', 'bsd', 'dvd', 'business'):
    summary = json.loads((root / name / 'summary.json').read_text())
    any_red_flag |= summary['status'] != 'PASS'
    print(
        f"{name.upper()} status={summary['status']} frames={summary['frames']} "
        f"size={summary['width']}x{summary['height']} "
        f"psnr={summary['mean_psnr_rgb']} dark={summary['output_dark_clip_rate']:.6f} "
        f"shift_rgb={summary['mean_shift_rgb']} "
        f"peak_gpu_gib={summary['peak_gpu_gib']:.3f}"
    )
print('AUTOMATED_DARK_OUTPUT_RED_FLAG=' + ('YES' if any_red_flag else 'NO'))
print('HUMAN_ACTION_REQUIRED=YES')
print('HUMAN_ACTION=inspect all four preview.jpg and comparison MP4 files')
PY

echo "OUTPUT_DIR=$OUTPUT_DIR"
