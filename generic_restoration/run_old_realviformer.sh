#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 smoke|full" >&2
  exit 2
fi

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur/generic_restoration}"
CODE="${CODE:-$ROOT/benchmark_code}"
CONDA_BASE="${CONDA_BASE:-/mnt/ssd1/z00919662/anaconda3}"
ENV_NAME="${ENV_NAME:-realviformer_rwvsr}"
REPO="${REALVIFORMER_REPO:-$ROOT/envs/RealViformer}"
CHECKPOINT="${REALVIFORMER_CHECKPOINT:-$ROOT/weights/realviformer/weights.pth}"
INPUT_VIDEO="${INPUT_VIDEO:-/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4}"
GPU="${GPU:-0}"
CANONICAL="$ROOT/canonical"
RUN_DIR="$ROOT/runs/realviformer_${MODE}"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "STOP: input business stream does not exist: $INPUT_VIDEO" >&2
  echo "Manual action: set INPUT_VIDEO to the same MP4 on both servers." >&2
  exit 3
fi
if [[ "$MODE" == "full" && ! -f "$ROOT/APPROVE_REALVIFORMER_FULL" ]]; then
  echo "STOP: smoke preview has not been approved." >&2
  echo "Manual action after reviewing the smoke JPG: touch $ROOT/APPROVE_REALVIFORMER_FULL" >&2
  exit 4
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true PYTHONHTTPSVERIFY=0 HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

cd "$CODE"
python -m generic_restoration.prepare_stream \
  --input-video "$INPUT_VIDEO" \
  --work-dir "$CANONICAL" \
  --smoke-frames 25

if [[ "$MODE" == "smoke" ]]; then
  FRAME_DIR="$CANONICAL/frames_smoke"
  MANIFEST="$CANONICAL/manifest_smoke.json"
else
  FRAME_DIR="$CANONICAL/frames_full"
  MANIFEST="$CANONICAL/manifest.json"
fi

mkdir -p "$RUN_DIR"
python -m generic_restoration.realviformer_business \
  --repo "$REPO" \
  --checkpoint "$CHECKPOINT" \
  --input-frames "$FRAME_DIR" \
  --output-frames "$RUN_DIR/frames" \
  --metadata "$RUN_DIR/run_metadata.json" \
  --tile "${REALVIFORMER_TILE:-256}" \
  --overlap "${REALVIFORMER_OVERLAP:-48}" \
  --core-frames "${REALVIFORMER_CORE_FRAMES:-8}" \
  --warmup-frames "${REALVIFORMER_WARMUP_FRAMES:-4}" \
  --precision "${REALVIFORMER_PRECISION:-fp16}"

python -m generic_restoration.finalize_video \
  --manifest "$MANIFEST" \
  --frames "$RUN_DIR/frames" \
  --output "$RUN_DIR/output_1x.mp4" \
  --report "$RUN_DIR/check_report.json"

python -m generic_restoration.make_preview \
  --input-frames "$FRAME_DIR" \
  --model "RealViformer-4x-to-1x=$RUN_DIR/frames" \
  --output "$RUN_DIR/preview_input_output.jpg"

if [[ "$MODE" == "smoke" ]]; then
  echo "REALVIFORMER_SMOKE_PASS"
  echo "STOP FOR MANUAL REVIEW: $RUN_DIR/preview_input_output.jpg"
else
  echo "REALVIFORMER_FULL_PASS"
fi
