#!/usr/bin/env bash
# All three COMPLETE local test sets, no training, no first-two or max-clips limit.
# BSD GT must already have been verified/restored by benchmark_v2.bsd_gt.
# Usage: CONFIG=/absolute/datasets.json BENCH=/absolute/new_result_dir \
#        GPU=6 bash run_nanovnr_fair_benchmark.sh
# Optional: CONTEXT=16 for strict same-input-window comparison (slower).
# Optional: SHIFT_REPO=/official/Shift-Net SHIFT_CHECKPOINT=/verified/ours_small.pth
#           SHIFT_METADATA=/absolute/shift_metadata.json to run an existing comparator.
# Existing environment only; NO pip, curl, downloads, source edits or dataset deletion.
set +x
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/nanovnr_nafnet_rgb_fullframe_bsd_train_test_20260904/train/step_0125000.pth}"
: "${CONFIG:?Set CONFIG to verified three-dataset JSON (see CodeAgent MD)}"
: "${BENCH:?Set BENCH to the output directory; reuse it only for resume of the same experiment}"
CONTEXT="${CONTEXT:-0}"
PRECISION="${PRECISION:-fp32}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != deblur_runtime ]]; then
    CONDA_SH=""
    if command -v conda >/dev/null 2>&1; then
      CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    else
      for base in /mnt/ssd1/z00919662/anaconda3 "$HOME/anaconda3" "$HOME/miniconda3"; do
        if [[ -f "$base/etc/profile.d/conda.sh" ]]; then CONDA_SH="$base/etc/profile.d/conda.sh"; break; fi
      done
    fi
    [[ -f "$CONDA_SH" ]] || { echo 'Activate deblur_runtime or set PYTHON_BIN.' >&2; exit 2; }
    set +u
    source "$CONDA_SH"
    conda activate deblur_runtime
    set -u
  fi
  PYTHON_BIN="$(command -v python)"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ -n "${GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
cd "$CODE"
"$PYTHON_BIN" -c 'import torch, PIL, numpy; assert torch.cuda.is_available(), "CUDA unavailable"'
[[ -s "$CONFIG" && -s "$CHECKPOINT" ]] || { echo 'Missing config/checkpoint' >&2; exit 2; }
mkdir -p "$BENCH"
if [[ ! -f "$BENCH/manifest.json" ]]; then
  "$PYTHON_BIN" -m benchmark_v2.data --config "$CONFIG" --out "$BENCH/manifest.json" | tee "$BENCH/manifest.log"
else
  "$PYTHON_BIN" - "$CONFIG" "$BENCH/manifest.json" <<'PY'
import sys
from benchmark_v2.data import read_json, load_manifest
assert read_json(sys.argv[1]) == load_manifest(sys.argv[2])['configuration'], 'Changed config: use a new BENCH directory'
PY
fi
if [[ ! -f "$BENCH/nano_metadata.json" ]]; then
  cat > "$BENCH/nano_metadata.json" <<'JSON'
{
  "training_data": ["GoPro/train", "DVD/train", "BSD/direct_train"],
  "checkpoint_origin": "User-reported NanoVNRNAFNetRGB native-fullframe mixed training; verify actual training log separately",
  "checkpoint_selection": "125k checkpoint previously selected using a GoPro test subset; NOT an untouched final holdout",
  "policy": "Frozen before this evaluation; no tuning, no retraining, no GT-conditioned output correction"
}
JSON
fi
"$PYTHON_BIN" -m benchmark_v2.evaluate run --manifest "$BENCH/manifest.json" \
  --metadata "$BENCH/nano_metadata.json" --method nano --checkpoint "$CHECKPOINT" \
  --out "$BENCH/nano" --device cuda:0 --precision "$PRECISION" --chunk 15 --context "$CONTEXT" --resume \
  2>&1 | tee -a "$BENCH/nano.log"
if [[ -n "${SHIFT_REPO:-}" || -n "${SHIFT_CHECKPOINT:-}" || -n "${SHIFT_METADATA:-}" ]]; then
  : "${SHIFT_REPO:?All three Shift-Net variables are required}"
  : "${SHIFT_CHECKPOINT:?All three Shift-Net variables are required}"
  : "${SHIFT_METADATA:?All three Shift-Net variables are required}"
  "$PYTHON_BIN" -m benchmark_v2.evaluate run --manifest "$BENCH/manifest.json" \
    --metadata "$SHIFT_METADATA" --method shift-small --shift-repo "$SHIFT_REPO" \
    --checkpoint "$SHIFT_CHECKPOINT" --out "$BENCH/shift_small" --device cuda:0 \
    --precision "$PRECISION" --one-len 16 --context "$CONTEXT" --resume \
    2>&1 | tee -a "$BENCH/shift_small.log"
  if [[ ! -f "$BENCH/comparison.json" ]]; then
    "$PYTHON_BIN" -m benchmark_v2.evaluate compare \
      --reports "$BENCH/nano/summary.json" "$BENCH/shift_small/summary.json" --out "$BENCH/comparison.json"
  fi
fi
printf 'RESULTS=%s\n' "$BENCH"
