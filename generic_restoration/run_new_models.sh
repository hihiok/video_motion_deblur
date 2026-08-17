#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
MODEL="${2:-all}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 smoke|full [all|flashvsr|seedvr2|dove]" >&2
  exit 2
fi
if [[ "$MODEL" == "all" ]]; then
  status=0
  for name in flashvsr seedvr2 dove; do
    if ! "$0" "$MODE" "$name"; then
      echo "INFERENCE_FAILED: $name" >&2
      status=1
    fi
  done
  ROOT="${ROOT:-/data/pub1/z00919662/motion_deblur/generic_restoration}"
  CODE="${CODE:-$ROOT/benchmark_code}"
  CANONICAL="$ROOT/canonical"
  FRAME_DIR="$CANONICAL/frames_${MODE}"
  PREVIEW_ARGS=()
  [[ -d "$ROOT/runs/flashvsr_${MODE}/frames" ]] && PREVIEW_ARGS+=(--model "FlashVSR-v1.1-1x=$ROOT/runs/flashvsr_${MODE}/frames")
  [[ -d "$ROOT/runs/seedvr2_${MODE}/frames" ]] && PREVIEW_ARGS+=(--model "SeedVR2-3B-1x=$ROOT/runs/seedvr2_${MODE}/frames")
  [[ -d "$ROOT/runs/dove_${MODE}/frames" ]] && PREVIEW_ARGS+=(--model "DOVE-Final-1x=$ROOT/runs/dove_${MODE}/frames")
  if [[ ${#PREVIEW_ARGS[@]} -gt 0 ]]; then
    PYTHONPATH="$CODE" python -m generic_restoration.make_preview \
      --input-frames "$FRAME_DIR" \
      "${PREVIEW_ARGS[@]}" \
      --output "$ROOT/runs/new_models_${MODE}_combined_preview.jpg"
  fi
  if [[ "$MODE" == "smoke" ]]; then
    echo "STOP FOR MANUAL REVIEW: $ROOT/runs/new_models_smoke_combined_preview.jpg"
  fi
  exit "$status"
fi
if [[ "$MODEL" != "flashvsr" && "$MODEL" != "seedvr2" && "$MODEL" != "dove" ]]; then
  echo "Usage: $0 smoke|full [all|flashvsr|seedvr2|dove]" >&2
  exit 2
fi

ROOT="${ROOT:-/data/pub1/z00919662/motion_deblur/generic_restoration}"
CODE="${CODE:-$ROOT/benchmark_code}"
CONDA_BASE="${CONDA_BASE:-/data/pub1/z00919662/anaconda3}"
INPUT_VIDEO="${INPUT_VIDEO:-/data/pub1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4}"
GPU="${GPU:-0}"
CANONICAL="$ROOT/canonical"
RUN_DIR="$ROOT/runs/${MODEL}_${MODE}"
FLASH_REPO="$ROOT/envs/FlashVSR"
SEED_REPO="$ROOT/envs/SeedVR"
DOVE_REPO="$ROOT/envs/DOVE"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "STOP: input business stream does not exist: $INPUT_VIDEO" >&2
  echo "Manual action: set INPUT_VIDEO to the exact same MP4 used on the old server." >&2
  exit 3
fi
if [[ "$MODE" == "full" && ! -f "$ROOT/APPROVE_NEW_MODELS_FULL" ]]; then
  echo "STOP: combined new-server smoke preview has not been approved." >&2
  echo "Manual action after reviewing it: touch $ROOT/APPROVE_NEW_MODELS_FULL" >&2
  exit 4
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true PYTHONHTTPSVERIFY=0 SSL_NO_VERIFY=1 HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

archive_existing() {
  local target="$1"
  if [[ -e "$target" ]]; then
    mv "$target" "${target}.previous.$(date +%Y%m%d_%H%M%S)"
  fi
}

conda activate flashvsr_blackwell
if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi
cd "$CODE"
python -m generic_restoration.prepare_stream \
  --input-video "$INPUT_VIDEO" \
  --work-dir "$CANONICAL" \
  --smoke-frames 25

FRAME_DIR="$CANONICAL/frames_${MODE}"
MANIFEST="$CANONICAL/manifest${MODE/smoke/_smoke}.json"
if [[ "$MODE" == "full" ]]; then
  MANIFEST="$CANONICAL/manifest.json"
fi
mkdir -p "$RUN_DIR"

if [[ "$MODEL" == "flashvsr" ]]; then
  conda activate flashvsr_blackwell
  cd "$CODE"
  python -m generic_restoration.flashvsr_v11_business \
    --repo "$FLASH_REPO" \
    --weights "$ROOT/weights/FlashVSR-v1.1" \
    --input-frames "$FRAME_DIR" \
    --output-frames "$RUN_DIR/frames" \
    --metadata "$RUN_DIR/run_metadata.json" \
    --scale "${FLASHVSR_SCALE:-1.0}" \
    --seed "${FLASHVSR_SEED:-0}" \
    --sparse-ratio "${FLASHVSR_SPARSE_RATIO:-2.0}" \
    --kv-ratio "${FLASHVSR_KV_RATIO:-3.0}" \
    --local-range "${FLASHVSR_LOCAL_RANGE:-11}"
fi

if [[ "$MODEL" == "seedvr2" ]]; then
  conda activate seedvr2_blackwell
  CLIPS="$RUN_DIR/input_clips"
  CHUNK_MANIFEST="$RUN_DIR/chunks.json"
  RAW_OUTPUT="$RUN_DIR/seed_raw"
  archive_existing "$RAW_OUTPUT"
  mkdir -p "$RAW_OUTPUT"
  cd "$CODE"
  python -m generic_restoration.prepare_seedvr2_clips \
    --frames "$FRAME_DIR" \
    --manifest "$MANIFEST" \
    --output-dir "$CLIPS" \
    --chunk-manifest "$CHUNK_MANIFEST" \
    --core-frames "${SEEDVR2_CORE_FRAMES:-49}" \
    --context-frames "${SEEDVR2_CONTEXT_FRAMES:-4}"
  read -r HEIGHT WIDTH FPS < <(python - "$MANIFEST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["height"], d["width"], d["fps"])
PY
)
  cd "$SEED_REPO"
  torchrun --nproc-per-node=1 projects/inference_seedvr2_3b.py \
    --video_path "$CLIPS" \
    --output_dir "$RAW_OUTPUT" \
    --seed "${SEEDVR2_SEED:-666}" \
    --res_h "$HEIGHT" \
    --res_w "$WIDTH" \
    --sp_size 1 \
    --out_fps "$FPS"
  cd "$CODE"
  python -m generic_restoration.stitch_seedvr2_png \
    --chunk-manifest "$CHUNK_MANIFEST" \
    --seed-output "$RAW_OUTPUT" \
    --output-frames "$RUN_DIR/frames" \
    --report "$RUN_DIR/stitch_report.json"
  python -m generic_restoration.record_external_run \
    --model "SeedVR2-3B" \
    --repo "$SEED_REPO" \
    --output-frames "$RUN_DIR/frames" \
    --metadata "$RUN_DIR/run_metadata.json" \
    --mode "native direct 1x restoration" \
    --weights "$ROOT/weights/SeedVR2-3B"
fi

if [[ "$MODEL" == "dove" ]]; then
  conda activate dove_blackwell
  INPUT_DIR="$RUN_DIR/dove_input"
  RAW_OUTPUT="$RUN_DIR/dove_raw"
  archive_existing "$RAW_OUTPUT"
  mkdir -p "$INPUT_DIR" "$RAW_OUTPUT"
  if [[ "$MODE" == "smoke" ]]; then
    MODEL_VIDEO="$CANONICAL/business_smoke_lossless.mkv"
    DOVE_CHUNK_LEN="${DOVE_CHUNK_LEN_SMOKE:-0}"
  else
    MODEL_VIDEO="$CANONICAL/business_full_lossless.mkv"
    DOVE_CHUNK_LEN="${DOVE_CHUNK_LEN_FULL:-33}"
  fi
  ln -sfn "$MODEL_VIDEO" "$INPUT_DIR/business.mkv"
  read -r FPS_INT < <(python - "$MANIFEST" <<'PY'
import json, sys
print(max(1, round(json.load(open(sys.argv[1]))["fps"])))
PY
)
  cd "$DOVE_REPO"
  python inference_script.py \
    --input_dir "$INPUT_DIR" \
    --model_path "$DOVE_REPO/pretrained_models/DOVE" \
    --output_path "$RAW_OUTPUT" \
    --fps "$FPS_INT" \
    --dtype bfloat16 \
    --upscale 1 \
    --is_vae_st \
    --png_save \
    --chunk_len "$DOVE_CHUNK_LEN" \
    --overlap_t "${DOVE_OVERLAP_T:-8}" \
    --tile_size_hw "${DOVE_TILE_H:-0}" "${DOVE_TILE_W:-0}" \
    --overlap_hw "${DOVE_OVERLAP_H:-64}" "${DOVE_OVERLAP_W:-64}"
  if [[ ! -d "$RAW_OUTPUT/business" ]]; then
    echo "DOVE output folder missing: $RAW_OUTPUT/business" >&2
    exit 5
  fi
  archive_existing "$RUN_DIR/frames"
  ln -sfn "$RAW_OUTPUT/business" "$RUN_DIR/frames"
  cd "$CODE"
  python -m generic_restoration.record_external_run \
    --model "DOVE Final" \
    --repo "$DOVE_REPO" \
    --output-frames "$RUN_DIR/frames" \
    --metadata "$RUN_DIR/run_metadata.json" \
    --mode "official --upscale 1 with corrected padding crop" \
    --weights "$DOVE_REPO/pretrained_models/DOVE"
fi

cd "$CODE"
python -m generic_restoration.finalize_video \
  --manifest "$MANIFEST" \
  --frames "$RUN_DIR/frames" \
  --output "$RUN_DIR/output_1x.mp4" \
  --report "$RUN_DIR/check_report.json"
python -m generic_restoration.make_preview \
  --input-frames "$FRAME_DIR" \
  --model "$MODEL=$RUN_DIR/frames" \
  --output "$RUN_DIR/preview_input_output.jpg"

echo "${MODEL^^}_${MODE^^}_PASS"
