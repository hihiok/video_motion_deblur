#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
BENCH="$ROOT/benchmark"
INPUT_SRC="${INPUT_SRC:-$ROOT/input/xiaobieli38_trimmed}"
INPUT_MP4="${INPUT_MP4:-$ROOT/input/xiaobieli38_trimmed.mp4}"
INPUT="${INPUT:-$BENCH/input_frames}"
GPU="${GPU:-0}"
MODE="${1:---all}"

# The project already has this Torch 2.4/CUDA 11.8 environment.  The fixed
# DSTNet and Shift-Net adapters can run there without installing mmcv/CuPy.
COMMON_ENV="${COMMON_ENV:-turtle_joint_py222}"
REAL_ENV="${REAL_ENV:-$COMMON_ENV}"
DST_ENV="${DST_ENV:-$COMMON_ENV}"
SHIFT_ENV="${SHIFT_ENV:-$COMMON_ENV}"
BSST_ENV="${BSST_ENV:-bsstnet}"

mkdir -p "$BENCH"/{outputs,logs,manifests,videos}
python3 "$CODE/scripts/prepare_input.py" --source-frames "$INPUT_SRC" --source-mp4 "$INPUT_MP4" --output "$INPUT"
FPS=$(python3 -c "import json; print(json.load(open('$BENCH/manifests/input.json'))['fps'])")

conda_env_exists() {
  command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$1"
}

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

run_python_logged() {
  local name="$1" env_name="$2"; shift 2
  if conda_env_exists "$env_name"; then
    run_logged "$name" env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$env_name" python "$@"
  else
    echo "Conda env '$env_name' not found; using current python for $name" >&2
    run_logged "$name" env CUDA_VISIBLE_DEVICES="$GPU" python "$@"
  fi
}

require_file() {
  [[ -s "$1" ]] || { echo "SKIP: missing $1" >&2; return 1; }
}

if [[ "$MODE" == "--download-only" ]]; then
  ROOT="$ROOT" bash "$CODE/scripts/download_weights.sh"
  exit
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=realvdeblur" ]]; then
  if require_file "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" && \
     require_file "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" && \
     require_file "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"; then
    run_python_logged realvdeblur "$REAL_ENV" "$CODE/adapters/realvdeblur_infer.py" \
      --repo "$ROOT/envs/realvdeblur_repo" --input "$INPUT" \
      --output "$BENCH/outputs/realvdeblur_dmd/frames" \
      --wan-model-dir "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B" \
      --checkpoint "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" \
      --device cuda:0 --dtype float16 --temporal-window-size 21 || true
  fi
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=dstnet" ]]; then
  for domain in GOPRO DVD BSD; do
    lname=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
    checkpoint="$BENCH/weights/dstnet/${domain}.pth"
    require_file "$checkpoint" || continue
    # Float32 is the conservative default for the pure-PyTorch dynamic-conv fallback.
    run_python_logged "dstnet_${lname}" "$DST_ENV" "$CODE/adapters/dstnet_infer.py" \
      --repo "$ROOT/envs/dstnet_repo" --input "$INPUT" \
      --output "$BENCH/outputs/dstnet_${lname}/frames" \
      --checkpoint "$checkpoint" \
      --clip-len 30 --overlap 10 --device cuda:0 || true
  done
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=shiftnet" ]]; then
  for domain in gopro dvd; do
    checkpoint="$BENCH/weights/shiftnet/net_${domain}_deblur.pth"
    require_file "$checkpoint" || continue
    run_python_logged "shiftnet_${domain}_plus" "$SHIFT_ENV" "$CODE/adapters/shiftnet_infer.py" \
      --repo "$ROOT/envs/shiftnet_repo" --input "$INPUT" \
      --output "$BENCH/outputs/shiftnet_${domain}_plus/frames" \
      --checkpoint "$checkpoint" \
      --one-len 48 --device cuda:0 --fp16 || true
  done
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=bsstnet" ]]; then
  for domain in gopro dvd; do
    checkpoint="$BENCH/weights/bsstnet/BSST_${domain}.pth"
    raft="$BENCH/weights/bsstnet/raft-things.pth"
    require_file "$checkpoint" || continue
    require_file "$raft" || continue
    run_python_logged "bsstnet_${domain}" "$BSST_ENV" "$CODE/adapters/bsstnet_infer.py" \
      --repo "$ROOT/envs/bsstnet_repo" --input "$INPUT" \
      --output "$BENCH/outputs/bsstnet_${domain}/frames" \
      --checkpoint "$checkpoint" --raft-checkpoint "$raft" \
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
