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

COMMON_ENV="${COMMON_ENV:-turtle_joint_py222}"
REAL_ENV="${REAL_ENV:-$COMMON_ENV}"
DST_ENV="${DST_ENV:-$COMMON_ENV}"
SHIFT_ENV="${SHIFT_ENV:-$COMMON_ENV}"
BSST_ENV="${BSST_ENV:-bsstnet}"

overall_status=0
mkdir -p "$BENCH"/{outputs,logs,manifests,videos}
python3 "$CODE/scripts/prepare_input.py" \
  --source-frames "$INPUT_SRC" \
  --source-mp4 "$INPUT_MP4" \
  --output "$INPUT" || exit 1
FPS=$(python3 -c "import json; print(json.load(open('$BENCH/manifests/input.json'))['fps'])")

conda_env_exists() {
  command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$1"
}

run_logged() {
  local name="$1"; shift
  echo "===== $name ====="
  if "$@" 2>&1 | tee "$BENCH/logs/${name}.log"; then
    echo "$name: complete"
    return 0
  fi
  echo "$name: FAILED; see $BENCH/logs/${name}.log" >&2
  return 1
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

fresh_output() {
  local output="$1"
  rm -rf "$output"
  mkdir -p "$output"
}

if [[ "$MODE" == "--download-only" ]]; then
  ROOT="$ROOT" bash "$CODE/scripts/download_weights.sh"
  exit
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=realvdeblur" ]]; then
  if require_file "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" && \
     require_file "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" && \
     require_file "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"; then
    out="$BENCH/outputs/realvdeblur_dmd/frames"
    fresh_output "$out"
    run_python_logged realvdeblur "$REAL_ENV" "$CODE/adapters/realvdeblur_infer.py" \
      --repo "$ROOT/envs/realvdeblur_repo" --input "$INPUT" \
      --output "$out" \
      --wan-model-dir "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B" \
      --checkpoint "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" \
      --device cuda:0 --dtype float16 --temporal-window-size 21 || overall_status=1
  else
    overall_status=1
  fi
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=dstnet" ]]; then
  for domain in GOPRO DVD BSD; do
    lname=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
    checkpoint="$BENCH/weights/dstnet/${domain}.pth"
    if ! require_file "$checkpoint"; then
      overall_status=1
      continue
    fi
    out="$BENCH/outputs/dstnet_${lname}/frames"
    fresh_output "$out"
    run_python_logged "dstnet_${lname}" "$DST_ENV" "$CODE/adapters/dstnet_infer.py" \
      --repo "$ROOT/envs/dstnet_repo" --input "$INPUT" \
      --output "$out" \
      --checkpoint "$checkpoint" \
      --clip-len 30 --overlap 10 --device cuda:0 || overall_status=1
  done
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=shiftnet" ]]; then
  for domain in gopro dvd; do
    checkpoint="$BENCH/weights/shiftnet/net_${domain}_deblur.pth"
    if ! require_file "$checkpoint"; then
      overall_status=1
      continue
    fi
    out="$BENCH/outputs/shiftnet_${domain}_plus/frames"
    fresh_output "$out"
    run_python_logged "shiftnet_${domain}_plus" "$SHIFT_ENV" "$CODE/adapters/shiftnet_infer.py" \
      --repo "$ROOT/envs/shiftnet_repo" --input "$INPUT" \
      --output "$out" \
      --checkpoint "$checkpoint" \
      --one-len 48 --device cuda:0 --fp16 || overall_status=1
  done
fi

if [[ "$MODE" == "--all" || "$MODE" == "--model=bsstnet" ]]; then
  # BSSTNet requires its official Torch 1.9.1 + torchvision 0.10.1 stack.
  # Never fall back to the shared/current Python because torchvision.ops.dcn
  # must be compiled for the matching Torch/CUDA version.
  if ! conda_env_exists "$BSST_ENV"; then
    echo "BSSTNet requires conda env '$BSST_ENV'; refusing current-Python fallback." >&2
    echo "Create it with the official environment commands in docs/MANUAL_RUNTIME_FIXES.md." >&2
    overall_status=1
  else
    for domain in gopro dvd; do
      checkpoint="$BENCH/weights/bsstnet/BSST_${domain}.pth"
      raft="$BENCH/weights/bsstnet/raft-things.pth"
      if ! require_file "$checkpoint" || ! require_file "$raft"; then
        overall_status=1
        continue
      fi
      out="$BENCH/outputs/bsstnet_${domain}/frames"
      fresh_output "$out"
      run_logged "bsstnet_${domain}" \
        env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$BSST_ENV" python \
        "$CODE/adapters/bsstnet_infer.py" \
        --repo "$ROOT/envs/bsstnet_repo" --input "$INPUT" \
        --output "$out" \
        --checkpoint "$checkpoint" --raft-checkpoint "$raft" \
        --clip-len 48 --temporal-overlap 16 \
        --patch-size 256 --patch-overlap 64 --device cuda:0 || overall_status=1
    done
  fi
fi

for frames in "$BENCH"/outputs/*/frames; do
  [[ -d "$frames" ]] || continue
  model_dir=$(dirname "$frames")
  if ! find "$frames" -maxdepth 1 -type f | grep -q .; then
    continue
  fi
  if ! python3 "$CODE/scripts/check_output.py" \
      --input "$INPUT" --output "$frames" --report "$model_dir/check_report.json"; then
    overall_status=1
    continue
  fi
  python3 "$CODE/scripts/frames_to_video.py" \
    --frames "$frames" --output "$model_dir/output.mp4" --fps "$FPS" || overall_status=1
done

echo "Finished. Outputs: $BENCH/outputs"
exit "$overall_status"
