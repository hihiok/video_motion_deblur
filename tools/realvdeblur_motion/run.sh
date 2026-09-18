#!/usr/bin/env bash
# Run ONLY the official independent 00_motion.mp4 through unmodified RealVDeblur.
# Resolve the required environment variables according to the accompanying task MD.
set +x
set -Eeuo pipefail
umask 077
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

: "${MODEL_REPO:?Set the verified official RealVDeblur checkout path}"
: "${PYTHON_BIN:?Set an absolute executable Python path from a working inference environment}"
: "${WAN_MODEL_DIR:?Set directory containing diffusion_pytorch_model.safetensors and Wan2.1_VAE.pth}"
: "${DMD_CHECKPOINT:?Set the released realvdeblur_dmd.safetensors path}"
: "${GPU:?Set one idle physical NVIDIA GPU index}"
UPSTREAM_COMMIT=63a2eb0ed4a22cb76b796a65d5d3052616352573
EXPECTED_BLOB=9c020c496aa4b456474017b0f4108b0e927188c7
EXPECTED_SIZE=3254985
DTYPE="${DTYPE:-bfloat16}"
INSECURE="${INSECURE:-0}"
INPUT_ROOT="${INPUT_ROOT:-/data/pub/z00919662/dataset/RealVDeblur_official_motion}"
RUNS_ROOT="${RUNS_ROOT:-/data/pub/z00919662/motion_deblur/runs}"

fail() { echo "ERROR: $*" >&2; exit 1; }
for exe in git curl ffprobe ffmpeg nvidia-smi; do command -v "$exe" >/dev/null || fail "Missing executable: $exe"; done
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || fail 'PYTHON_BIN must be an absolute executable path'
[[ "$MODEL_REPO" == /* && "$WAN_MODEL_DIR" == /* && "$DMD_CHECKPOINT" == /* ]] || fail 'Model and weight paths must be absolute'
[[ "$GPU" =~ ^[0-9]+$ ]] || fail 'GPU must be ONE physical numeric GPU index'
[[ "$DTYPE" == bfloat16 || "$DTYPE" == float16 ]] || fail 'Unsupported DTYPE'
[[ "$INSECURE" == 0 || "$INSECURE" == 1 ]] || fail 'INSECURE must be 0 or 1'
[[ -f "$MODEL_REPO/inference.py" ]] || fail 'Missing official inference.py'
[[ "$(git -C "$MODEL_REPO" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] || fail 'Use the pinned official checkout; do not reset another worktree'
[[ -z "$(git -C "$MODEL_REPO" status --porcelain --untracked-files=no)" ]] || fail 'Tracked upstream files are modified; preserve and report the diff'
for file in "$WAN_MODEL_DIR/diffusion_pytorch_model.safetensors" "$WAN_MODEL_DIR/Wan2.1_VAE.pth" "$DMD_CHECKPOINT"; do
  [[ -s "$file" ]] || fail "Missing or empty checkpoint: $file"
done
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$INPUT_ROOT" "$RUNS_ROOT"
RUN_DIR="$(mktemp -d "$RUNS_ROOT/realvdeblur_official_motion_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
STAGE=preflight
trap 'rc=$?; printf "STATUS: FAILED\nSTAGE: %s\nEXIT_CODE: %s\nRUN_DIR: %s\n" "$STAGE" "$rc" "$RUN_DIR" > "$RUN_DIR/STATUS.txt"; exit "$rc"' ERR
printf 'RUN_DIR=%s\n' "$RUN_DIR"
printf 'STATUS: IN_PROGRESS\n' > "$RUN_DIR/STATUS.txt"
nvidia-smi -i "$GPU" --query-gpu=name,memory.total,memory.free --format=csv > "$RUN_DIR/gpu_before.csv"
"$PYTHON_BIN" -c 'import sys, torch; print(sys.version); print("torch",torch.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' > "$RUN_DIR/environment.txt"
(cd "$MODEL_REPO"; "$PYTHON_BIN" inference.py --help) > "$RUN_DIR/inference_help.txt"

STAGE=download
INPUT="$INPUT_ROOT/00_motion.mp4"
URL="https://raw.githubusercontent.com/OpenImagingLab/RealVDeblur/$UPSTREAM_COMMIT/example/00_motion.mp4"
TLS=()
if [[ "$INSECURE" == 1 ]]; then
  TLS=(-k)
  echo 'WARNING: certificate verification disabled for this curl process only.' >&2
elif [[ -n "${CA_BUNDLE:-}" ]]; then
  [[ -s "$CA_BUNDLE" ]] || fail 'CA_BUNDLE does not exist'
  TLS=(--cacert "$CA_BUNDLE")
fi
check_input() {
  [[ "$(stat -c %s "$1")" == "$EXPECTED_SIZE" ]] && [[ "$(git hash-object --no-filters "$1")" == "$EXPECTED_BLOB" ]]
}
if [[ -e "$INPUT" ]]; then
  check_input "$INPUT" || fail 'Existing input differs from pinned official sample; preserve it, do not overwrite'
else
  PART="$INPUT.part"
  if [[ -f "$PART" ]] && check_input "$PART"; then
    mv -n "$PART" "$INPUT"
  else
    curl -fL "${TLS[@]}" --retry 3 --connect-timeout 30 -C - "$URL" -o "$PART"
    check_input "$PART" || fail 'Official sample size/blob verification failed; partial file preserved'
    mv -n "$PART" "$INPUT"
  fi
fi
check_input "$INPUT"
printf '%s\n' "$URL" > "$RUN_DIR/input_source.txt"
sha256sum "$INPUT" > "$RUN_DIR/input_sha256.txt"
sha256sum "$WAN_MODEL_DIR/diffusion_pytorch_model.safetensors" "$WAN_MODEL_DIR/Wan2.1_VAE.pth" "$DMD_CHECKPOINT" > "$RUN_DIR/weights_sha256.txt"

STAGE=decode
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames:format=duration,size \
  -of json "$INPUT" > "$RUN_DIR/input_probe.json"
read -r W H FPS N < <("$PYTHON_BIN" -c '
import json,sys
from fractions import Fraction
s=json.load(open(sys.argv[1]))["streams"][0]
w,h,n=int(s["width"]),int(s["height"]),int(s["nb_read_frames"])
f=Fraction(s["avg_frame_rate"])
assert min(w,h,n)>0 and f>0
print(w,h,str(f),n)
' "$RUN_DIR/input_probe.json")
# Check actual timestamps rather than silently assigning a false rate to VFR media.
ffprobe -v error -select_streams v:0 -show_frames \
  -show_entries frame=best_effort_timestamp_time -of json "$INPUT" > "$RUN_DIR/input_timestamps.json"
"$PYTHON_BIN" - "$RUN_DIR/input_timestamps.json" "$FPS" "$N" <<'PY'
import json,sys
from fractions import Fraction
frames=json.load(open(sys.argv[1]))['frames']
t=[float(f['best_effort_timestamp_time']) for f in frames]
assert len(t)==int(sys.argv[3]), 'Frame timestamp count mismatch'
step=1/float(Fraction(sys.argv[2]))
assert all(abs((b-a)-step)<=max(0.0002, step*0.002) for a,b in zip(t,t[1:])), 'VFR input: preserve it and report; do not silently resample'
PY
mkdir "$RUN_DIR/input_frames" "$RUN_DIR/frames_model"
ffmpeg -nostdin -v error -xerror -n -threads 2 -filter_threads 1 -i "$INPUT" \
  -map 0:v:0 -vsync 0 -start_number 0 -threads 2 "$RUN_DIR/input_frames/%08d.png"
[[ "$(find "$RUN_DIR/input_frames" -maxdepth 1 -name '*.png' -type f | wc -l)" -eq "$N" ]]
MODEL_INPUT="$RUN_DIR/input_frames"
PAD_W=$(( (W+15)/16*16 )); PAD_H=$(( (H+15)/16*16 ))
if [[ "$PAD_W" -ne "$W" || "$PAD_H" -ne "$H" ]]; then
  MODEL_INPUT="$RUN_DIR/input_padded"
  mkdir "$MODEL_INPUT"
  ffmpeg -nostdin -v error -xerror -n -threads 2 -filter_threads 1 \
    -framerate "$FPS" -start_number 0 -i "$RUN_DIR/input_frames/%08d.png" \
    -vf "pad=$PAD_W:$PAD_H:0:0:color=black" -vsync 0 -start_number 0 -threads 2 "$MODEL_INPUT/%08d.png"
fi

STAGE=inference
START=$SECONDS
(cd "$MODEL_REPO"; "$PYTHON_BIN" inference.py \
  --input "$MODEL_INPUT" --output "$RUN_DIR/frames_model" \
  --wan_model_dir "$WAN_MODEL_DIR" --checkpoint "$DMD_CHECKPOINT" \
  --device cuda:0 --dtype "$DTYPE" --seed 0 --stride 1 \
  --num_inference_steps 1 --enable_twm --temporal_window_size 21) \
  2>&1 | tee "$RUN_DIR/inference.log"
ELAPSED=$((SECONDS-START))

STAGE=verify_and_encode
find "$RUN_DIR/input_frames" -maxdepth 1 -name '*.png' -type f -printf '%f\n' | sort > "$RUN_DIR/input_names.txt"
find "$RUN_DIR/frames_model" -maxdepth 1 -name '*.png' -type f -printf '%f\n' | sort > "$RUN_DIR/output_names.txt"
diff -u "$RUN_DIR/input_names.txt" "$RUN_DIR/output_names.txt" > "$RUN_DIR/frame_name_diff.txt"
mkdir "$RUN_DIR/frames" "$RUN_DIR/previews"
# Remove ONLY the border that this runner added; never crop original image content.
ffmpeg -nostdin -v error -xerror -n -threads 2 -filter_threads 1 \
  -framerate "$FPS" -start_number 0 -i "$RUN_DIR/frames_model/%08d.png" \
  -vf "crop=$W:$H:0:0" -vsync 0 -start_number 0 -threads 2 "$RUN_DIR/frames/%08d.png"
[[ "$(find "$RUN_DIR/frames" -maxdepth 1 -name '*.png' -type f | wc -l)" -eq "$N" ]]
PIX_FMT=yuv420p
if (( W%2 || H%2 )); then PIX_FMT=yuv444p; fi
ffmpeg -nostdin -v error -xerror -n -threads 2 -filter_threads 1 \
  -framerate "$FPS" -start_number 0 -i "$RUN_DIR/frames/%08d.png" \
  -c:v libx264 -preset medium -crf 16 -pix_fmt "$PIX_FMT" -threads 2 \
  -movflags +faststart "$RUN_DIR/output_00_motion.mp4"
ffmpeg -nostdin -v error -xerror -n -threads 2 -filter_complex_threads 1 \
  -framerate "$FPS" -start_number 0 -i "$RUN_DIR/input_frames/%08d.png" \
  -framerate "$FPS" -start_number 0 -i "$RUN_DIR/frames/%08d.png" \
  -filter_complex '[0:v][1:v]hstack=inputs=2[v]' -map '[v]' \
  -c:v libx264 -preset medium -crf 16 -pix_fmt "$PIX_FMT" -threads 2 \
  -movflags +faststart "$RUN_DIR/input_vs_realvdeblur.mp4"
for INDEX in 0 $(( N/2 )) $(( N-1 )); do
  NAME="$(printf '%08d' "$INDEX")"
  ffmpeg -nostdin -v error -n -threads 2 -filter_complex_threads 1 \
    -i "$RUN_DIR/input_frames/$NAME.png" -i "$RUN_DIR/frames/$NAME.png" \
    -filter_complex '[0:v][1:v]hstack=inputs=2[v]' -map '[v]' -frames:v 1 -threads 2 \
    "$RUN_DIR/previews/input_output_$NAME.png"
done
for VIDEO in output_00_motion.mp4 input_vs_realvdeblur.mp4; do
  COUNT="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$RUN_DIR/$VIDEO")"
  [[ "$COUNT" -eq "$N" ]]
  ffmpeg -nostdin -v error -xerror -threads 2 -i "$RUN_DIR/$VIDEO" -f null -
done
[[ -z "$(git -C "$MODEL_REPO" status --porcelain --untracked-files=no)" ]]
cat > "$RUN_DIR/STATUS.txt" <<EOF
STATUS: INFERENCE_AND_ENCODING_COMPLETE
VISUAL_REVIEW: REQUIRED_BY_CODEAGENT
MODEL: RealVDeblur official one-step DMD
UPSTREAM_COMMIT: $UPSTREAM_COMMIT
INPUT: $INPUT
INPUT_GIT_BLOB: $EXPECTED_BLOB
FRAME_COUNT: $N
ORIGINAL_RESOLUTION: ${W}x${H}
MODEL_RESOLUTION: ${PAD_W}x${PAD_H}
RESIZE: NO
PADDING: RIGHT_BOTTOM_ONLY_REMOVED_AFTER_INFERENCE
FPS: $FPS
AUDIO: NOT_INCLUDED_IN_REVIEW_VIDEOS
GPU: $GPU
DTYPE: $DTYPE
SEED: 0
TWM: 21
STEPS: 1
INFERENCE_WALL_SECONDS_INCLUDING_MODEL_LOAD: $ELAPSED
MODEL_CODE_MODIFIED: NO
GT: NOT_PROVIDED
PSNR_SSIM: NOT_COMPUTED
RUN_DIR: $RUN_DIR
EOF
cat "$RUN_DIR/STATUS.txt"
