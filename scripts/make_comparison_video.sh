#!/usr/bin/env bash
set -euo pipefail
INPUT="$1"
OUTPUT="$2"
shift 2
# Remaining arguments: label=video_path
FILTERS=""
INPUT_ARGS=(-i "$INPUT")
LABELS=("Input")
INDEX=1
for item in "$@"; do
  label="${item%%=*}"
  path="${item#*=}"
  INPUT_ARGS+=(-i "$path")
  LABELS+=("$label")
  INDEX=$((INDEX+1))
done
N=${#LABELS[@]}
for ((i=0;i<N;i++)); do
  FILTERS+="[$i:v]setpts=PTS-STARTPTS,drawtext=text='${LABELS[$i]}':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[v$i];"
done
STACK=""
for ((i=0;i<N;i++)); do STACK+="[v$i]"; done
FILTERS+="${STACK}hstack=inputs=${N}[outv]"
mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -y "${INPUT_ARGS[@]}" -filter_complex "$FILTERS" -map '[outv]' -c:v libx264 -crf 12 -preset slow -pix_fmt yuv420p "$OUTPUT"
