#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
cd "$CODE"
ROOT=${ROOT:-/data/pub/z00919662/motion_deblur}
DATA=${DATA:-/data/pub/z00919662/dataset}
GOPRO_ROOT=${GOPRO_ROOT:-$DATA/GoPro}
DVD_ROOT=${DVD_ROOT:-$DATA/DVD}
: "${BSD_ROOT:?Set BSD_ROOT to the dedicated BSD 3ms-24ms data root}"
UPSTREAM=${UPSTREAM:-$ROOT/envs/dstnetplus_500g_upstream}
WEIGHT=${WEIGHT:-$ROOT/weights/DSTNetPlus_base_gopro.pth}
RUN=${RUN:-$ROOT/runs/dstplus_waveshift_500g_v1}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ -n "${GPUS:-}" ]]; then
  SELECTED=$(python -m dst500.gpus --gpus "$GPUS")
else
  SELECTED=$(python -m dst500.gpus)
fi
export CUDA_VISIBLE_DEVICES="$SELECTED"
echo "Using exactly two physical GPUs: $SELECTED"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'SOURCE_DIRTY: preserve diff and send it to the author; do not overwrite.' >&2
  exit 1
fi
mkdir -p "$RUN" "$(dirname "$UPSTREAM")" "$(dirname "$WEIGHT")"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
if [[ ! -d "$UPSTREAM/.git" ]]; then
  git clone https://github.com/sunny2109/DSTNet-plus.git "$UPSTREAM"
fi
if [[ -n "$(git -C "$UPSTREAM" status --porcelain --untracked-files=no)" ]]; then
  echo 'UPSTREAM_DIRTY: preserve changes and report.' >&2
  exit 1
fi
git -C "$UPSTREAM" fetch origin 54363c15d8b924aa1ae56b8f835c1f0289954e95
git -C "$UPSTREAM" checkout --detach 54363c15d8b924aa1ae56b8f835c1f0289954e95
if [[ ! -s "$WEIGHT" ]]; then
  curl -k -f -L --retry 3 --connect-timeout 20 \
    -o "$WEIGHT.part" https://github.com/sunny2109/DSTNet-plus/releases/download/v0.1.0/DSTNetPlus_base_gopro.pth
  mv "$WEIGHT.part" "$WEIGHT"
fi
python - "$UPSTREAM" "$WEIGHT" <<'PY'
import sys
from dst500.model import teacher
m=teacher(sys.argv[1],sys.argv[2])
print('Official Base teacher strict load passed:',sum(p.numel() for p in m.parameters()))
PY
if [[ ! -f "$RUN/manifest.json" ]]; then
  python -m dst500.data --gopro-root "$GOPRO_ROOT" --dvd-root "$DVD_ROOT" --bsd-root "$BSD_ROOT" --output "$RUN/manifest.json"
fi
python -m dst500.run configure --config "$RUN/config.json" --upstream "$UPSTREAM" \
  --teacher-checkpoint "$WEIGHT" --manifest "$RUN/manifest.json" --output "$RUN"
python -m dst500.profile --upstream "$UPSTREAM" --variant teacher --output "$RUN/profile_teacher.json"
if [[ ! -f "$RUN/teacher_test.json" ]]; then
  python -m dst500.evaluate --config "$RUN/config.json" --teacher --output "$RUN/teacher_test.json"
fi
python - "$RUN/teacher_test.json" "$WEIGHT" "$RUN/manifest.json" <<'PY'
import json,sys
from dst500.data import sha256
r=json.load(open(sys.argv[1]));g=r['summary']['gopro']
assert r['checkpoint_sha256']==sha256(sys.argv[2]), 'Stale teacher baseline'
assert r['manifest_sha256']==sha256(sys.argv[3]), 'Stale baseline manifest'
assert g['frames']==1111 and g['sequences']==11, 'Incomplete GoPro baseline'
assert abs(g['input_psnr_rgb8']-25.6401)<.05, 'GoPro input/pairing mismatch; inspect blur vs blur_gamma'
assert g['psnr_rgb8']>=33, 'TEACHER_BELOW_TARGET: inspect pipeline/weights before costly training'
PY
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m dst500.run train --config "$RUN/config.json" --preflight
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m dst500.run train --config "$RUN/config.json"
python -m dst500.evaluate --config "$RUN/config.json" --checkpoint "$RUN/best_gopro_val.pth" --output "$RUN/final_test.json"
python - "$RUN/final_test.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print(r['status']);print(r['summary']['gopro'])
if r['status']!='TARGET_MET': raise SystemExit('TARGET_NOT_MET: return artifacts for author analysis; no autonomous code edits')
PY
