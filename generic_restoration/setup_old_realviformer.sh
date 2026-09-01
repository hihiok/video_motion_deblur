#!/usr/bin/env bash
set -euo pipefail

# Proxy credentials are intentionally not stored here. Activating the server's
# source environment loads its private proxy_env.sh without printing secrets.
ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur/generic_restoration}"
CODE="${CODE:-$ROOT/benchmark_code}"
CONDA_BASE="${CONDA_BASE:-/mnt/ssd1/z00919662/anaconda3}"
SOURCE_ENV="${SOURCE_ENV:-RVRT}"
ENV_NAME="${ENV_NAME:-realviformer_rwvsr}"
REPO="${REALVIFORMER_REPO:-$ROOT/envs/RealViformer}"
WEIGHT_ROOT="${WEIGHT_ROOT:-$ROOT/weights/realviformer}"
CHECKPOINT="${REALVIFORMER_CHECKPOINT:-$WEIGHT_ROOT/weights.pth}"
OFFICIAL_COMMIT="bd5f88d05ba62136727a61cb162da53f22560465"

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda initialization: $CONDA_BASE/etc/profile.d/conda.sh" >&2
  exit 2
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# The corporate proxy is supplied by the activated environment. Never echo it.
conda activate "$SOURCE_ENV"
if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi
git config --global http.sslVerify false
git config --global http.version HTTP/1.1
conda config --set ssl_verify false
export GIT_SSL_NO_VERIFY=true PYTHONHTTPSVERIFY=0 HF_HUB_DISABLE_XET=1
export PYTHONPATH="$CODE/generic_restoration/no_ssl:$CODE${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$ROOT/envs" "$WEIGHT_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  GIT_SSL_NO_VERIFY=true git clone https://github.com/Yuehan717/RealViformer.git "$REPO"
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "STOP: official RealViformer checkout is dirty: $REPO" >&2
  exit 3
fi
git -C "$REPO" fetch --all --tags
git -C "$REPO" checkout --detach "$OFFICIAL_COMMIT"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" --clone "$SOURCE_ENV"
fi
conda activate "$ENV_NAME"
python -m pip install --disable-pip-version-check \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  opencv-python einops pillow numpy gdown

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable"
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY

if [[ ! -s "$CHECKPOINT" ]]; then
  DOWNLOAD_DIR="$WEIGHT_ROOT/gdrive_download"
  mkdir -p "$DOWNLOAD_DIR"
  set +e
  python -m gdown --folder \
    'https://drive.google.com/drive/folders/1UzDfFSy5oELl7Z-umF_QhMQhUbUU378y?usp=sharing' \
    --remaining-ok -O "$DOWNLOAD_DIR"
  GDOWN_RC=$?
  set -e
  if [[ $GDOWN_RC -eq 0 ]]; then
    FOUND="$(find "$DOWNLOAD_DIR" -type f -name 'weights.pth' -print -quit)"
    if [[ -z "$FOUND" ]]; then
      FOUND="$(find "$DOWNLOAD_DIR" -type f -name '*.pth' -print -quit)"
    fi
    if [[ -n "$FOUND" ]]; then
      ln -sfn "$FOUND" "$CHECKPOINT"
    fi
  fi
fi

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "STOP: RealViformer checkpoint was not downloaded." >&2
  echo "Manual action: download the official Google Drive folder and place weights.pth at:" >&2
  echo "$CHECKPOINT" >&2
  exit 4
fi

python - "$CHECKPOINT" <<'PY'
import hashlib
import pathlib
import sys
import torch

path = pathlib.Path(sys.argv[1])
head = path.read_bytes()[:128]
if b"<html" in head.lower() or b"git-lfs" in head.lower():
    raise SystemExit(f"Invalid checkpoint payload: {path}")
state = torch.load(path, map_location="cpu")
if not isinstance(state, dict) or "params" not in state:
    raise SystemExit("Expected a RealViformer checkpoint containing the 'params' state dict")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print("checkpoint", path, "bytes", path.stat().st_size, "sha256", digest)
PY

echo "REALVIFORMER_SETUP_PASS"
