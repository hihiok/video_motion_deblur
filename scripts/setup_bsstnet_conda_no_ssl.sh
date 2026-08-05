#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
ENV_NAME="${BSST_ENV:-bsstnet}"
REPO="${BSST_REPO:-$ROOT/envs/bsstnet_repo}"
GPU="${GPU:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available" >&2
  exit 1
fi
if [[ ! -d "$REPO" ]]; then
  echo "BSSTNet repository not found: $REPO" >&2
  exit 1
fi

# The server uses a trusted internal TLS-inspection proxy.  Disable certificate
# verification for this installation workflow.  Never print proxy variables.
conda config --set ssl_verify false
git config --global http.sslVerify false
export CONDA_SSL_VERIFY=false
export GIT_SSL_NO_VERIFY=true
export PYTHONHTTPSVERIFY=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

TRUSTED_HOSTS=(
  --trusted-host pypi.org
  --trusted-host files.pythonhosted.org
  --trusted-host download.pytorch.org
  --trusted-host download.openmmlab.com
)

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

if ! conda_env_exists "$ENV_NAME"; then
  echo "Creating official BSSTNet Python 3.8 environment: $ENV_NAME"
  CONDA_SSL_VERIFY=false conda create -n "$ENV_NAME" python=3.8 pip -y
else
  echo "Using existing conda environment: $ENV_NAME"
fi

PIP=(conda run -n "$ENV_NAME" python -m pip)
PY=(conda run -n "$ENV_NAME" python)

# Keep packaging tools old enough for the official Torch 1.9 / Python 3.8 stack.
"${PIP[@]}" install "pip<24.1" "setuptools<70" wheel "${TRUSTED_HOSTS[@]}"

# Pin numerical packages to mutually compatible Python 3.8 wheels.
"${PIP[@]}" install \
  numpy==1.23.5 scipy==1.10.1 scikit-image==0.21.0 \
  "${TRUSTED_HOSTS[@]}"

# Official BSSTNet versions.
"${PIP[@]}" install \
  torch==1.9.1+cu111 \
  torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html \
  "${TRUSTED_HOSTS[@]}"

"${PIP[@]}" install \
  mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html \
  "${TRUSTED_HOSTS[@]}"

# Inference-only dependencies.  Avoid tb-nightly/wandb because they are not
# needed by the benchmark adapter and can disturb this legacy environment.
"${PIP[@]}" install \
  addict future lmdb opencv-python Pillow pyyaml requests tqdm yapf einops ninja \
  "${TRUSTED_HOSTS[@]}"

# Install the repository in develop mode exactly as requested upstream.
cd "$REPO"
BASICSR_EXT=True "${PY[@]}" setup.py develop

# Verify version alignment and the CUDA deformable-convolution operator.
CUDA_VISIBLE_DEVICES="$GPU" "${PY[@]}" - <<'PY'
import torch
import torchvision
import mmcv
from torchvision.ops import deform_conv2d

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("mmcv:", mmcv.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the BSSTNet environment")

x = torch.randn(1, 1, 8, 8, device="cuda")
offset = torch.zeros(1, 18, 8, 8, device="cuda")
weight = torch.randn(1, 1, 3, 3, device="cuda")
y = deform_conv2d(x, offset, weight, padding=(1, 1))
print("torchvision deform_conv2d:", tuple(y.shape))
PY

echo "BSSTNet environment setup passed: $ENV_NAME"
