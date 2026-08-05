#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
ENVS="$ROOT/envs"
SOURCE_ENV="${SOURCE_ENV:-turtle_joint_py222}"
RUNTIME_ENV="${RUNTIME_ENV:-deblur_runtime}"
FRESH_ENVS="${FRESH_ENVS:-0}"
mkdir -p "$ENVS"

if [[ ! -d "$ENVS/realvdeblur_repo/.git" ]]; then
  git clone https://github.com/OpenImagingLab/RealVDeblur.git "$ENVS/realvdeblur_repo"
fi

for repo in bsstnet_repo dstnet_repo shiftnet_repo; do
  [[ -d "$ENVS/$repo" ]] || { echo "Missing existing repo: $ENVS/$repo" >&2; exit 1; }
done

conda_env_exists() {
  command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$1"
}

# Default path for this server: clone the already-working Torch 2.4/CUDA 11.8
# environment locally.  This operation does not need PyPI or conda channels.
if [[ "$FRESH_ENVS" != "1" ]]; then
  if conda_env_exists "$RUNTIME_ENV"; then
    echo "Using existing runtime environment: $RUNTIME_ENV"
  elif conda_env_exists "$SOURCE_ENV"; then
    echo "Cloning $SOURCE_ENV -> $RUNTIME_ENV without network downloads..."
    conda create -n "$RUNTIME_ENV" --clone "$SOURCE_ENV" -y
  else
    echo "Existing source environment '$SOURCE_ENV' not found." >&2
    echo "Set SOURCE_ENV to a local Torch environment, or use FRESH_ENVS=1." >&2
    exit 1
  fi

  conda run -n "$RUNTIME_ENV" python -c \
    'import sys, torch; print(sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
  echo "Local runtime prepared. Run scripts/recover_after_codeagent.sh next."
  exit 0
fi

# Optional clean-environment path.  It requires external package channels and
# is intentionally not the default on proxy-restricted servers.
echo "FRESH_ENVS=1: creating the original isolated environments."

if ! conda_env_exists realvdeblur; then
  conda create -n realvdeblur python=3.10 -y
fi
conda run -n realvdeblur python -m pip install -r "$ENVS/realvdeblur_repo/requirements.txt"
conda run -n realvdeblur python -m pip install -U 'huggingface_hub[cli]'

if ! conda_env_exists dstnet; then
  conda create -n dstnet python=3.8 -y
  conda install -n dstnet -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
fi
# Do not install DSTNet's full requirements by default: the inference adapter
# supplies mmcv/CuPy compatibility and imports only the released architecture.
conda run -n dstnet python -m pip install numpy opencv-python Pillow einops tqdm pyyaml

if ! conda_env_exists shiftnet; then
  conda create -n shiftnet python=3.8 -y
  conda run -n shiftnet python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
fi
conda run -n shiftnet python -m pip install -r "$ENVS/shiftnet_repo/requirements.txt"
(cd "$ENVS/shiftnet_repo" && conda run -n shiftnet python setup.py develop --no_cuda_ext)

if ! conda_env_exists bsstnet; then
  conda create -n bsstnet python=3.8 -y
  conda run -n bsstnet python -m pip install \
    torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
fi
conda run -n bsstnet python -m pip install mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
conda run -n bsstnet python -m pip install -r "$ENVS/bsstnet_repo/requirements.txt"
(cd "$ENVS/bsstnet_repo" && BASICSR_EXT=True conda run -n bsstnet python setup.py develop)

echo "Fresh isolated environments prepared."
