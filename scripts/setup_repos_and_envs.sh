#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
ENVS="$ROOT/envs"
mkdir -p "$ENVS"

if [[ ! -d "$ENVS/realvdeblur_repo/.git" ]]; then
  git clone https://github.com/OpenImagingLab/RealVDeblur.git "$ENVS/realvdeblur_repo"
fi

for repo in bsstnet_repo dstnet_repo shiftnet_repo; do
  [[ -d "$ENVS/$repo" ]] || { echo "Missing existing repo: $ENVS/$repo" >&2; exit 1; }
done

# RealVDeblur official environment
if ! conda env list | awk '{print $1}' | grep -qx realvdeblur; then
  conda create -n realvdeblur python=3.10 -y
fi
conda run -n realvdeblur python -m pip install -r "$ENVS/realvdeblur_repo/requirements.txt"
conda run -n realvdeblur python -m pip install -U 'huggingface_hub[cli]'

# DSTNet environment
if ! conda env list | awk '{print $1}' | grep -qx dstnet; then
  conda create -n dstnet python=3.8 -y
  conda install -n dstnet -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
fi
conda run -n dstnet python -m pip install -r "$ENVS/dstnet_repo/requirements.txt"
(cd "$ENVS/dstnet_repo" && conda run -n dstnet python setup.py develop)

# Shift-Net environment. Keep CUDA extensions disabled as recommended by the official repo.
if ! conda env list | awk '{print $1}' | grep -qx shiftnet; then
  conda create -n shiftnet python=3.8 -y
  conda run -n shiftnet python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
fi
conda run -n shiftnet python -m pip install -r "$ENVS/shiftnet_repo/requirements.txt"
(cd "$ENVS/shiftnet_repo" && conda run -n shiftnet python setup.py develop --no_cuda_ext)

# BSSTNet environment. This is intentionally isolated because of mmcv-full.
if ! conda env list | awk '{print $1}' | grep -qx bsstnet; then
  conda create -n bsstnet python=3.8 -y
  conda run -n bsstnet python -m pip install \
    torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
fi
conda run -n bsstnet python -m pip install mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
conda run -n bsstnet python -m pip install -r "$ENVS/bsstnet_repo/requirements.txt"
(cd "$ENVS/bsstnet_repo" && BASICSR_EXT=True conda run -n bsstnet python setup.py develop)

echo "Repository and environment setup complete."
