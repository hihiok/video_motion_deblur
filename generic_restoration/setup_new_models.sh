#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-all}"
if [[ "$MODEL" == "all" ]]; then
  status=0
  for name in flashvsr seedvr2 dove; do
    if ! "$0" "$name"; then
      echo "SETUP_FAILED: $name" >&2
      status=1
    fi
  done
  exit "$status"
fi
if [[ "$MODEL" != "flashvsr" && "$MODEL" != "seedvr2" && "$MODEL" != "dove" ]]; then
  echo "Usage: $0 all|flashvsr|seedvr2|dove" >&2
  exit 2
fi

# Proxy credentials are supplied by the private source conda environment.
# This repository deliberately contains no authenticated proxy URL.
ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur/generic_restoration}"
CODE="${CODE:-$ROOT/benchmark_code}"
CONDA_BASE="${CONDA_BASE:-/mnt/ssd1/z00919662/anaconda3}"
SOURCE_ENV="${BLACKWELL_SOURCE_ENV:-StereoPilot}"
FLASH_REPO="$ROOT/envs/FlashVSR"
SEED_REPO="$ROOT/envs/SeedVR"
DOVE_REPO="$ROOT/envs/DOVE"
BLOCK_SPARSE_REPO="$ROOT/envs/Block-Sparse-Attention"
FLASH_ATTN_REPO="$ROOT/envs/flash-attention"
APEX_REPO="$ROOT/envs/apex"

FLASH_COMMIT="6dd38e57203af4efca97df82c659f5d5a2dcf51a"
SEED_COMMIT="e4de8c24441a67e1b7df56abea10645059bb1185"
DOVE_COMMIT="0cd4240442cb5d122839c279977142cb6d648987"
BLOCK_SPARSE_COMMIT="49d6c39e4dc0303442cda3bb758b3925d4399c49"
FLASH_ATTN_COMMIT="0251105a2fb19d2957484b7f023cd8c115286ced"
APEX_COMMIT="9e3568a6f90fbc1996a06f8f9e99310bdaf2253a"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SOURCE_ENV"
if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_PREFIX/etc/conda/activate.d/proxy_env.sh"
fi
git config --global http.sslVerify false
git config --global http.version HTTP/1.1
conda config --set ssl_verify false
export GIT_SSL_NO_VERIFY=true PYTHONHTTPSVERIFY=0 SSL_NO_VERIFY=1 HF_HUB_DISABLE_XET=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org huggingface.co cdn-lfs.huggingface.co cas-bridge.xethub.hf.co github.com raw.githubusercontent.com objects.githubusercontent.com"
export PYTHONPATH="$CODE/generic_restoration/no_ssl:$CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/envs" "$ROOT/weights"

clone_pinned() {
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "$destination/.git" ]]; then
    GIT_SSL_NO_VERIFY=true git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$commit"
  fi
  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$commit" ]]; then
    echo "STOP: $destination is not at pinned commit $commit" >&2
    exit 3
  fi
}

ensure_env() {
  local env_name="$1"
  if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    conda create -y -n "$env_name" --clone "$SOURCE_ENV"
  fi
  conda activate "$env_name"
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable"
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
if torch.cuda.get_device_capability(0) != (12, 0):
    raise SystemExit("This setup is reserved for the RTX PRO 6000 Blackwell server (sm_120)")
x = torch.ones(1, device="cuda")
print("blackwell_tensor_test", x.item())
PY
}

ensure_nvcc_128() {
  if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version | grep -Eq 'release 12\.(8|9)|release 13\.'; then
    conda install -y -k -c "nvidia/label/cuda-12.8.0" cuda-toolkit
  fi
  export CUDA_HOME
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  nvcc --version
}

if [[ "$MODEL" == "flashvsr" ]]; then
  clone_pinned https://github.com/OpenImagingLab/FlashVSR.git "$FLASH_REPO" "$FLASH_COMMIT"
  clone_pinned https://github.com/mit-han-lab/Block-Sparse-Attention.git "$BLOCK_SPARSE_REPO" "$BLOCK_SPARSE_COMMIT"
  if [[ -n "$(git -C "$FLASH_REPO" status --porcelain)" ]]; then
    echo "STOP: FlashVSR official checkout is dirty" >&2
    exit 4
  fi
  ensure_env flashvsr_blackwell
  python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    packaging ninja wheel setuptools
  python -m pip install -e "$FLASH_REPO" --no-deps
  python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    torchmetrics==1.7.3 torchsde==0.2.6 accelerate==1.8.1 einops==0.8.1 \
    huggingface-hub==0.34.4 matplotlib==3.10.3 numpy==1.26.4 \
    opencv-python-headless==4.11.0.86 peft==0.16.0 pillow==11.0.0 \
    safetensors==0.5.3 sentencepiece==0.2.0 transformers==4.46.2 \
    pytorch-lightning==2.5.2 imageio==2.37.0 imageio-ffmpeg==0.6.0 \
    protobuf==3.20.3 ftfy==6.3.1 pandas==2.3.0 tqdm datasets
  if ! python -c 'import block_sparse_attn'; then
    ensure_nvcc_128
    git -C "$BLOCK_SPARSE_REPO" submodule update --init --recursive
    (
      cd "$BLOCK_SPARSE_REPO"
      export BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE
      export BLOCK_SPARSE_ATTN_CUDA_ARCHS=120
      export MAX_JOBS="${MAX_JOBS:-4}" NVCC_THREADS="${NVCC_THREADS:-2}"
      python -m pip install -v . --no-build-isolation
    )
  fi
  python -c 'import block_sparse_attn; print("block_sparse_attn", block_sparse_attn.__version__)'

  FLASH_WEIGHTS="$ROOT/weights/FlashVSR-v1.1"
  python - "$FLASH_WEIGHTS" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    repo_id="JunhaoZhuang/FlashVSR-v1.1",
    local_dir=sys.argv[1],
    allow_patterns=["*.ckpt", "*.pth", "*.safetensors", "*.json", "*.md"],
)
PY
  python - "$FLASH_WEIGHTS" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
required = ["LQ_proj_in.ckpt", "TCDecoder.ckpt", "Wan2.1_VAE.pth", "diffusion_pytorch_model_streaming_dmd.safetensors"]
for name in required:
    path = root / name
    if not path.is_file() or path.stat().st_size < 1_000_000:
        raise SystemExit(f"Missing/incomplete FlashVSR weight: {path}")
    head = path.read_bytes()[:128].lower()
    if b"<html" in head or b"git-lfs" in head:
        raise SystemExit(f"Invalid FlashVSR payload: {path}")
    print(name, path.stat().st_size)
PY
  echo "FLASHVSR_SETUP_PASS"
fi

if [[ "$MODEL" == "seedvr2" ]]; then
  clone_pinned https://github.com/ByteDance-Seed/SeedVR.git "$SEED_REPO" "$SEED_COMMIT"
  clone_pinned https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_REPO" "$FLASH_ATTN_COMMIT"
  clone_pinned https://github.com/NVIDIA/apex.git "$APEX_REPO" "$APEX_COMMIT"
  if [[ -n "$(git -C "$SEED_REPO" status --porcelain)" && ! -f "$SEED_REPO/.business_benchmark_patch.json" ]]; then
    echo "STOP: SeedVR checkout contains unknown modifications" >&2
    exit 4
  fi
  ensure_env seedvr2_blackwell
  python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    packaging ninja wheel setuptools einops==0.7.0 omegaconf==2.3.0 \
    opencv-python-headless==4.10.0.84 diffusers==0.33.1 \
    rotary-embedding-torch==0.5.3 transformers==4.46.2 mediapy==1.2.0 \
    accelerate safetensors imageio imageio-ffmpeg av
  ensure_nvcc_128
  SDPA_FALLBACK=0
  if ! python -c 'from flash_attn import flash_attn_varlen_func'; then
    set +e
    (
      cd "$FLASH_ATTN_REPO"
      export MAX_JOBS="${MAX_JOBS:-4}" FLASH_ATTENTION_FORCE_BUILD=TRUE
      python -m pip install -v . --no-build-isolation
    )
    FLASH_RC=$?
    set -e
    if [[ $FLASH_RC -ne 0 ]]; then
      SDPA_FALLBACK=1
    fi
  fi
  APEX_FALLBACK=0
  if ! python -c 'from apex.normalization import FusedLayerNorm, FusedRMSNorm'; then
    set +e
    (
      cd "$APEX_REPO"
      export APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS="${MAX_JOBS:-4}"
      python -m pip install -v . --no-build-isolation
    )
    APEX_RC=$?
    set -e
    if [[ $APEX_RC -ne 0 ]]; then
      APEX_FALLBACK=1
    fi
  fi
  if [[ ! -f "$SEED_REPO/.business_benchmark_patch.json" ]]; then
    PATCH_ARGS=(--repo "$SEED_REPO")
    [[ $SDPA_FALLBACK -eq 1 ]] && PATCH_ARGS+=(--sdpa-fallback)
    [[ $APEX_FALLBACK -eq 1 ]] && PATCH_ARGS+=(--apex-fallback)
    PYTHONPATH="$CODE" python -m generic_restoration.patch_seedvr2_blackwell "${PATCH_ARGS[@]}"
  fi

  SEED_WEIGHTS="$ROOT/weights/SeedVR2-3B"
  python - "$SEED_WEIGHTS" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    repo_id="ByteDance-Seed/SeedVR2-3B",
    local_dir=sys.argv[1],
    allow_patterns=["*.pth", "*.pt", "*.safetensors", "*.json", "*.md", "*.txt"],
)
PY
  for name in seedvr2_ema_3b.pth ema_vae.pth; do
    if [[ ! -s "$SEED_WEIGHTS/$name" ]]; then
      echo "STOP: missing SeedVR2 checkpoint $SEED_WEIGHTS/$name" >&2
      exit 5
    fi
  done
  if [[ -e "$SEED_REPO/ckpts" && ! -L "$SEED_REPO/ckpts" ]]; then
    echo "STOP: $SEED_REPO/ckpts exists and is not our symlink" >&2
    exit 6
  fi
  ln -sfn "$SEED_WEIGHTS" "$SEED_REPO/ckpts"
  echo "SEEDVR2_SETUP_PASS attention_fallback=$SDPA_FALLBACK apex_fallback=$APEX_FALLBACK"
fi

if [[ "$MODEL" == "dove" ]]; then
  clone_pinned https://github.com/zhengchen1999/DOVE.git "$DOVE_REPO" "$DOVE_COMMIT"
  if [[ -n "$(git -C "$DOVE_REPO" status --porcelain)" && ! -f "$DOVE_REPO/.business_benchmark_patch.json" ]]; then
    echo "STOP: DOVE checkout contains unknown modifications" >&2
    exit 4
  fi
  ensure_env dove_blackwell
  python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    accelerate transformers==4.46.2 numpy==1.26.0 sentencepiece imageio \
    imageio-ffmpeg pydantic peft opencv-python-headless decord av torchdiffeq \
    diffusers pyiqa safetensors tqdm pillow gdown
  if [[ ! -f "$DOVE_REPO/.business_benchmark_patch.json" ]]; then
    PYTHONPATH="$CODE" python -m generic_restoration.patch_dove_business --repo "$DOVE_REPO"
  fi

  DOVE_WEIGHT_ROOT="$ROOT/weights/DOVE_Final"
  DOVE_MODEL_LINK="$DOVE_REPO/pretrained_models/DOVE"
  if [[ ! -f "$DOVE_MODEL_LINK/model_index.json" ]]; then
    mkdir -p "$DOVE_WEIGHT_ROOT"
    ARCHIVE="$DOVE_WEIGHT_ROOT/dove_final_download"
    if [[ ! -s "$ARCHIVE" ]]; then
      set +e
      python -m gdown --fuzzy \
        'https://drive.google.com/file/d/1Nl3XoJndMtpu6KPFcskUTkI0qWBiSXF2/view?usp=drive_link' \
        -O "$ARCHIVE"
      GDOWN_RC=$?
      set -e
      if [[ $GDOWN_RC -ne 0 ]]; then
        echo "STOP: DOVE Final Google Drive download failed." >&2
        echo "Manual action: download the official Stage-2 Final archive to $ARCHIVE" >&2
        exit 7
      fi
    fi
    EXTRACTED="$DOVE_WEIGHT_ROOT/extracted"
    MODEL_DIR="$(PYTHONPATH="$CODE" python -m generic_restoration.unpack_model_archive \
      --archive "$ARCHIVE" --destination "$EXTRACTED" | tail -n 1)"
    if [[ -e "$DOVE_MODEL_LINK" && ! -L "$DOVE_MODEL_LINK" ]]; then
      echo "STOP: $DOVE_MODEL_LINK exists and is not our symlink" >&2
      exit 8
    fi
    ln -sfn "$MODEL_DIR" "$DOVE_MODEL_LINK"
  fi
  python - "$DOVE_MODEL_LINK" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
if not (root / "model_index.json").is_file():
    raise SystemExit(f"Invalid DOVE Diffusers checkpoint: {root}")
total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
if total < 1_000_000_000:
    raise SystemExit(f"DOVE checkpoint is unexpectedly small: {total}")
print("DOVE model bytes", total)
PY
  echo "DOVE_SETUP_PASS"
fi
