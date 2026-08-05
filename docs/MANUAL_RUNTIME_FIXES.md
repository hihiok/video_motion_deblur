# Manual runtime fixes after smoke-test failures

This document addresses the four failures reported on the target server without modifying the official model architectures.

Workspace:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur
CODE=$ROOT/benchmark_code
```

## 1. Pull the adapter fixes first

```bash
cd "$CODE"
git pull --ff-only
git log -5 --oneline
```

The checkout must contain both fixes:

```bash
grep -n 'tensor.contiguous' adapters/shiftnet_infer.py
grep -n 'cupy.ndarray' adapters/dstnet_compat.py
```

### Why these fixes are safe

- Shift-Net's official `channel_shift()` uses `Tensor.view()`. The adapter now makes the input contiguous before entering the unchanged official network.
- DSTNet still uses the unchanged official network. When real CuPy is unavailable, its import-only fake CuPy module now exposes a dummy `ndarray` type so einops correctly selects the PyTorch backend.

## 2. Run Shift-Net+ only

```bash
cd "$CODE"
ROOT="$ROOT" \
CODE="$CODE" \
COMMON_ENV=deblur_runtime \
SHIFT_ENV=deblur_runtime \
GPU=0 \
bash run_all.sh --model=shiftnet
```

The old NumPy/SciPy error should not occur because the adapter directly imports only the official architecture file. The previous `view()` error should not occur because the model input is now contiguous.

Expected outputs:

```text
benchmark/outputs/shiftnet_gopro_plus/
benchmark/outputs/shiftnet_dvd_plus/
```

## 3. Run DSTNet only

```bash
cd "$CODE"
ROOT="$ROOT" \
CODE="$CODE" \
COMMON_ENV=deblur_runtime \
DST_ENV=deblur_runtime \
GPU=0 \
bash run_all.sh --model=dstnet
```

Expected log backend:

```text
dynamic_backend=official_cupy
```

or:

```text
dynamic_backend=pytorch_unfold
```

Both are valid. The latter is slower but does not change checkpoint parameters or the network topology.

Expected outputs:

```text
benchmark/outputs/dstnet_gopro/
benchmark/outputs/dstnet_dvd/
benchmark/outputs/dstnet_bsd/
```

## 4. Create the official BSSTNet environment

BSSTNet must not use `deblur_runtime`. Its architecture directly calls `torchvision.ops.deform_conv2d`, so Torch and torchvision compiled operators must match. The official repository specifies Python 3.8, Torch 1.9.1+cu111, torchvision 0.10.1+cu111, torchaudio 0.9.1, and mmcv-full 1.7.1.

### Route A: normal network access

```bash
conda create -n bsstnet python=3.8 -y

conda run -n bsstnet python -m pip install \
  torch==1.9.1+cu111 \
  torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

conda run -n bsstnet python -m pip install \
  mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html

conda run -n bsstnet python -m pip install \
  -r "$ROOT/envs/bsstnet_repo/requirements.txt"

cd "$ROOT/envs/bsstnet_repo"
BASICSR_EXT=True conda run -n bsstnet python setup.py develop
```

### Route B: conda channel SSL is blocked but an existing Python 3.8 env exists

List local environments and find one using Python 3.8:

```bash
for env in $(conda env list | awk 'NF && $1 !~ /^#/ {print $1}'); do
  conda run -n "$env" python -c 'import sys; print(sys.version_info[:2])' 2>/dev/null \
    | grep -q '(3, 8)' && echo "Python 3.8 env: $env"
done
```

Clone that environment locally, then replace its Torch stack:

```bash
conda create -n bsstnet --clone <EXISTING_PY38_ENV> -y
conda run -n bsstnet python -m pip uninstall -y torch torchvision torchaudio mmcv mmcv-full
```

Then run the pip installation commands from Route A. Cloning is local and does not contact conda channels.

Do not clone `deblur_runtime` when it uses Python 3.9 or a different Torch ABI.

### Verify the compiled torchvision operator

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n bsstnet python - <<'PY'
import torch
import torchvision
from torchvision.ops import deform_conv2d

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda:', torch.version.cuda, torch.cuda.is_available())

x = torch.randn(1, 1, 8, 8, device='cuda')
offset = torch.zeros(1, 18, 8, 8, device='cuda')
weight = torch.randn(1, 1, 3, 3, device='cuda')
y = deform_conv2d(x, offset, weight, padding=(1, 1))
print('deform_conv2d:', tuple(y.shape))
PY
```

Expected versions:

```text
torch: 1.9.1+cu111
torchvision: 0.10.1+cu111
deform_conv2d: (1, 1, 8, 8)
```

Then run:

```bash
cd "$CODE"
ROOT="$ROOT" CODE="$CODE" BSST_ENV=bsstnet GPU=0 \
  bash run_all.sh --model=bsstnet
```

## 5. RealVDeblur: distinguish DMD checkpoint from Wan base model

RealVDeblur requires all three files:

```text
benchmark/weights/realvdeblur/realvdeblur_dmd.safetensors
benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
```

The 179 MB DMD file is the RealVDeblur checkpoint. The approximately 3.6 GB diffusion file is a separate Wan2.1 base-model file.

First search for an already downloaded copy:

```bash
find /mnt/ssd1/z00919662 -type f \
  -name diffusion_pytorch_model.safetensors -size +3000M -print
```

If found elsewhere, symlink it into the expected directory:

```bash
mkdir -p "$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B"
ln -sfn <FOUND_FILE> \
  "$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
```

If no copy exists, download only that file with resume support:

```bash
mkdir -p "$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B"

curl -L --fail --retry 5 --retry-delay 5 -C - \
  'https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors?download=true' \
  -o "$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
```

Basic file validation:

```bash
ls -lh "$ROOT/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/"*.safetensors

conda run -n deblur_runtime python - <<'PY'
from safetensors import safe_open
p = '/mnt/ssd1/z00919662/motion_deblur/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors'
with safe_open(p, framework='pt', device='cpu') as f:
    keys = list(f.keys())
print('valid safetensors; tensors:', len(keys), 'first:', keys[:3])
PY
```

## 6. Run the complete smoke test

Once BSSTNet's environment exists and the three RealVDeblur files are present:

```bash
cd "$CODE"
ROOT="$ROOT" \
SOURCE_ENV=turtle_joint_py222 \
RUNTIME_ENV=deblur_runtime \
REAL_ENV=deblur_runtime \
SHIFT_ENV=deblur_runtime \
DST_ENV=deblur_runtime \
BSST_ENV=bsstnet \
SKIP_REALVDEBLUR=0 \
GPU=0 \
bash scripts/recover_after_codeagent.sh
```

Each model is executed independently; one failure does not suppress later models.
