#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/ssd1/z00919662/motion_deblur}"
BENCH="$ROOT/benchmark"
mkdir -p "$BENCH/weights" "$BENCH/manifests" "$BENCH/tmp"

command -v hf >/dev/null || python -m pip install -U 'huggingface_hub[cli]'
command -v gdown >/dev/null || python -m pip install -U gdown

# RealVDeblur + Wan2.1
mkdir -p "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B"
hf download Wan-AI/Wan2.1-T2V-1.3B \
  diffusion_pytorch_model.safetensors Wan2.1_VAE.pth \
  --local-dir "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B"
hf download RBJin/RealVDeblur realvdeblur_dmd.safetensors \
  --local-dir "$BENCH/weights/realvdeblur"

# DSTNet official Git-LFS weights
rm -rf "$BENCH/tmp/DSTNet_weights"
git lfs install >/dev/null 2>&1 || true
git clone --depth 1 https://github.com/xuboming8/DSTNet.git "$BENCH/tmp/DSTNet_weights"
(cd "$BENCH/tmp/DSTNet_weights" && git lfs pull)
mkdir -p "$BENCH/weights/dstnet"
cp "$BENCH/tmp/DSTNet_weights/experiments/GOPRO.pth" "$BENCH/weights/dstnet/"
cp "$BENCH/tmp/DSTNet_weights/experiments/DVD.pth" "$BENCH/weights/dstnet/"
cp "$BENCH/tmp/DSTNet_weights/experiments/BSD.pth" "$BENCH/weights/dstnet/"

# Shift-Net+ official Google Drive checkpoints
mkdir -p "$BENCH/weights/shiftnet"
gdown 1f79zxmCL-ygVmoJd86OT6uksPGk2BpL0 -O "$BENCH/weights/shiftnet/net_gopro_deblur.pth"
gdown 1vPQkAznRaVawQOMOuvhnCD8DKS2RQToM -O "$BENCH/weights/shiftnet/net_dvd_deblur.pth"

# BSSTNet official Google Drive folder. It contains BSST and RAFT files.
rm -rf "$BENCH/tmp/bsstnet_drive"
gdown --folder 'https://drive.google.com/drive/folders/19v8wsg8aWayaVhNBmnj2vk4LrvmdViW8' \
  -O "$BENCH/tmp/bsstnet_drive"
mkdir -p "$BENCH/weights/bsstnet"
find "$BENCH/tmp/bsstnet_drive" -type f -iname 'BSST_gopro.pth' -exec cp {} "$BENCH/weights/bsstnet/BSST_gopro.pth" \;
find "$BENCH/tmp/bsstnet_drive" -type f -iname 'BSST_dvd.pth' -exec cp {} "$BENCH/weights/bsstnet/BSST_dvd.pth" \;
find "$BENCH/tmp/bsstnet_drive" -type f -iname 'raft-things.pth' -exec cp {} "$BENCH/weights/bsstnet/raft-things.pth" \;

for required in \
  "$BENCH/weights/realvdeblur/realvdeblur_dmd.safetensors" \
  "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  "$BENCH/weights/realvdeblur/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  "$BENCH/weights/dstnet/GOPRO.pth" "$BENCH/weights/dstnet/DVD.pth" "$BENCH/weights/dstnet/BSD.pth" \
  "$BENCH/weights/shiftnet/net_gopro_deblur.pth" "$BENCH/weights/shiftnet/net_dvd_deblur.pth" \
  "$BENCH/weights/bsstnet/BSST_gopro.pth" "$BENCH/weights/bsstnet/BSST_dvd.pth" \
  "$BENCH/weights/bsstnet/raft-things.pth"; do
  test -s "$required" || { echo "Missing weight: $required" >&2; exit 1; }
done

find "$BENCH/weights" -type f -print0 | sort -z | xargs -0 sha256sum > "$BENCH/manifests/weights.sha256"
echo "Weights downloaded and hashed: $BENCH/manifests/weights.sha256"
