# Video Motion Deblur — Four-Model Business Stream Benchmark

Unified inference wrappers for comparing these official models on the same business video:

- **RealVDeblur** — `OpenImagingLab/RealVDeblur`
- **BSSTNet** — `huicongzhang/BSSTNet`
- **DSTNet** — `xuboming8/DSTNet`
- **Shift-Net+** — `dasongli1/Shift-Net`

The default test input is:

```text
/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed
/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Existing repositories are expected at:

```text
/mnt/ssd1/z00919662/motion_deblur/envs/bsstnet_repo
/mnt/ssd1/z00919662/motion_deblur/envs/dstnet_repo
/mnt/ssd1/z00919662/motion_deblur/envs/shiftnet_repo
```

RealVDeblur is cloned into:

```text
/mnt/ssd1/z00919662/motion_deblur/envs/realvdeblur_repo
```

## 1. Download this repository on the server

```bash
cd /mnt/ssd1/z00919662/motion_deblur
git clone https://github.com/hihiok/video_motion_deblur.git benchmark_code
cd benchmark_code
```

## 2. Prepare repositories and isolated environments

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
  bash scripts/setup_repos_and_envs.sh
```

The four projects use incompatible dependency stacks, especially BSSTNet's old `mmcv-full`, so they must remain in separate conda environments.

## 3. Download official weights

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
  bash scripts/download_weights.sh
```

Weights are saved under:

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark/weights
```

The script downloads:

| Model | Checkpoints |
|---|---|
| RealVDeblur | Wan2.1 1.3B model, Wan VAE, RealVDeblur DMD |
| BSSTNet | GoPro, DVD, RAFT Things |
| DSTNet | GoPro, DVD, BSD |
| Shift-Net+ | GoPro Ours+, DVD Ours+ |

All files are SHA256-hashed in `benchmark/manifests/weights.sha256`.

## 4. Run inference

Run all models/checkpoints:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
CODE=$PWD \
GPU=0 \
bash run_all.sh --all
```

Run one architecture:

```bash
bash run_all.sh --model=realvdeblur
bash run_all.sh --model=bsstnet
bash run_all.sh --model=dstnet
bash run_all.sh --model=shiftnet
```

Outputs are written to:

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs
```

Each completed run contains:

```text
frames/
output.mp4
run_metadata.json
check_report.json
```

## Fair-comparison behavior

- All models read the same canonical frame folder.
- Original aspect ratio is preserved.
- Spatial padding is removed before saving.
- First and last frames are preserved.
- DSTNet and BSSTNet use overlapping temporal clips and weighted blending.
- Shift-Net+ uses reflected two-frame context at each boundary.
- BSSTNet follows its official 256×256 spatial-patch testing path and official RAFT flow computation.
- RealVDeblur uses the official `inference.py` with Temporal Window Mask enabled.
- GoPro/DVD/BSD checkpoints are kept separate because source-domain choice can materially affect business-video behavior.

## Important limitations

1. The business video has no sharp GT, so PSNR/SSIM cannot be computed honestly.
2. RealVDeblur is generative. A sharper-looking result may contain hallucinated details.
3. The adapters are statically validated here, but the actual CUDA models must be smoke-tested on the target V100/server environment.
4. BSSTNet depends on an old `mmcv-full` stack. Never replace failed deformable-convolution operators with zero/stub implementations.
5. Run a short smoke test and inspect RGB order, frame order, borders, temporal seams, NaNs, and identity output before accepting full-sequence results.

See [docs/HANDOFF_CODEAGENT.md](docs/HANDOFF_CODEAGENT.md) for the execution checklist.
