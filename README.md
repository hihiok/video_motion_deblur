# Video Motion Deblur — Four-Model Business Stream Benchmark

## New: generic mixed-degradation restoration benchmark

The `generic_restoration/` package adds a same-business-stream, source-resolution comparison of:

- RealViformer on the old V100 server;
- FlashVSR v1.1, SeedVR2-3B, and DOVE Final on the RTX PRO 6000 Blackwell server.

It includes pinned official repositories, separate environments, canonical MP4 decoding, source SHA256 checks across servers, exact frame-count validation, original timing/audio remuxing, Blackwell build paths, and mandatory 25-frame manual smoke gates. Start with [`generic_restoration/README.md`](generic_restoration/README.md) and the two CodeAgent instruction files in that directory.

Unified inference wrappers for comparing the following official video-deblurring models on the same business video:

- **RealVDeblur** — `OpenImagingLab/RealVDeblur`
- **BSSTNet** — `huicongzhang/BSSTNet`
- **DSTNet** — `xuboming8/DSTNet`
- **Shift-Net+** — `dasongli1/Shift-Net`

Default business input:

```text
/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed
/mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4
```

Existing repositories:

```text
/mnt/ssd1/z00919662/motion_deblur/envs/bsstnet_repo
/mnt/ssd1/z00919662/motion_deblur/envs/dstnet_repo
/mnt/ssd1/z00919662/motion_deblur/envs/shiftnet_repo
```

RealVDeblur repository:

```text
/mnt/ssd1/z00919662/motion_deblur/envs/realvdeblur_repo
```

## Current recommended workflow

The server already has `turtle_joint_py222` with Torch 2.4/CUDA 11.8. The repository now defaults to cloning that local environment rather than downloading old conda stacks.

```bash
cd /mnt/ssd1/z00919662/motion_deblur/benchmark_code
git pull

ROOT=/mnt/ssd1/z00919662/motion_deblur \
SOURCE_ENV=turtle_joint_py222 \
RUNTIME_ENV=deblur_runtime \
GPU=0 \
bash scripts/recover_after_codeagent.sh
```

This performs a 24-frame smoke test and:

- fixes Shift-Net's Torch 2.4 5D reflect-padding issue;
- runs DSTNet without requiring mmcv;
- uses the official CuPy dynamic operator when available;
- otherwise uses an equivalent PyTorch `unfold` inference backend;
- resumes and verifies the missing Wan2.1 VAE download;
- skips unavailable models instead of aborting all remaining models.

Detailed recovery instructions:

```text
docs/RECOVERY_AFTER_FIRST_RUN.md
```

## Full run after smoke tests pass

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
CODE=$PWD \
COMMON_ENV=deblur_runtime \
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

`run_all.sh` verifies required checkpoint files and skips only the model whose files are missing.

## Weights

Weights stay on the server under:

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark/weights
```

Expected files:

| Model | Checkpoints |
|---|---|
| RealVDeblur | Wan2.1 diffusion model, `Wan2.1_VAE.pth`, RealVDeblur DMD |
| BSSTNet | `BSST_gopro.pth`, `BSST_dvd.pth`, `raft-things.pth` |
| DSTNet | `GOPRO.pth`, `DVD.pth`, `BSD.pth` |
| Shift-Net+ | `net_gopro_deblur.pth`, `net_dvd_deblur.pth` |

The verified Wan VAE downloader is:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
bash scripts/download_realvdeblur_vae.sh
```

BSSTNet remains dependent on the authors' official Google Drive folder. Do not substitute an unrelated checkpoint.

## Output structure

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark/outputs/<run>/
├── frames/
├── output.mp4
├── run_metadata.json
└── check_report.json
```

## Fair-comparison behavior

- All models read the same canonical decoded frames.
- Original aspect ratio is preserved.
- Spatial padding is removed before saving.
- First and last frames are preserved.
- DSTNet and BSSTNet use overlapping temporal clips and weighted blending.
- Shift-Net+ uses reflected two-frame context at each boundary.
- BSSTNet follows its official 256×256 patch testing path and official RAFT flow computation.
- RealVDeblur calls the official `inference.py` with Temporal Window Mask.
- GoPro/DVD/BSD checkpoint results remain separate.

## Important limitations

1. The business video has no sharp ground truth, so PSNR/SSIM cannot be computed honestly.
2. RealVDeblur is generative. Sharper-looking output may include hallucinated details.
3. The CUDA models must pass the 24-frame smoke gate on the target server before the full sequence starts.
4. No output should be accepted when frame count, RGB order, borders, temporal order or checkpoint strict-loading checks fail.
5. Do not describe a broken inference result as domain mismatch.
