# CodeAgent execution handoff

## Objective

Run RealVDeblur, BSSTNet, DSTNet and Shift-Net+ on the same business frame sequence and produce aligned frame folders plus MP4 outputs for subjective comparison.

Do not train or fine-tune. Do not compute PSNR/SSIM without ground truth.

## Fixed paths

```text
Workspace: /mnt/ssd1/z00919662/motion_deblur
Input frames: /mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed
Input MP4: /mnt/ssd1/z00919662/motion_deblur/input/xiaobieli38_trimmed.mp4

Existing repos:
  envs/bsstnet_repo
  envs/dstnet_repo
  envs/shiftnet_repo

Clone:
  envs/realvdeblur_repo <- https://github.com/OpenImagingLab/RealVDeblur
```

## Execution

```bash
cd /mnt/ssd1/z00919662/motion_deblur
git clone https://github.com/hihiok/video_motion_deblur.git benchmark_code
cd benchmark_code

ROOT=/mnt/ssd1/z00919662/motion_deblur bash scripts/setup_repos_and_envs.sh
ROOT=/mnt/ssd1/z00919662/motion_deblur bash scripts/download_weights.sh
ROOT=/mnt/ssd1/z00919662/motion_deblur CODE=$PWD GPU=0 bash run_all.sh --all
```

## Required smoke gate

Before committing to a long full run, use the first 24 frames by creating a temporary input folder and invoke each adapter directly. Inspect at least frames 0, middle and last.

Reject a run when any of the following occurs:

- frame count mismatch;
- RGB/BGR reversal;
- black/constant/NaN output;
- output nearly identical to input due to a checkpoint-loading failure;
- severe border seams;
- temporal chunk seams;
- first or last frames missing;
- checkpoint loaded with missing/unexpected keys;
- downloaded file is HTML or a Git-LFS pointer;
- BSSTNet deformable convolution is replaced by a stub.

## Official checkpoint matrix

```text
RealVDeblur: DMD + Wan2.1-T2V-1.3B + Wan VAE
BSSTNet: BSST_gopro.pth, BSST_dvd.pth, raft-things.pth
DSTNet: GOPRO.pth, DVD.pth, BSD.pth
Shift-Net+: net_gopro_deblur.pth, net_dvd_deblur.pth
```

Do not collapse GoPro/DVD/BSD runs into one. The business stream is out-of-domain and checkpoint source matters.

## Adapter notes

### RealVDeblur

The wrapper launches the official `OpenImagingLab/RealVDeblur/inference.py` with:

```text
num_inference_steps=1
enable_twm=true
temporal_window_size=21
stride=1
```

It accepts the canonical frame directory. If float16 fails, retry bfloat16 only when the GPU supports it.

### DSTNet

- Official `Deblur(num_feat=64, num_block=15)`.
- Tensor format: `B,T,C,H,W` in RGB `[0,1]`.
- H/W reflection-padded to a multiple of 4.
- Temporal clips: 30, overlap: 10.
- Overlap is linearly blended.

### Shift-Net+

- Official `GShiftNet(future_frames=2, past_frames=2)`.
- Two reflected context frames on both sides.
- Default central chunk length: 48.
- The wrapper accepts either a 5D batch output or the official valid `T-4` output.

### BSSTNet

- Official BSST architecture plus official RAFT.
- RAFT checkpoint must appear as `bsstnet_repo/model_zoos/raft-things.pth`; the adapter creates a symlink.
- The official sparse transformer fixes `fold_feat_size=(64,64)`, corresponding to 256×256 image patches.
- Use patch size 256, overlap 64.
- Compute full-sequence 1/4-scale bidirectional flow, crop matching flow patches, then blend output patches.
- Temporal clips: 48, overlap: 16.

## Outputs

```text
benchmark/outputs/
  realvdeblur_dmd/
  bsstnet_gopro/
  bsstnet_dvd/
  dstnet_gopro/
  dstnet_dvd/
  dstnet_bsd/
  shiftnet_gopro_plus/
  shiftnet_dvd_plus/
```

For each completed run verify:

```text
frames/
output.mp4
run_metadata.json
check_report.json
```

## Final comparison

Prepare side-by-side videos and still crops. Evaluate separately:

- blur removal strength;
- face/identity preservation;
- text and edge recovery;
- hallucinated texture;
- flicker;
- color/brightness shift;
- ringing/oversharpening;
- residual blur;
- scene-transition behavior;
- speed and peak VRAM.

Do not force one overall winner. RealVDeblur may win perceptual sharpness while losing fidelity; the conventional models may preserve observed content better.
