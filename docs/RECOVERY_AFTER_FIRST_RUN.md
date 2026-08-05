# Recovery after the first CodeAgent run

## Current state reported by CodeAgent

```text
Available:
  RealVDeblur DMD checkpoint
  Wan2.1 diffusion checkpoint
  DSTNet GoPro/DVD/BSD checkpoints
  Shift-Net+ GoPro/DVD checkpoints

Missing:
  Wan2.1_VAE.pth
  BSSTNet GoPro/DVD checkpoints
  BSSTNet raft-things.pth

Observed blockers:
  conda/pip SSL errors
  DSTNet mmcv import
  Shift-Net 5D reflect-padding error
```

The repository has now been updated to address the code/environment blockers:

- Shift-Net pads flattened 4D frames and reshapes back to `B,T,C,H,W`.
- DSTNet no longer needs mmcv for inference.
- When CuPy is absent, DSTNet uses an inference-equivalent PyTorch `unfold` implementation for the released per-pixel dynamic depthwise convolution.
- `run_all.sh` defaults to the already-existing `turtle_joint_py222` environment.
- The recovery workflow clones that local environment instead of downloading a new conda stack.
- The Wan VAE downloader uses the official direct URL, resume support and exact SHA256 verification.

## Execute now

```bash
cd /mnt/ssd1/z00919662/motion_deblur/benchmark_code
git pull

ROOT=/mnt/ssd1/z00919662/motion_deblur \
SOURCE_ENV=turtle_joint_py222 \
RUNTIME_ENV=deblur_runtime \
GPU=0 \
bash scripts/recover_after_codeagent.sh
```

This command performs a 24-frame smoke run. It should attempt, in this order:

1. verified Wan VAE recovery;
2. Shift-Net+ GoPro and DVD;
3. DSTNet GoPro, DVD and BSD;
4. RealVDeblur when all required files and imports are present;
5. BSSTNet only when its three official files are present.

## Only when the proxy certificate causes pip/curl SSL failures

First retry the normal command. If the failure is specifically certificate verification by the trusted internal proxy, run:

```bash
ROOT=/mnt/ssd1/z00919662/motion_deblur \
SOURCE_ENV=turtle_joint_py222 \
RUNTIME_ENV=deblur_runtime \
INSTALL_MISSING=1 \
ALLOW_INSECURE_SSL=1 \
GPU=0 \
bash scripts/recover_after_codeagent.sh
```

`ALLOW_INSECURE_SSL=1` is intentionally opt-in and must not be used on an untrusted network.

## Wan VAE verification

Expected path:

```text
/mnt/ssd1/z00919662/motion_deblur/benchmark/weights/realvdeblur/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

Expected SHA256:

```text
38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981
```

The official file is approximately 508 MB. Delete any file that is only a few bytes, an HTML response, or an Xet pointer.

## DSTNet expectation

The log must state one of:

```text
dynamic_backend=official_cupy
dynamic_backend=pytorch_unfold
```

`pytorch_unfold` is slower but removes the mmcv/CuPy installation blocker. It changes only the implementation of the released dynamic depthwise convolution; model weights and architecture remain unchanged.

Run DSTNet in float32 during the first smoke test. Do not enable AMP until output correctness is confirmed.

## Shift-Net expectation

The previous error:

```text
F.pad() does not support reflect mode on 5D tensors
```

must no longer occur. Confirm exactly 24 output frames and inspect first/middle/last frames.

## RealVDeblur expectation

RealVDeblur is attempted only after all three files are present:

```text
realvdeblur_dmd.safetensors
diffusion_pytorch_model.safetensors
Wan2.1_VAE.pth
```

If the official CLI does not import in `deblur_runtime`, capture:

```bash
conda run -n deblur_runtime python \
  /mnt/ssd1/z00919662/motion_deblur/envs/realvdeblur_repo/inference.py --help
```

Do not create another environment until the exact missing package or Python-version error is known.

## BSSTNet

BSSTNet is still blocked by the official Google Drive files. Required paths:

```text
benchmark/weights/bsstnet/BSST_gopro.pth
benchmark/weights/bsstnet/BSST_dvd.pth
benchmark/weights/bsstnet/raft-things.pth
```

Do not substitute unofficial or randomly named checkpoints. Do not mark BSSTNet as tested until all three files load with strict checkpoint matching.

## Required report back

Return:

```text
1. git commit checked out
2. runtime environment name and Python/Torch/CUDA versions
3. Wan VAE SHA256 result
4. output frame count for every attempted checkpoint
5. check_report.json result for every completed output
6. exact remaining traceback, if any
7. peak VRAM and runtime for each successful smoke test
```

Do not start the full video until at least one GoPro/DVD checkpoint per available conventional model passes the 24-frame smoke gate.
