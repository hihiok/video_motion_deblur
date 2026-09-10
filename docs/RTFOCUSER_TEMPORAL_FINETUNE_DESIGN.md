# Causal RT-Focuser temporal fine-tuning v1

The task is to preserve useful pretrained single-frame deblurring while reducing temporal flicker.
This branch adds a learned causal module to the full Standard backbone. It does not use the
pruned T3/T6 Shift/DST candidate; existing experiments remain separate.

## Model

`rtf_temporal/model.py` reuses the checkpoint-compatible MIT backbone in `rtf_t6/model.py`.
All Standard depths `[3,4,4,3,2]`, channels, nearest feature resizing, bilinear decoder upsampling,
BatchNorm buffers and final residual/sigmoid are preserved. Arbitrary input sizes are padded to
multiples of 16, then the output is sliced back to its original dimensions.

A temporal residual is inserted after `Up_conv4`, at H/4 x W/4 with 128 channels. It compresses
the feature to 32 channels and forms a context from current, previous and absolute-difference
features. Pointwise/depthwise convolutions predict a sigmoid gate and bounded candidate update.
The 128-channel output projection starts at zero. Thus every initial frame equals the original
backbone, while the projection has a gradient on subsequent frames. History is an explicit tensor,
never hidden mutable module state; activation checkpoint recomputation cannot advance it twice.

Training retains the recurrent gradient within T=4 and resets at each sampled clip. Evaluation
and inference retain state throughout each sequence, with no four-frame reset. The first frame,
cut and resolution change are cold starts. Reset masks clear history before learned operations.
No future frames, flow inference, feature warping, output-frame blending or extra frame delay.
Large motion remains a limitation of this unaligned first version; a low temporal metric alone
does not establish absence of ghosting.

## Optimization and evaluation

- Official weights are loaded strictly into the unchanged backbone; no partial match is accepted.
- Updates 1–2,000 train the temporal adapter; 2,001–20,000 jointly fine-tune the backbone.
- BatchNorm running statistics stay frozen in both phases. Backbone affine parameters train in phase 2.
- Reconstruction MSE has weight 1. Motion-aligned GT-relative temporal L1 ramps to 0.01 over 2,000 updates.
- GT-only Farneback flow is estimated at half resolution and scaled to full image coordinates.
  Forward/backward consistency, photometric confidence, image bounds and cut masks limit supervision.
  It needs no external flow weights, but is less reliable on large/complex motion than a strong learned flow estimator.
- Full native frames, T=4, two clips/eight frames per global update for both one and two GPUs.
  Single GPU accumulates twice; two GPUs take one clip per rank. Each global sample index has a
  deterministic domain/sequence/start/augmentation, so resume and GPU-count changes preserve data budget.
- Data holdout uses official training sequences only, with GoPro acquisition chunks grouped together.
  Official test is untouched. The fixed 32-frame development windows are not full test-set benchmarks.
- A checkpoint qualifies only if every domain loses <=0.2 dB PSNR, has non-worse aligned temporal L1,
  and >=5% valid-flow coverage. `best_stable` additionally requires strict balanced temporal improvement.
- Stage 2 is automatic. A latest checkpoint is a recovery artifact, not a claim of improved quality.

## Measured arithmetic (untrained model, CPU hooks)

The deployed Standard implementation has 5,864,531 parameters; the adapter adds 13,888 (0.237%).
Total is 5,878,419. This is the actual code count rather than the rounded paper claim of 5.85M.

| Native image | Baseline Conv/SN/Linear GMAC/frame | With adapter | Added GMAC | With adapter GFLOPs at 2 FLOPs/MAC |
|---|---:|---:|---:|---:|
| 640×360 (padded H=368) | 55.322728 | 55.522920 | 0.200192 | 111.045839 |
| 1280×720 | 216.480175 | 217.263535 | 0.783360 | 434.527070 |

These counts include both frames of a causal call divided by two; the same adapter convolutions
run at cold start. Increase is 0.362% in counted MACs. Activation/BN/interpolation/pooling/gate
elementwise operations and memory traffic are excluded. This is not measured latency or total
hardware utilization. Do not extrapolate speed gains directly from these numbers.

## Validation before publication

- 8 local tests passed, including an exact FP32 comparison against the official source file
  `ReaganWu/RT-Focuser/model/rt_focuser_model.py` (Git blob `c314f62633ebdbc5c5614cfc42f04e495d69fa57`).
- Tested native odd-size padding, causal/step equivalence, no future-frame influence, reset isolation,
  cross-frame gradients, frozen backbone/BN, flow direction and valid-mask normalization.
- Real Standard-model CPU integration exercised both training phases, all three synthetic dataset
  layouts, preflight, validation, checkpoint write and resume at a stage boundary.
- Local environment: PyTorch 2.14.0+cu130, CPU execution. Server retains its existing torch >=2.4.
- Two-rank Gloo was attempted but blocked by container socket permissions. It is opt-in through
  `RTF_TEST_DDP=1` in the same integration test, and required on the server when choosing dual GPU.
- Actual A100/NCCL/BF16/native-data peak memory, training throughput and quality are not yet measured.
  The supplied preflight executes backward and optimizer steps at each domain's largest image in
  both phases, discards test weights, and binds success to code/config/weights/device identity.

Complete execution instructions are in `CODEAGENT_RTFOCUSER_TEMPORAL_FINETUNE_A100_20260910.md`.
