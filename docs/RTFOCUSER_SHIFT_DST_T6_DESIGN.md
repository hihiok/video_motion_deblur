# RT-Focuser + Shift-Net-s + DSTNet: T=6 design

## Decision

Use RT-Focuser as the checkpoint and image-restoration baseline. Do not run
three complete networks in series. The candidate keeps the RT-Focuser
LD/MLIA/X-Fuse encoder-decoder and inserts two parameter-free temporal
operators into its feature path:

1. **Grouped spatial-temporal shift**: selected channel groups take features
   from the previous/next frame and one-pixel spatial neighbors. This transfers
   the zero-parameter grouped-shift principle from Shift-Net-s.
2. **Similarity-gated bidirectional propagation**: at the H/8 feature level,
   forward and backward recurrent evidence is blended only where it is similar
   to the current feature. Large motion/difference suppresses propagation. This
   transfers DSTNet's bidirectional propagation and discriminative-fusion idea
   without its learned dynamic-convolution branch.

The shift output is mixed with the current feature at strength 0.5. The
bidirectional branch uses strength 0.25. Both are fixed operations and add no
model parameters.

## Checkpoint inheritance

The frame backbone preserves the official RT-Focuser module names and tensor
shapes. The official GoPro checkpoint is mapped below the `backbone.` prefix.
The default candidate removes the last LD block from encoder2, encoder3 and
encoder4; all retained target tensors load from the checkpoint.

Verified checkpoint:

```text
GoPro_RT_Focuser_Standard_256.pth
SHA256 6bc1310af51b2e47cf631c987ce7da1057aa469c3770d973195b2856014ad3fb
```

Local verification against official RT-Focuser source at commit
`4c8e12d28c2801f34cc1153e9ad8702b7bce657a` produced exactly equal frame-model
outputs (`max_abs_error=0`) for a divisible test resolution. Candidate loading
reported 100% coverage of retained target state tensors; removed source block
keys are explicitly listed in the complexity report.

## Complexity gate

The repository counts all Conv2d, functional SN depthwise convolution and
Linear MACs, then adds a conservative elementwise-op estimate for the temporal
operators. Interpolation and activations are not included, matching common MAC
reporting conventions. T=6 cost is divided by six because the model outputs six
restored frames.

| Metric | RT-Focuser Standard | T=6 candidate | Change |
|---|---:|---:|---:|
| Parameters | 5,864,531 | 5,710,131 | -2.63% |
| Conv/SN MAC + temporal estimate @256x256, per output frame | 15.394 G | 15.182 G | -1.38% |
| Same estimate @1280x720, per output frame | 216.49 G | 213.50 G | -1.38% |

The RT-Focuser paper reports 15.76 GMAC at 256x256. Applying the measured
candidate ratio to that external number gives approximately 15.54 GMAC at
256x256 or 218.6 GMAC per 720p output frame. A T=6 clip therefore costs about
1.31 TMAC in total while producing six frames. MAC and FLOP must not be mixed;
some profilers count one MAC as two FLOPs.

Both parameter and arithmetic comparisons are executable hard gates. Training
stops before the first update if the candidate exceeds the baseline.

## Training protocol

- Input/target: six consecutive aligned RGB frames.
- Datasets: GoPro, BSD and DVD, sampled 1:1:1 so the largest dataset cannot
  dominate.
- Crop: shared 256x256 crop across all six blur/GT frames; flip, rotation and
  temporal reversal are also shared.
- Optimizer: AdamW, initial LR 5e-5, 2k warmup, cosine decay to 1e-6.
- Default duration: 180k iterations, batch 1, gradient accumulation 2, AMP,
  EMA 0.999, gradient clipping 1.0.
- Selection: unweighted mean of GoPro/BSD/DVD validation PSNR, not pooled frame
  PSNR.

Loss:

```text
L = Charbonnier
  + 0.05 * FFT
  + 0.10 * edge
  + ramp * 0.15 * first-order temporal-difference matching
  + ramp * 0.03 * second-order temporal-difference matching
```

Temporal terms compare prediction motion with GT motion. They do not force
adjacent outputs to be identical, which would reduce flicker by producing
motion blur or ghost trails. Temporal loss starts at iteration 5k and ramps for
20k iterations.

## Required evidence

Training is not itself proof that the model is better. Accept the candidate
only if all conditions hold:

1. GoPro, BSD and DVD each improve over input PSNR and do not regress severely
   against the official RT-Focuser baseline.
2. Balanced PSNR improves, or stays essentially flat while temporal residual
   error and subjective flicker clearly improve.
3. Business-video crops show no new double edges, trails, face deformation,
   color shift or six-frame boundary pulse.
4. Complexity JSON continues to pass on the exact committed configuration.

Recommended ablations use the same code/config: set `shift_strength: 0` for
pruned RT-Focuser without shift; set `propagation_strength: 0` for shift only;
use both defaults for the full candidate.
