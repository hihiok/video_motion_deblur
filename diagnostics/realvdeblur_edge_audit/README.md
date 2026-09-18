# RealVDeblur edge diagnostics — read-only audit

This tool collects evidence for jagged/soft edges. It is not a deblurring fix and does not change the official checkout, existing FP16/BF16 frames, weights, or environment.

## Scope

- Target official revision: `OpenImagingLab/RealVDeblur@63a2eb0ed4a22cb76b796a65d5d3052616352573`.
- Reuse the existing working Python/CUDA environment; no installation, downgrade, training, or automatic repairs.
- Run on the current target server, not over SSH.
- Read existing 452-frame results; run only a paired 24-frame diagnostic clip, then three-frame VAE tests.
- Compare source/code hashes, canonical/source-video sizes, exact filenames, LoRA key coverage and sampled weight updates.
- GPU selection uses numeric free MiB from `nvidia-smi`, pins a UUID before importing PyTorch, and reports the physical index plus logical `cuda:0`. Respect scheduler allocations with `--allowed-gpus`.
- The 24 GiB free-memory guard is a conservative diagnostic guard, not a measured model minimum.

## Entry points

Use an isolated NEW output such as `benchmark/diagnostics/realvdeblur_edge_audit_<timestamp>`.

```bash
PY=/absolute/path/to/the/ALREADY_WORKING/environment/bin/python
SCRIPT=/absolute/path/to/diagnostics/realvdeblur_edge_audit/audit_realvdeblur.py
OUT=/absolute/path/to/a/new/diagnostic/directory

"$PY" "$SCRIPT" audit --out "$OUT"
"$PY" "$SCRIPT" capture --out "$OUT" --precision bfloat16 --clip-start 214
"$PY" "$SCRIPT" vae --out "$OUT" --precision bfloat16
"$PY" "$SCRIPT" capture --out "$OUT" --precision float16 --clip-start 214
"$PY" "$SCRIPT" vae --out "$OUT" --precision float16
"$PY" "$SCRIPT" verify --out "$OUT"
"$PY" "$SCRIPT" summary --out "$OUT"
```

Run each command in a separate process. Inspect return codes and retained failure logs; do not continue GPU stages on incompatible or changed source. A changed `model/`, `utils/`, or `inference.py` is exported as a private diff. Do not reset those changes: return them for review. The artifact directory must not overlap protected production directories.

`--root` and `--repo` can override the default server paths. `--clip-start` is a zero-based ordinal in the verified filename order. Use the SAME start for FP16 and BF16. Default is the center 24 frames. The center three relative frames 10, 11, 12 are sampled. Native-resolution frames enter the model; ROIs are created only AFTER inference for display.

## Controlled experiments

1. **Export arithmetic:** capture the same decoded RGB values, compare official low-precision image export with FP32 range conversion. Preserve truncation in both; do not also change rounding.
2. **Same-latent decoder:** intercept (capture only) the latent entering the official VAE decode. Replay it using a fresh native-precision VAE and a fresh FP32 VAE loaded independently from the original checkpoint. DiT/conditioning/noise/latent values do not change within this comparison. Record checkpoint storage dtypes; casting cannot recover precision absent from the checkpoint.
3. **Replay gate:** native replay must match captured native RGB. If `REVIEW_NATIVE_REPLAY_MISMATCH`, treat decoder-only attribution as inconclusive. Do not relax the gate autonomously.
4. **VAE-only roundtrip:** input -> encoder -> decoder, without DiT. Native and FP32 recipes include their corresponding input normalization precision. These test reconstruction fidelity, not deblur performance.
5. **Paired FP16/BF16:** matched 24-frame context, seed 0, step 1, TWM 21. These are paired diagnostics, NOT numerically equivalent to the historical 452-frame results.

TF32 and autocast are disabled for the FP32 reference. No VAE interpolation, scheduler, LoRA alpha, temporal stride, or network math is edited. Hooks are confined to the diagnostic process, call the original decode, and are restored afterward.

## Evidence and interpretation

`static.json`, source hashes/private diff, `capture_*/lora_audit.json`, `capture_*/captured_tensors.pt`, `vae_tests_*/vae_report.json`, PNG sheets, and `preservation_check.json` are the evidence. `MEASUREMENTS_SUMMARY.md` is a measurements index, not a certified root cause.

1x sheets preserve pixels. 4x NEAREST sheets are labelled pixel inspection and necessarily enlarge pixel steps; they cannot alone prove new aliasing. Automatic ROIs use only input gradients and coordinates, not semantic person detection or a validated aliasing score. Inspect matching full frames and consecutive frames.

Between-output PSNR/MAE measure differences, not quality. Sharper gradients do not prove more faithful edges. Native PNG problems must be distinguished from viewer resampling, input aliasing, combing, ringing, and temporal contour changes. A 24-frame test that does not reproduce an artifact cannot clear the 452-frame path.

## Validation and prerequisites

Required existing packages: PyTorch, NumPy, Pillow, and the official RealVDeblur dependencies. OpenCV and ffprobe source-video checks are optional; absence is reported rather than treated as a blocker. No ffmpeg/MP4 generation is required.

```bash
python diagnostics/realvdeblur_edge_audit/test_audit.py
```

Fifteen CPU regression tests cover image comparison/export, symlinks, numeric GPU selection/allocation, source differences, and protected outputs. These are not real GPU/model tests. Actual checkpoint/CUDA/visual results must be obtained on the target server.

## Network and private configuration

Network access is only for obtaining the diagnostic code. Use the user's LOCAL execution MD for proxy exports and the requested Git `http.sslVerify=false` setting before fetch/clone. Disabling TLS verification weakens transport verification; restrict it to the requested inspected environment, prefer an approved CA where available. Do not upload proxy/SSH credentials, source diffs containing secrets, business images, or reports to this public repository. An offline copy of these files can be used instead of GitHub.

## Technical references

- Pinned pipeline: https://github.com/OpenImagingLab/RealVDeblur/blob/63a2eb0ed4a22cb76b796a65d5d3052616352573/model/src/realvdeblur_pipeline.py
- Pinned VAE: https://github.com/OpenImagingLab/RealVDeblur/blob/63a2eb0ed4a22cb76b796a65d5d3052616352573/model/src/wan_video_vae.py
- Pinned loader: https://github.com/OpenImagingLab/RealVDeblur/blob/63a2eb0ed4a22cb76b796a65d5d3052616352573/utils/data_load.py
- Pinned LoRA loader: https://github.com/OpenImagingLab/RealVDeblur/blob/63a2eb0ed4a22cb76b796a65d5d3052616352573/model/src/lora.py
- Wan Diffusers recommends FP32 VAE decoding: https://huggingface.co/docs/diffusers/api/pipelines/wan . This motivates a TEST; it does not establish the cause in this RealVDeblur implementation, and does not authorize replacing its VAE or scheduler.
- CUDA device UUID selection: https://docs.nvidia.com/deploy/topics/topic_5_2_1.html
