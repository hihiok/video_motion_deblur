# Generic mixed-degradation video-restoration benchmark

This package compares four released checkpoints on one canonical business MP4:

| Server | Model | Benchmark output | Status of 1x path |
|---|---|---|---|
| Old V100 server | RealViformer | source-size PNG/MP4 | Official fixed-4x network, then area downsample to 1x |
| Blackwell server | FlashVSR v1.1 Tiny Long | source-size PNG/MP4 | Experimental direct `scale=1`; the official project is optimized for 4x |
| Blackwell server | SeedVR2-3B | source-size PNG/MP4 | Native arbitrary target size set equal to input size |
| Blackwell server | DOVE Final | source-size PNG/MP4 | Official `--upscale 1`, with the official hard-coded padding crop corrected |

The goal is blind restoration of mixed real-world degradation, not PSNR ranking. The business stream has no sharp ground truth, so the output must be reviewed for face deformation, text/logo changes, hallucinated detail, flicker, ringing, color shifts, residual blur, and scene-cut behavior.

## Reproducibility controls

- Decode the MP4 once per server into a canonical lossless PNG sequence.
- Record the source MP4 SHA256, canonical-frame SHA256, frame count, size, nominal FPS, frame timestamps, and audio metadata.
- Require the source MP4 SHA256 to match across the old and new servers.
- Pin every official repository to a specific commit.
- Keep one conda environment per model on the Blackwell server.
- Require a 25-frame smoke run and manual preview approval before full inference.
- Reject missing frames, changed geometry, NaN/Inf output, broken checkpoint payloads, and unknown edits in official checkouts.
- Re-encode model PNGs with the original relative frame timing and original audio stream.

## Official repository pins

| Repository | Commit |
|---|---|
| `Yuehan717/RealViformer` | `bd5f88d05ba62136727a61cb162da53f22560465` |
| `OpenImagingLab/FlashVSR` | `6dd38e57203af4efca97df82c659f5d5a2dcf51a` |
| `ByteDance-Seed/SeedVR` | `e4de8c24441a67e1b7df56abea10645059bb1185` |
| `zhengchen1999/DOVE` | `0cd4240442cb5d122839c279977142cb6d648987` |
| `mit-han-lab/Block-Sparse-Attention` | `49d6c39e4dc0303442cda3bb758b3925d4399c49` |

## Entry points

- Old server: `setup_old_realviformer.sh`, then `run_old_realviformer.sh smoke|full`.
- Blackwell server: `setup_new_models.sh all`, then `run_new_models.sh smoke|full all`.
- Complete CodeAgent instructions are in `CODEAGENT_OLD_REALVIFORMER_20260817.md` and `CODEAGENT_NEW_BLACKWELL_3MODELS_20260817.md`.

## Important interpretation rule

The four 1x outputs are useful for subjective screening, but they are not four identical native tasks. RealViformer is structurally fixed to 4x, while FlashVSR is officially optimized for 4x. The report therefore retains each inference path instead of presenting the results as a perfectly controlled 1x leaderboard.
