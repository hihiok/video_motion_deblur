# NanoVSR-Deblur Full-Resolution Recurrent U-Net

Quality-first experiment replacing the old RepVGG propagation with a U-Net recurrent update while keeping the temporal hidden state at full spatial resolution.

Key properties:
- full-resolution recurrent state for both forward and backward propagation
- U-Net internal multi-scale processing only; recurrent input/output remain HxW
- 48/64/96 channels, 2/2/4 residual blocks
- native full-frame training, no random crop, no resize
- GoPro + DVD + BSD family-balanced mixture
- Charbonnier-only loss
- T=7 for steps 1-50000, then T=30 through step 150000
- one continuous Adam + cosine schedule
- AMP + gradient checkpointing

Run instructions are in:
`CODEAGENT_NANOVSR_UNET_FULLRES_RECURRENCE_MIX_20260903.md`
