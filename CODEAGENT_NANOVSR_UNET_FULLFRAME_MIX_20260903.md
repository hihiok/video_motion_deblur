# DEPRECATED EXPERIMENT GUIDE

This older guide described a NanoVSR-Deblur U-Net whose recurrent hidden state lived at 1/4 spatial resolution.

Do NOT use it for the current quality-first experiment.

Use instead:

`CODEAGENT_NANOVSR_UNET_FULLRES_RECURRENCE_MIX_20260903.md`

Current required architecture:
- `NanoVSRFullResUNetDeblur`
- recurrent hidden state remains full HxW resolution at every time step
- native full-frame training
- no crop / no resize
- GoPro + DVD + BSD
- Charbonnier-only
- T=7 -> T=30 curriculum

The old 1/4-scale recurrent implementation remains in the branch only as an archived comparison implementation and must not be trained by the current CodeAgent task.
