# SUPERSEDED — use strict BSD train/test guide

This original execution guide has been superseded for the current experiment because the BSD source policy was tightened.

Use instead:

`CODEAGENT_NANOVNR_NAFNET_RGB_FULLFRAME_BSD_TRAIN_TEST_20260904.md`

Mandatory BSD policy:
- training may read only `/mnt/ssd1/z00919662/datasets/BSD/train`
- evaluation/audit may read only `/mnt/ssd1/z00919662/datasets/BSD/test`
- do not scan or sample any `BSD/<config>/train` or `BSD/<config>/test` directory

The model remains `NanoVNRNAFNetRGB`, matching the user-supplied NAFNet structure with the sole architecture change of RGB input (`Conv2d(3,12,3,1,1)` instead of `Conv2d(4,12,3,1,1)`).
