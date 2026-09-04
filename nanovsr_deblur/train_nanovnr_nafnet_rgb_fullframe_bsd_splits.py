"""Strict BSD train/test protocol entrypoint for NanoVNR NAFNet RGB training.

This wrapper intentionally reuses the existing training implementation and only
changes the recipe identity. Dataset selection is enforced in data/mixed_deblur.py:
for family BSD, only <BSD_ROOT>/train and <BSD_ROOT>/test are eligible.
"""

import train_nanovnr_nafnet_rgb_fullframe as _base


_base.RECIPE_ID = 'nanovnr_nafnet_rgb_native_fullframe_mix_bsd_train_test_v2'


if __name__ == '__main__':
    _base.main()
