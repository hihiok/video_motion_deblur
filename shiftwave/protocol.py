"""Fixed warm-start compression experiment; NOT an official-recipe reproduction."""
import math
from shift500.protocol import training_targets, learning_rate
from .model import MODEL_ID


def training_settings():
    return dict(training_recipe=MODEL_ID, frames=13, training_context=1,
                crop_size=256, training_spatial_mode='paired_random_crop',
                evaluation_outputs=12, seed=10, clips_per_update=8,
                total_updates=120000, lr=1e-4, min_lr=1e-7, warmup_updates=0,
                optimizer='AdamW', betas=[.9,.99], weight_decay=0.,
                loss='L1+HaarHF+outputKD', grad_clip=.01,
                precision='fp16_amp_gradscaler', initialization='pretrained_retained_tensors',
                teacher_training=False, n_frames_per_video=100,
                sampling='domain_shuffled_windows',
                domain_sampling={'gopro':.5,'dvd':.25,'bsd':.25},
                train_partition='all_official_train', validate_every=0,
                save_every=1000, workers=2, hf_weight=.05, kd_weight=.1,
                kd_end_update=100000, required_training_gpus=2,
                target_gopro_rgb8_psnr=33., target_gflops=500.)


def validate_training_settings(c):
    for k,v in training_settings().items():
        if c.get(k)!=v:
            raise ValueError(f'Expected {k}={v!r}; do not change recipe on server')


def distillation_weight(step, c):
    if step>=c['kd_end_update']:
        return 0.
    return c['kd_weight']*.5*(1+math.cos(math.pi*step/c['kd_end_update']))
