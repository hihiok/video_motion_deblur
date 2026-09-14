"""Pinned GoPro Ours-s executable training recipe, adapted to three domains."""
import copy
import math

TRAINING_RECIPE = 'shiftnet_official_recipe_crop256_t13_v3'


def training_settings():
    return dict(training_recipe=TRAINING_RECIPE, frames=13, training_context=1,
                crop_size=256, training_spatial_mode='paired_random_crop',
                evaluation_outputs=12, seed=10, clips_per_update=8,
                total_updates=300000, lr=4e-4, min_lr=1e-7, warmup_updates=0,
                optimizer='AdamW', betas=[0.9, 0.99], weight_decay=0.,
                loss='L1', grad_clip=0.01, precision='fp16_amp_gradscaler',
                initialization='random', teacher_training=False,
                n_frames_per_video=100, sampling='domain_shuffled_windows',
                domain_sampling={'gopro':.5, 'dvd':.25, 'bsd':.25},
                train_partition='all_official_train', validate_every=0,
                save_every=1000, workers=2)


def validate_training_settings(config):
    for key, value in training_settings().items():
        if config.get(key) != value:
            raise ValueError(f'Expected {key}={value!r}; migrate old config with shift500.prepare_official')


def learning_rate(step, config):
    # Upstream steps CosineAnnealingLR before updates 2..N: update 1 uses lr_max.
    return config['min_lr'] + (config['lr']-config['min_lr'])*.5*(1+math.cos(math.pi*step/config['total_updates']))


def official_training_partition(source):
    """Return internal holdout to official train; never move official test frames."""
    manifest=copy.deepcopy(source)
    if source.get('split_audit', {}).get('gt_file_sha256_cross_split_check') != 'passed':
        raise ValueError('Need a successfully audited manifest')
    manifest['train']=sorted(manifest['train']+manifest['val'],key=lambda r:(r['domain'],r['name']))
    keys=[(r['domain'],r['name']) for r in manifest['train']]
    test_keys={(r['domain'],r['name']) for r in manifest['test']}
    if len(keys)!=len(set(keys)) or set(keys)&test_keys:
        raise ValueError('Duplicate official train clip or train/test overlap')
    manifest['val']=[]
    manifest.update(version=4, frames=13, split_policy='all official train; untouched official test; no validation selection')
    for d in ('gopro','dvd','bsd'):
        audit=manifest['split_audit'].get(d)
        if audit is not None:
            audit['previous_train_val_holdout_groups']=audit.pop('train_val_holdout_groups', [])
            audit['internal_holdout_policy']='merged back into official train'
    return manifest


def training_targets(gt, config):
    context = config['training_context']
    return gt[:, context:-context]
