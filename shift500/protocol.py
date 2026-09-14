"""The user-authorized upstream training input recipe; deployment is separate."""
TRAINING_RECIPE = 'shiftnet_official_input_crop256_t13_v2'


def training_settings():
    return dict(training_recipe=TRAINING_RECIPE, frames=13, training_context=1,
                crop_size=256, training_spatial_mode='paired_random_crop',
                evaluation_outputs=12)


def validate_training_settings(config):
    for key, value in training_settings().items():
        if config.get(key) != value:
            raise ValueError(f'Expected {key}={value!r}; migrate old config with shift500.prepare_crop')


def training_targets(gt, config):
    context = config['training_context']
    return gt[:, context:-context]
