import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from rtf_t6.losses import temporal_difference_loss, temporal_acceleration_loss
from rtf_t6.model import RT_Focuser_Standard
from rtf_t6.protocol import check_checkpoint_protocol, check_fullframe, inference_window


REPO = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((REPO / 'configs/rtf_t3_gopro_bsd_dvd_fullframe.yaml').read_text())


def test_t3_preserves_optimizer_update_and_frame_budget():
    old = yaml.safe_load((REPO / 'configs/rtf_t6_gopro_bsd_dvd_fullframe.yaml').read_text())
    new = configuration()
    check_fullframe(old)
    check_fullframe(new)
    for config in (old, new):
        train = config['train']
        assert train['total_iters'] // train['gradient_accumulation'] == 90000
        assert train['total_iters'] * train['clip_length'] == 1080000
    for section, keys in [('train', ['warmup_iters', 'validate_every', 'save_every']),
                          ('loss', ['temporal_start_iter', 'temporal_ramp_iters'])]:
        for key in keys:
            assert old[section][key] / 2 == new[section][key] / 4


def test_t3_losses_detect_middle_frame_flicker_with_finite_gradients():
    target = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1, 1).expand(1, 3, 3, 4, 4)
    correct = temporal_difference_loss(target, target) + temporal_acceleration_loss(target, target)
    prediction = target.clone()
    prediction[:, 1] += 0.2
    prediction.requires_grad_()
    loss = temporal_difference_loss(prediction, target) + temporal_acceleration_loss(prediction, target)
    assert loss > correct + 0.1
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 1].abs().sum() > 0


def test_rejects_wrong_window_crop_and_resume_protocol():
    config = configuration()
    assert inference_window(config, None, None) == (3, 2)
    with pytest.raises(ValueError, match='window'):
        inference_window(config, 6, 4)
    with pytest.raises(ValueError, match='overlap'):
        inference_window(config, 3, 3)
    saved = copy.deepcopy(config)
    saved['train']['clip_length'] = 6
    with pytest.raises(ValueError, match='clip_length'):
        check_checkpoint_protocol({'config': saved, 'iteration': 4}, config, resume=True)
    with pytest.raises(ValueError, match='boundary'):
        check_checkpoint_protocol({'config': config, 'iteration': 3}, config, resume=True)
    config['validation']['crop_size'] = 256
    with pytest.raises(ValueError, match='crop_size'):
        check_fullframe(config)


def test_t3_fullframe_train_resume_eval_and_infer(tmp_path):
    """Real CLI chain on tiny native frames; no mocks of forward/backward/I/O."""
    config = configuration()
    config['model'].update(dims=[8, 16, 24, 32, 48], depths=[1]*5, kernels=[3]*5)
    config['train'].update(total_iters=8, samples_per_epoch=4, workers=0,
        validate_every=4, save_every=4, log_every=4, warmup_iters=0)
    config['loss'].update(temporal_start_iter=0, temporal_ramp_iters=0)
    config['validation'].update(max_clips_per_domain=1)
    config['output'] = str(tmp_path / 'run')
    for domain in ('bsd', 'dvd', 'gopro'):
        root = tmp_path / domain
        config['datasets'][domain] = {'roots': [str(root)]}
        # Different native sizes stress cross-domain transitions and padding.
        shape = (20, 24) if domain == 'bsd' else (24, 32)
        for split in ('train', 'val'):
            for kind in ('blur', 'gt'):
                folder = root / split / kind / f'{split}_scene'
                folder.mkdir(parents=True)
                for index in range(7):
                    values = np.random.default_rng(index).integers(0, 200, (*shape, 3), dtype=np.uint8)
                    if kind == 'gt':
                        values += 5
                    Image.fromarray(values).save(folder / f'{index:06d}.png')
    pretrained = tmp_path / 'official_format.pth'
    torch.save(RT_Focuser_Standard(dims=[8,16,24,32,48], depths=[2]*5, kernels=[3]*5).state_dict(), pretrained)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')

    def run(tool, *arguments):
        result = subprocess.run([sys.executable, str(REPO / 'tools' / tool), *map(str, arguments)],
            cwd=REPO, env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr

    run('train_rtf_t6.py', '--config', config_path, '--pretrained', pretrained, '--stop-after', 4)
    checkpoint = Path(config['output']) / 'checkpoints/latest.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['iteration'] == 4
    assert payload['config']['train']['total_iters'] == 8
    assert {int(state['step']) for state in payload['optimizer']['state'].values()} == {1}
    records = [json.loads(line) for line in (Path(config['output']) / 'train_metrics.jsonl').read_text().splitlines()]
    first = next(r for r in records if r['event'] == 'full_frame_first_batch')
    assert first['shape'][1:3] == [3, 3] and not first['crop_applied'] and not first['resize_applied']
    # Resume an actual saved checkpoint, preserving the full LR schedule.
    run('train_rtf_t6.py', '--config', config_path, '--resume', checkpoint, '--output', tmp_path / 'resumed')
    resumed = torch.load(tmp_path / 'resumed/checkpoints/latest.pth', map_location='cpu', weights_only=False)
    assert resumed['iteration'] == 8
    assert {int(state['step']) for state in resumed['optimizer']['state'].values()} == {2}
    run('eval_rtf_t6.py', '--config', config_path, '--checkpoint', checkpoint,
        '--output', tmp_path / 'eval.json', '--device', 'cpu', '--tile-size', 0, '--tile-overlap', 0)
    report = json.loads((tmp_path / 'eval.json').read_text())
    assert report['window'] == 3 and report['temporal_overlap'] == 2
    assert all(domain['frames'] == 7 for domain in report['domains'].values())
    inputs = tmp_path / 'dvd/val/blur/val_scene'
    run('infer_rtf_t6.py', '--config', config_path, '--checkpoint', checkpoint,
        '--input', inputs, '--output', tmp_path / 'inferred', '--device', 'cpu',
        '--tile-size', 0, '--tile-overlap', 0)
    assert sorted(p.name for p in (tmp_path / 'inferred').glob('*.png')) == sorted(p.name for p in inputs.glob('*.png'))
    assert all(Image.open(p).size == (32, 24) for p in (tmp_path / 'inferred').glob('*.png'))
    run('train_rtf_t6.py', '--config', config_path, '--pretrained', pretrained,
        '--output', tmp_path / 'benchmark', '--benchmark-updates', 2, '--benchmark-warmup-updates', 1)
    timing = json.loads((tmp_path / 'benchmark/benchmark.json').read_text())
    assert timing['status'] == 'BENCHMARK_PASS' and timing['frames_per_update'] == 12
    assert timing['measured_updates'] == 1 and timing['mean_seconds_per_update'] > 0
    assert timing['weights_discarded'] and not (tmp_path / 'benchmark/checkpoints').exists()
