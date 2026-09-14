"""Small CPU integration tests; no training, datasets or real weights required."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

from rtf_temporal import posttrain as p
from rtf_temporal.model import TemporalRTFocuser


def frames(directory, count=3, offset=0, shape=(32, 48)):
    directory.mkdir(parents=True)
    rng = np.random.default_rng(12 + offset)
    for i in range(count):
        Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8)).save(directory / f'{i:08d}.png')
    return directory


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    root = tmp_path_factory.mktemp('models')
    official = p.RT_Focuser_Standard().eval()
    torch.save(official.state_dict(), root / 'official.pth')
    temporal = TemporalRTFocuser(activation_checkpointing=False).eval()
    temporal.load_official(root / 'official.pth')
    # Nonzero adapter exercises actual recurrent inference rather than identity only.
    torch.nn.init.normal_(temporal.temporal.project.weight, std=.01)
    torch.save({'model': temporal.state_dict(), 'config': {'model': {'hidden': 32}, 'train': {'cut_threshold': 1.0}}}, root / 'best.pth')
    audit = {'checkpoints': {'official': {'snapshot': str(root / 'official.pth')}, 'best_stable': {'snapshot': str(root / 'best.pth')}}}
    return p.Models(audit, 'cpu'), audit


def test_identity_native_padding_and_metrics(models):
    model, audit = models
    x = torch.rand(1, 3, 33, 49)
    assert model.verify_wrapper(x, audit)['max_abs_error'] <= 1e-5
    assert model.original(x).shape == x.shape
    metric = p.image_metrics(torch.full((3, 16, 16), .1), torch.zeros(3, 16, 16))
    assert metric['psnr'] == pytest.approx(20, abs=1e-5)


def test_variants_full_evaluation_and_bad_pair(tmp_path, models):
    for v in ('sharp', 'blur', 'blur_gamma'):
        frames(tmp_path / 'test' / 'GOPR0001_11_00' / v, offset=int(v != 'sharp'))
    available, absent = p.gopro_test(tmp_path, expected_frames=3, expected_sequences=1)
    assert set(available) == {'blur', 'blur_gamma'} and not absent
    out = tmp_path / 'results'; out.mkdir()
    result = p.full_gopro(models[0], available, out)
    assert result['variants']['blur']['official']['frames'] == 3
    assert result['variants']['blur']['best_stable']['aligned_temporal_l1'] is not None
    assert len((out / 'gopro_blur_per_frame.csv').read_text().splitlines()) == 10
    (tmp_path / 'test/GOPR0001_11_00/blur/00000001.png').unlink()
    with pytest.raises(ValueError, match='nonconsecutive'):
        p.gopro_test(tmp_path, expected_frames=3, expected_sequences=1)


def test_exact_overlap_not_acquisition_prefix(tmp_path):
    train = frames(tmp_path / 'train', count=1)
    test = frames(tmp_path / 'test', count=1, offset=1)
    manifest = {'train': [{'domain': 'gopro', 'name': 'GOPR0001_11_00', 'gt': [str(train / '00000000.png')]}]}
    records = [{'name': 'GOPR0001_11_01', 'gt': [test / '00000000.png']}]
    p.verify_holdout_no_exact_test_overlap(manifest, records)
    shutil.copyfile(train / '00000000.png', test / '00000000.png')
    with pytest.raises(ValueError, match='contents overlap'):
        p.verify_holdout_no_exact_test_overlap(manifest, records)


def test_audit_snapshots_real_best_update_and_hash(tmp_path, monkeypatch, models):
    run = tmp_path / 'training'; (run / 'checkpoints').mkdir(parents=True)
    manifest = {'train': [], 'val': []}
    p.write_json(run / 'manifest.json', manifest)
    payload = torch.load(models[1]['checkpoints']['best_stable']['snapshot'], weights_only=False)
    payload.update(update=17000, git_commit='training-commit', baseline={'psnr': 24.9})
    payload['config'].update(protocol='rtfocuser_causal_temporal_finetune_v1', manifest_sha256=p.sha256(run / 'manifest.json'))
    torch.save(payload, run / 'checkpoints/best_stable.pth')
    official = Path(models[1]['checkpoints']['official']['snapshot'])
    monkeypatch.setattr(p, 'OFFICIAL_SHA', p.sha256(official))
    audit, loaded = p.checkpoint_audit(run, official, tmp_path / 'audit')
    assert audit['best_update'] == 17000
    assert loaded == manifest
    snapshot = Path(audit['checkpoints']['best_stable']['snapshot'])
    assert snapshot.is_file() and p.sha256(snapshot) == p.sha256(run / 'checkpoints/best_stable.pth')
    p.write_json(run / 'manifest.json', {'train': [], 'val': [], 'changed': True})
    with pytest.raises(ValueError, match='Manifest differs'):
        p.checkpoint_audit(run, official, tmp_path / 'audit2')


@pytest.mark.parametrize('vfr', [False, True])
def test_encode_timestamps(tmp_path, vfr):
    directory = frames(tmp_path / 'frames', count=4)
    info = {'frames': 4, 'avg_frame_rate': '25', 'variable_frame_rate': vfr,
            'pts': [0, .04, .12, .16] if vfr else [0, .04, .08, .12],
            'durations': [.04, .08, .04, .06] if vfr else [.04] * 4}
    result = p.encode_frames(directory, tmp_path / 'encoded.mp4', info)
    assert result['frames'] == 4
    assert result['max_timestamp_error_seconds'] < .002
    assert result['duration_seconds'] == pytest.approx(sum(info['durations']), abs=.002)


def test_business_end_to_end(tmp_path, models):
    directory = frames(tmp_path / 'frames', count=3)
    info = {'frames': 3, 'avg_frame_rate': '25', 'variable_frame_rate': False,
            'pts': [0, .04, .08], 'durations': [.04] * 3}
    source = tmp_path / 'business input.mp4'
    p.encode_frames(directory, source, info)
    output = tmp_path / 'business'
    report = p.business_video(models[0], source, output)
    assert report['reset_indices'] == [0]
    for name in ('official', 'best_stable', 'comparison'):
        assert report['encoded'][name]['frames'] == 3
        assert (output / f'{name}.mp4').is_file()
    with Image.open(output / 'best_stable_frames/00000000.png') as image:
        assert image.size == (48, 32)
