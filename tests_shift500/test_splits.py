"""Regression cases for official GoPro chunk splits, without GPU or weights."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import pytest
from shift500.data import DOMAINS, discover, make_manifest


def write_clip(root, split, name, value):
    for kind in ('blur', 'gt'):
        folder = root / split / name / kind
        folder.mkdir(parents=True)
        for i in range(2):
            array = np.full((4, 5, 3), value + i, dtype=np.uint8)
            Image.fromarray(array).save(folder / f'{i:04d}.png')


@pytest.fixture
def roots(tmp_path):
    result = {}
    for di, domain in enumerate(DOMAINS):
        root = tmp_path / domain
        result[domain] = str(root)
        if domain == 'gopro':
            train = ['GOPR0384_11_00', 'GOPR0384_11_01', 'GOPR0385_11_00', 'GOPR0868_11_00']
            test = ['GOPR0384_11_02']
        else:
            train, test = ['scene0', 'scene1', 'scene2'], ['scene3']
        for i, name in enumerate(train):
            write_clip(root, 'train', name, 10 + di * 60 + i * 4)
        for name in test:
            write_clip(root, 'test', name, 50 + di * 60)
    return result


def test_official_acquisition_overlap_is_audited_without_repartition(roots):
    manifest = make_manifest(roots, frames=2)
    assert manifest['version'] == 2
    audit = manifest['split_audit']['gopro']
    assert audit['official_train_test_shared_acquisitions'] == ['GOPR0384_11']
    assert audit['official_train_clips'] == 4
    assert audit['official_test_clips'] == 1
    for domain in DOMAINS:
        expected_train = {r['name'] for r in discover(roots[domain], domain, 'train')}
        actual_train = {r['name'] for s in ('train', 'val') for r in manifest[s] if r['domain'] == domain}
        assert actual_train == expected_train
        assert [r['name'] for r in manifest['test'] if r['domain'] == domain] == [
            r['name'] for r in discover(roots[domain], domain, 'test')]
        train_groups = {r['group'] for r in manifest['train'] if r['domain'] == domain}
        val_groups = {r['group'] for r in manifest['val'] if r['domain'] == domain}
        assert train_groups.isdisjoint(val_groups)
    assert manifest['split_audit']['gt_file_sha256_cross_split_check'] == 'passed'


def test_same_complete_clip_in_train_test_still_rejected(roots):
    write_clip(Path(roots['gopro']), 'test', 'GOPR0384_11_00', 240)
    with pytest.raises(ValueError, match='complete clip names overlap'):
        make_manifest(roots, frames=2)


def test_shared_acquisition_does_not_exempt_duplicate_gt_frames(roots):
    root = Path(roots['gopro'])
    shutil.copyfile(root / 'train/GOPR0384_11_00/gt/0000.png', root / 'test/GOPR0384_11_02/gt/0000.png')
    with pytest.raises(ValueError, match='GT content leakage'):
        make_manifest(roots, frames=2)


def test_internal_train_val_duplicate_frames_still_rejected(roots):
    manifest = make_manifest(roots, frames=2)
    tr = next(r for r in manifest['train'] if r['domain'] == 'gopro')
    val = next(r for r in manifest['val'] if r['domain'] == 'gopro')
    shutil.copyfile(tr['gt'][0], val['gt'][0])
    with pytest.raises(ValueError, match='GT content leakage'):
        make_manifest(roots, frames=2)
