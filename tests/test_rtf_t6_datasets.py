from pathlib import Path

import numpy as np
from PIL import Image

from rtf_t6.datasets import BalancedMultiDomainClips, discover_sequences


def make_sequence(root: Path, split: str, name: str, count: int = 7):
    for kind, offset in (("blur", 0), ("gt", 7)):
        directory = root / split / kind / name
        directory.mkdir(parents=True)
        for index in range(count):
            array = np.full((20, 24, 3), index + offset, dtype=np.uint8)
            Image.fromarray(array).save(directory / f"{index:06d}.png")


def test_discovers_root_split_kind_sequence_layout(tmp_path: Path):
    make_sequence(tmp_path, "train", "scene_a")
    make_sequence(tmp_path, "test", "scene_b")
    root, train = discover_sequences("demo", tmp_path, "train", min_frames=6)
    _, val = discover_sequences("demo", tmp_path, "val", min_frames=6)
    assert root == tmp_path.resolve()
    assert [sequence.name for sequence in train] == ["scene_a"]
    assert [sequence.name for sequence in val] == ["scene_b"]


def test_balanced_clip_shape_is_t6(tmp_path: Path):
    make_sequence(tmp_path, "train", "scene_a")
    _, sequences = discover_sequences("demo", tmp_path, "train", min_frames=6)
    dataset = BalancedMultiDomainClips(
        {"demo": sequences}, clip_length=6, crop_size=16, samples_per_epoch=2
    )
    item = dataset[0]
    assert item["blur"].shape == (6, 3, 16, 16)
    assert item["gt"].shape == (6, 3, 16, 16)
