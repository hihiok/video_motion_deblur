"""Strict paired-video discovery and balanced multi-domain clip sampling."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BLUR_NAMES = ("blur", "blur_gamma", "input", "lq", "blurry")
GT_NAMES = ("gt", "sharp", "target", "hq", "clear")
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation", "test"),
    "test": ("test", "val", "validation"),
}


@dataclass(frozen=True)
class PairedSequence:
    domain: str
    split: str
    name: str
    blur: tuple[Path, ...]
    gt: tuple[Path, ...]

    @property
    def length(self) -> int:
        return len(self.blur)


def _images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _pair_frames(blur_dir: Path, gt_dir: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    blur = _images(blur_dir)
    gt = _images(gt_dir)
    blur_by_stem = {path.stem: path for path in blur}
    gt_by_stem = {path.stem: path for path in gt}
    shared = sorted(set(blur_by_stem) & set(gt_by_stem))
    if len(shared) != len(blur) or len(shared) != len(gt):
        only_blur = sorted(set(blur_by_stem) - set(gt_by_stem))[:8]
        only_gt = sorted(set(gt_by_stem) - set(blur_by_stem))[:8]
        raise RuntimeError(
            f"Frame pairing mismatch: blur={blur_dir}, gt={gt_dir}, "
            f"blur_count={len(blur)}, gt_count={len(gt)}, shared={len(shared)}, "
            f"only_blur={only_blur}, only_gt={only_gt}"
        )
    return (
        tuple(blur_by_stem[stem] for stem in shared),
        tuple(gt_by_stem[stem] for stem in shared),
    )


def _named_child(parent: Path, names: Sequence[str]) -> Path | None:
    if not parent.is_dir():
        return None
    children = {path.name.lower(): path for path in parent.iterdir() if path.is_dir()}
    return next((children[name] for name in names if name in children), None)


def _split_roots(root: Path, split: str) -> list[Path]:
    aliases = SPLIT_ALIASES.get(split, (split,))
    candidates = [root / alias for alias in aliases if (root / alias).is_dir()]
    if candidates:
        return candidates
    known = {name for values in SPLIT_ALIASES.values() for name in values}
    if any((root / name).is_dir() for name in known):
        return []
    return [root]


def discover_sequences(
    domain: str,
    roots: str | Path | Sequence[str | Path],
    split: str,
    min_frames: int = 6,
) -> tuple[Path, list[PairedSequence]]:
    """Discover common GoPro/DVD/BSD layouts without fuzzy frame pairing."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    diagnostics = []
    for candidate in roots:
        root = Path(candidate).expanduser()
        if not root.is_dir():
            diagnostics.append(f"missing:{root}")
            continue
        found: list[PairedSequence] = []
        try:
            for split_root in _split_roots(root, split):
                # Layout A: split/{blur,gt}/sequence/frames
                blur_root = _named_child(split_root, BLUR_NAMES)
                gt_root = _named_child(split_root, GT_NAMES)
                if blur_root and gt_root:
                    blur_sequences = {
                        p.name: p for p in blur_root.iterdir() if p.is_dir()
                    }
                    gt_sequences = {p.name: p for p in gt_root.iterdir() if p.is_dir()}
                    names = sorted(set(blur_sequences) & set(gt_sequences))
                    if not names and _images(blur_root) and _images(gt_root):
                        names = [split_root.name]
                        blur_sequences[names[0]] = blur_root
                        gt_sequences[names[0]] = gt_root
                    for name in names:
                        blur, gt = _pair_frames(blur_sequences[name], gt_sequences[name])
                        if len(blur) >= min_frames:
                            found.append(PairedSequence(domain, split, name, blur, gt))

                # Layout B: split/sequence/{blur,gt}/frames
                for sequence_root in sorted(p for p in split_root.iterdir() if p.is_dir()):
                    blur_dir = _named_child(sequence_root, BLUR_NAMES)
                    gt_dir = _named_child(sequence_root, GT_NAMES)
                    if blur_dir and gt_dir:
                        blur, gt = _pair_frames(blur_dir, gt_dir)
                        if len(blur) >= min_frames:
                            found.append(
                                PairedSequence(domain, split, sequence_root.name, blur, gt)
                            )

                # Layout C: {blur,gt}/split/sequence/frames
                if split_root == root:
                    outer_blur = _named_child(root, BLUR_NAMES)
                    outer_gt = _named_child(root, GT_NAMES)
                    if outer_blur and outer_gt:
                        for alias in SPLIT_ALIASES.get(split, (split,)):
                            br = outer_blur / alias
                            gr = outer_gt / alias
                            if not br.is_dir() or not gr.is_dir():
                                continue
                            bseq = {p.name: p for p in br.iterdir() if p.is_dir()}
                            gseq = {p.name: p for p in gr.iterdir() if p.is_dir()}
                            for name in sorted(set(bseq) & set(gseq)):
                                blur, gt = _pair_frames(bseq[name], gseq[name])
                                if len(blur) >= min_frames:
                                    found.append(PairedSequence(domain, split, name, blur, gt))
        except RuntimeError:
            raise
        unique = {(seq.name, seq.blur[0].parent, seq.gt[0].parent): seq for seq in found}
        sequences = sorted(unique.values(), key=lambda seq: (seq.name, str(seq.blur[0])))
        if sequences:
            return root.resolve(), sequences
        diagnostics.append(f"no-valid-sequences:{root}")
    raise FileNotFoundError(
        f"Could not discover {domain}/{split} with >= {min_frames} paired frames. "
        + "; ".join(diagnostics)
    )


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _pad_to_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    pad_h = max(size - h, 0)
    pad_w = max(size - w, 0)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x


def load_clip(
    sequence: PairedSequence,
    indices: Sequence[int],
    crop_size: int,
    rng: random.Random,
    augment: bool,
    center_crop: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    blur = torch.stack([read_rgb(sequence.blur[index]) for index in indices])
    gt = torch.stack([read_rgb(sequence.gt[index]) for index in indices])
    if blur.shape != gt.shape:
        raise RuntimeError(
            f"Shape mismatch in {sequence.domain}/{sequence.name}: {blur.shape} vs {gt.shape}"
        )
    if crop_size > 0:
        blur = _pad_to_crop(blur, crop_size)
        gt = _pad_to_crop(gt, crop_size)
        h, w = blur.shape[-2:]
        if center_crop:
            top, left = (h - crop_size) // 2, (w - crop_size) // 2
        else:
            top = rng.randint(0, h - crop_size)
            left = rng.randint(0, w - crop_size)
        blur = blur[..., top : top + crop_size, left : left + crop_size]
        gt = gt[..., top : top + crop_size, left : left + crop_size]
    if augment:
        if rng.random() < 0.5:
            blur, gt = blur.flip(-1), gt.flip(-1)
        if rng.random() < 0.5:
            blur, gt = blur.flip(-2), gt.flip(-2)
        rotation = rng.randrange(4)
        if rotation:
            blur, gt = torch.rot90(blur, rotation, (-2, -1)), torch.rot90(gt, rotation, (-2, -1))
        if rng.random() < 0.5:
            blur, gt = blur.flip(0), gt.flip(0)
    return blur.contiguous(), gt.contiguous()


class BalancedMultiDomainClips(Dataset):
    """Fixed-length virtual epoch with deterministic balanced domain sampling."""

    def __init__(
        self,
        domains: dict[str, list[PairedSequence]],
        clip_length: int,
        crop_size: int,
        samples_per_epoch: int,
        domain_weights: dict[str, float] | None = None,
        seed: int = 123,
        augment: bool = True,
    ):
        self.domains = domains
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.augment = augment
        names = sorted(domains)
        if not names or any(not domains[name] for name in names):
            raise ValueError("Every configured domain must contain at least one sequence")
        weights = domain_weights or {name: 1.0 for name in names}
        self.names = names
        values = torch.tensor([float(weights.get(name, 0.0)) for name in names])
        if (values < 0).any() or values.sum() <= 0:
            raise ValueError(f"Invalid domain weights: {weights}")
        self.weights = (values / values.sum()).tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(index) * 9_176)
        domain = rng.choices(self.names, weights=self.weights, k=1)[0]
        sequence = rng.choice(self.domains[domain])
        maximum = sequence.length - self.clip_length
        if maximum < 0:
            raise RuntimeError(f"Sequence shorter than clip: {sequence.name}")
        start = rng.randint(0, maximum)
        indices = list(range(start, start + self.clip_length))
        blur, gt = load_clip(sequence, indices, self.crop_size, rng, self.augment)
        return {
            "blur": blur,
            "gt": gt,
            "domain": domain,
            "sequence": sequence.name,
            "start": start,
        }


class ValidationClips(Dataset):
    def __init__(
        self,
        domains: dict[str, list[PairedSequence]],
        clip_length: int,
        crop_size: int = 256,
        stride: int | None = None,
        max_clips_per_domain: int = 24,
        seed: int = 321,
    ):
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.seed = seed
        stride = stride or clip_length
        items = []
        for domain, sequences in sorted(domains.items()):
            domain_items = []
            for sequence in sequences:
                starts = list(range(0, sequence.length - clip_length + 1, stride))
                last = sequence.length - clip_length
                if starts and starts[-1] != last:
                    starts.append(last)
                domain_items.extend((sequence, start) for start in starts)
            if max_clips_per_domain > 0 and len(domain_items) > max_clips_per_domain:
                selector = random.Random(seed + sum(map(ord, domain)))
                selector.shuffle(domain_items)
                domain_items = domain_items[:max_clips_per_domain]
            items.extend(domain_items)
        self.items = sorted(items, key=lambda item: (item[0].domain, item[0].name, item[1]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence, start = self.items[index]
        indices = list(range(start, start + self.clip_length))
        blur, gt = load_clip(
            sequence,
            indices,
            self.crop_size,
            random.Random(self.seed + index),
            augment=False,
            center_crop=True,
        )
        return {
            "blur": blur,
            "gt": gt,
            "domain": sequence.domain,
            "sequence": sequence.name,
            "start": start,
        }


def build_domain_sequences(
    dataset_config: dict,
    split: str,
    clip_length: int,
) -> tuple[dict[str, list[PairedSequence]], dict[str, str]]:
    domains = {}
    roots = {}
    for domain, config in dataset_config.items():
        configured_roots = config.get("roots", config.get("root"))
        if not configured_roots:
            raise ValueError(f"No root(s) configured for {domain}")
        root, sequences = discover_sequences(domain, configured_roots, split, clip_length)
        domains[domain] = sequences
        roots[domain] = str(root)
    return domains, roots


def dataset_summary(domains: dict[str, Iterable[PairedSequence]]) -> dict[str, dict[str, int]]:
    return {
        domain: {
            "sequences": len(list(sequences)),
            "frames": sum(sequence.length for sequence in sequences),
        }
        for domain, sequences in domains.items()
    }
