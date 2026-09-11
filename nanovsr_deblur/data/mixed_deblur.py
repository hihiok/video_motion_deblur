import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
import torchvision.transforms.functional as TF

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
_BLUR_NAMES = ('blur', 'Blur', 'blurry', 'input')
_GT_NAMES = ('sharp', 'Sharp', 'gt', 'GT', 'target', 'label')


def _natural_key(path_or_name):
    s = str(path_or_name)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', s)]


def _image_files(path: Path):
    if not path.exists():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS],
        key=_natural_key,
    )


def _candidate_bases(root: Path, split: str, root_split_only=False):
    """Return candidate split directories below a dataset root.

    When root_split_only=True, only the exact direct split root is allowed:
      root/train or root/test
    This mode is used for BSD in the current NanoVNR NAFNet RGB experiment so
    directories such as root/<config>/train are intentionally excluded.
    """
    if root_split_only:
        if split not in ('train', 'test'):
            raise ValueError(
                f'root_split_only supports only train/test, got split={split!r}'
            )
        direct = root / split
        return [direct] if direct.is_dir() else []

    aliases = [split]
    if split == 'train':
        aliases += ['training']
    elif split in ('test', 'val'):
        aliases += ['testing' if split == 'test' else 'validation']

    bases = []
    for a in aliases:
        p = root / a
        if p.is_dir():
            bases.append(p)

    # Generic datasets may have configuration folders above train/test.
    # This scan is disabled for BSD by root_split_only=True.
    for child in (
        sorted([p for p in root.iterdir() if p.is_dir()], key=_natural_key)
        if root.is_dir()
        else []
    ):
        for a in aliases:
            p = child / a
            if p.is_dir():
                bases.append(p)

    # Only fall back to the root itself when there is no explicit split folder.
    explicit_split_present = any(
        (root / x).is_dir()
        for x in ('train', 'training', 'test', 'testing', 'val', 'validation')
    )
    if not explicit_split_present:
        bases.append(root)
        for child in (
            sorted([p for p in root.iterdir() if p.is_dir()], key=_natural_key)
            if root.is_dir()
            else []
        ):
            child_has_split = any(
                (child / x).is_dir()
                for x in ('train', 'training', 'test', 'testing', 'val', 'validation')
            )
            if not child_has_split:
                bases.append(child)

    out, seen = [], set()
    for p in bases:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _add_pair(pairs, seen, blur, gt):
    if not (blur.is_dir() and gt.is_dir()):
        return
    key = (blur.resolve(), gt.resolve())
    if key not in seen:
        seen.add(key)
        pairs.append((blur, gt))


def discover_pair_roots(root, split='train', root_split_only=False):
    """Discover one or more frame-aligned blur/GT directory pairs.

    Generic supported examples include:
      root/train/blur + root/train/gt
      root/blur + root/gt
      root/<config>/train/blur + root/<config>/gt
      root/<config>/train/<seq>/Blur/RGB + .../Sharp/RGB
      input/target and blur/sharp naming variants

    For the current BSD experiment, pass root_split_only=True. Then only
    root/train or root/test is searched, and nested configuration split roots
    outside those two directories are never included.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')

    pairs, seen = [], set()
    for base in _candidate_bases(root, split, root_split_only=root_split_only):
        # Standard layout: base/{blur,gt}/<sequence>/frames.
        for blur_name in _BLUR_NAMES:
            blur = base / blur_name
            if not blur.is_dir():
                continue
            for gt_name in _GT_NAMES:
                gt = base / gt_name
                if gt.is_dir():
                    _add_pair(pairs, seen, blur, gt)
                    break

        # Sequence-local layout:
        # base/<sequence>/Blur/RGB/*.png and base/<sequence>/Sharp/RGB/*.png
        for seq in (
            sorted([p for p in base.iterdir() if p.is_dir()], key=_natural_key)
            if base.is_dir()
            else []
        ):
            for blur_name in ('Blur', 'blur'):
                blur_container = seq / blur_name
                if not blur_container.is_dir():
                    continue
                blur = (
                    blur_container / 'RGB'
                    if (blur_container / 'RGB').is_dir()
                    else blur_container
                )
                for gt_name in ('Sharp', 'sharp', 'GT', 'gt'):
                    gt_container = seq / gt_name
                    if not gt_container.is_dir():
                        continue
                    gt = (
                        gt_container / 'RGB'
                        if (gt_container / 'RGB').is_dir()
                        else gt_container
                    )
                    if _image_files(blur) and _image_files(gt):
                        _add_pair(pairs, seen, blur, gt)
                        break

    return pairs


def _sequence_dirs(pair_root: Path):
    dirs = sorted([p for p in pair_root.iterdir() if p.is_dir()], key=_natural_key)
    return dirs if dirs else [pair_root]


def _match_frame_pairs(blur_dir: Path, gt_dir: Path):
    blur_files = _image_files(blur_dir)
    gt_files = _image_files(gt_dir)
    if not blur_files or not gt_files:
        return []

    gt_by_name = {p.name: p for p in gt_files}
    exact = [(b, gt_by_name[b.name]) for b in blur_files if b.name in gt_by_name]
    if len(exact) == len(blur_files) == len(gt_files):
        return exact

    raise RuntimeError(
        f'Frame-name mismatch: blur={blur_dir} ({len(blur_files)} files), '
        f'gt={gt_dir} ({len(gt_files)} files), exact_name_pairs={len(exact)}. '
        'Refusing index-based pairing because it could train on misaligned targets.'
    )


class VideoPairWindowDataset(Dataset):
    def __init__(
        self,
        family,
        blur_root,
        gt_root,
        num_frames,
        patch_size=None,
        train=True,
        stride=1,
    ):
        self.family = str(family)
        self.blur_root = Path(blur_root)
        self.gt_root = Path(gt_root)
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size) if patch_size else None
        self.train = bool(train)
        self.stride = int(stride)
        self.samples = []
        self.sequence_count = 0

        need = (self.num_frames - 1) * self.stride + 1
        for blur_seq in _sequence_dirs(self.blur_root):
            seq_name = blur_seq.name if blur_seq != self.blur_root else '__root__'
            gt_seq = self.gt_root / seq_name if seq_name != '__root__' else self.gt_root
            if not gt_seq.is_dir():
                raise RuntimeError(
                    f'Missing matching GT sequence for {blur_seq}: expected {gt_seq}'
                )
            pairs = _match_frame_pairs(blur_seq, gt_seq)
            if len(pairs) < need:
                continue
            self.sequence_count += 1
            for start in range(0, len(pairs) - need + 1):
                self.samples.append((seq_name, start, pairs))

        if not self.samples:
            raise RuntimeError(
                f'No {self.num_frames}-frame windows found for {self.family}: '
                f'{self.blur_root} -> {self.gt_root}'
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load(path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        seq, start, pairs = self.samples[idx]
        ids = [start + i * self.stride for i in range(self.num_frames)]
        blur = [self._load(pairs[i][0]) for i in ids]
        gt = [self._load(pairs[i][1]) for i in ids]

        if self.train and self.patch_size:
            w, h = blur[0].size
            ps = self.patch_size
            if w < ps or h < ps:
                raise RuntimeError(
                    f'Patch {ps} exceeds frame {w}x{h} in {self.family}/{seq}. '
                    'Choose a smaller global patch size; images are never resized silently.'
                )
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            box = (x, y, x + ps, y + ps)
            blur = [im.crop(box) for im in blur]
            gt = [im.crop(box) for im in gt]

            if random.random() < 0.5:
                blur = [TF.hflip(im) for im in blur]
                gt = [TF.hflip(im) for im in gt]
            if random.random() < 0.5:
                blur = [TF.vflip(im) for im in blur]
                gt = [TF.vflip(im) for im in gt]
            if random.random() < 0.5:
                blur.reverse()
                gt.reverse()
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                blur = [TF.rotate(im, angle) for im in blur]
                gt = [TF.rotate(im, angle) for im in gt]

        return {
            'blur': torch.stack([TF.to_tensor(im) for im in blur]),
            'sharp': torch.stack([TF.to_tensor(im) for im in gt]),
            'source': self.family,
            'seq': seq,
        }


def build_mixed_dataset(source_roots, split, num_frames, patch_size=None, train=True):
    """Build a family-balanced GoPro/DVD/BSD mixture.

    BSD has a strict source policy for this experiment: only
    <BSD_ROOT>/train and <BSD_ROOT>/test are eligible. No sibling/nested
    configuration directory outside those direct split roots can be sampled.
    """
    components = []
    family_lengths = defaultdict(int)
    audit = []

    for family, root in source_roots.items():
        is_bsd = str(family).strip().lower() == 'bsd'
        pairs = discover_pair_roots(
            root,
            split=split,
            root_split_only=is_bsd,
        )
        if not pairs:
            policy = ' (strict direct root split only)' if is_bsd else ''
            raise RuntimeError(
                f'No {split} blur/GT pair roots discovered for {family}: {root}{policy}'
            )

        allowed_bsd_base = (Path(root) / split).resolve() if is_bsd else None
        for blur_root, gt_root in pairs:
            if is_bsd:
                for p in (Path(blur_root).resolve(), Path(gt_root).resolve()):
                    try:
                        p.relative_to(allowed_bsd_base)
                    except ValueError as exc:
                        raise RuntimeError(
                            f'BSD path escaped allowed split root {allowed_bsd_base}: {p}'
                        ) from exc

            ds = VideoPairWindowDataset(
                family=family,
                blur_root=blur_root,
                gt_root=gt_root,
                num_frames=num_frames,
                patch_size=patch_size,
                train=train,
            )
            components.append(ds)
            family_lengths[family] += len(ds)
            audit.append(
                {
                    'family': family,
                    'blur_root': str(blur_root),
                    'gt_root': str(gt_root),
                    'sequences': ds.sequence_count,
                    'windows': len(ds),
                    'frames_per_window': num_frames,
                    'strict_root_split': bool(is_bsd),
                }
            )

    concat = ConcatDataset(components)
    sampler = None
    if train:
        weights = []
        for ds in components:
            per_sample = 1.0 / max(1, family_lengths[ds.family])
            weights.extend([per_sample] * len(ds))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(concat),
            replacement=True,
        )

    return concat, sampler, audit
