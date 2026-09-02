import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _find_pair_roots(root: Path, split: str):
    candidates = [
        (root / split / 'blur', root / split / 'sharp'),
        (root / split / 'blur', root / split / 'GT'),
        (root / split / 'blur', root / split / 'gt'),
        (root / split / 'blur_gamma', root / split / 'sharp'),
        (root / split / 'input', root / split / 'target'),
    ]
    for blur, sharp in candidates:
        if blur.exists() and sharp.exists():
            return blur, sharp
    raise FileNotFoundError(f'Cannot find GoPro pair roots below {root}/{split}')


class GoProVideoDataset(Dataset):
    def __init__(self, root, split='train', num_frames=7, patch_size=256, stride=1, teacher_root=None):
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.stride = stride
        self.blur_root, self.sharp_root = _find_pair_roots(self.root, split)
        self.teacher_root = Path(teacher_root) if teacher_root else None
        self.samples = []

        for seq in sorted([p for p in self.blur_root.iterdir() if p.is_dir()]):
            frames = sorted(list(seq.glob('*.png')) + list(seq.glob('*.jpg')))
            need = (num_frames - 1) * stride + 1
            for start in range(max(0, len(frames) - need + 1)):
                self.samples.append((seq.name, start, [p.name for p in frames]))
        if not self.samples:
            raise RuntimeError(f'No samples found under {self.blur_root}')

    def __len__(self):
        return len(self.samples)

    def _load(self, p):
        return Image.open(p).convert('RGB')

    def __getitem__(self, idx):
        seq, start, names = self.samples[idx]
        ids = [start + i * self.stride for i in range(self.num_frames)]
        blur = [self._load(self.blur_root / seq / names[i]) for i in ids]
        sharp = [self._load(self.sharp_root / seq / names[i]) for i in ids]
        teacher = None
        if self.teacher_root is not None:
            teacher = [self._load(self.teacher_root / self.split / seq / names[i]) for i in ids]

        if self.split == 'train' and self.patch_size:
            w, h = blur[0].size
            ps = min(self.patch_size, w, h)
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            box = (x, y, x + ps, y + ps)
            blur = [im.crop(box) for im in blur]
            sharp = [im.crop(box) for im in sharp]
            if teacher is not None:
                teacher = [im.crop(box) for im in teacher]
            if random.random() < 0.5:
                blur = [TF.hflip(im) for im in blur]
                sharp = [TF.hflip(im) for im in sharp]
                if teacher is not None: teacher = [TF.hflip(im) for im in teacher]
            if random.random() < 0.5:
                blur = [TF.vflip(im) for im in blur]
                sharp = [TF.vflip(im) for im in sharp]
                if teacher is not None: teacher = [TF.vflip(im) for im in teacher]
            if random.random() < 0.5:
                blur.reverse(); sharp.reverse()
                if teacher is not None: teacher.reverse()

        sample = {
            'blur': torch.stack([TF.to_tensor(x) for x in blur]),
            'sharp': torch.stack([TF.to_tensor(x) for x in sharp]),
            'seq': seq,
        }
        if teacher is not None:
            sample['teacher'] = torch.stack([TF.to_tensor(x) for x in teacher])
        return sample
