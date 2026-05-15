from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from leafsr.utils import load_rgb, to_tensor


def list_pngs(path: str | Path) -> list[Path]:
    return sorted(Path(path).glob("*.png"))


def split_paths(paths: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    split_idx = int(len(paths) * (1.0 - val_fraction))
    return paths[:split_idx], paths[split_idx:]


class SuperResolutionPatchDataset(Dataset):
    def __init__(
        self,
        lr_paths: list[Path],
        hr_dir: str | Path,
        scale: int = 4,
        patch_size: int = 32,
        patches_per_image: int = 16,
    ) -> None:
        if not lr_paths:
            raise ValueError("lr_paths cannot be empty")
        self.lr_paths = list(lr_paths)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

    def __len__(self) -> int:
        return len(self.lr_paths) * self.patches_per_image

    def __getitem__(self, idx: int):
        lr_path = self.lr_paths[idx % len(self.lr_paths)]
        hr_path = self.hr_dir / lr_path.name
        lr = load_rgb(lr_path)
        hr = load_rgb(hr_path)

        height, width = lr.shape[:2]
        patch_size = min(self.patch_size, height, width)
        top = random.randint(0, height - patch_size)
        left = random.randint(0, width - patch_size)

        lr_patch = lr[top : top + patch_size, left : left + patch_size]
        hr_patch = hr[
            top * self.scale : (top + patch_size) * self.scale,
            left * self.scale : (left + patch_size) * self.scale,
        ]

        lr_patch, hr_patch = self._augment(lr_patch, hr_patch)
        bicubic = np.asarray(
            Image.fromarray(np.clip(np.round(lr_patch * 255.0), 0, 255).astype(np.uint8)).resize(
                (patch_size * self.scale, patch_size * self.scale),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
        ) / 255.0
        residual = hr_patch - bicubic

        return to_tensor(lr_patch), to_tensor(bicubic), to_tensor(hr_patch), to_tensor(residual)

    @staticmethod
    def _augment(lr_patch: np.ndarray, hr_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            lr_patch = np.flip(lr_patch, axis=1).copy()
            hr_patch = np.flip(hr_patch, axis=1).copy()
        if random.random() < 0.5:
            lr_patch = np.flip(lr_patch, axis=0).copy()
            hr_patch = np.flip(hr_patch, axis=0).copy()
        rotations = random.randint(0, 3)
        if rotations:
            lr_patch = np.rot90(lr_patch, rotations).copy()
            hr_patch = np.rot90(hr_patch, rotations).copy()
        return lr_patch, hr_patch
