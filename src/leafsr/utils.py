from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, _ = torch.cuda.get_device_capability(0)
    return torch.device("cuda" if major >= 7 else "cpu")


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def image_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))


def bicubic_resize(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image(array: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
