from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from leafsr.utils import image_mae, load_rgb, to_tensor, to_uint8


def evaluate_generator(
    generator: nn.Module,
    lr_paths: list[Path],
    hr_dir: str | Path,
    device: torch.device,
    scale: int = 4,
    use_amp: bool = False,
) -> float:
    generator.eval()
    scores = []
    hr_dir = Path(hr_dir)
    with torch.no_grad():
        for lr_path in lr_paths:
            hr_path = hr_dir / lr_path.name
            lr_np = load_rgb(lr_path)
            height, width = lr_np.shape[:2]
            bicubic_np = np.asarray(
                Image.open(lr_path).convert("RGB").resize(
                    (width * scale, height * scale),
                    Image.Resampling.BICUBIC,
                ),
                dtype=np.float32,
            ) / 255.0
            lr = to_tensor(lr_np).unsqueeze(0).to(device)
            bicubic = to_tensor(bicubic_np).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = generator(lr, bicubic).squeeze(0)
            pred_u8 = to_uint8(pred)
            target_u8 = np.asarray(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
            scores.append(image_mae(pred_u8, target_u8))
    return float(np.mean(scores))


def bicubic_score(lr_paths: list[Path], hr_dir: str | Path, scale: int = 4) -> float:
    scores = []
    hr_dir = Path(hr_dir)
    for lr_path in lr_paths:
        lr_image = Image.open(lr_path).convert("RGB")
        pred = lr_image.resize((lr_image.width * scale, lr_image.height * scale), Image.Resampling.BICUBIC)
        target = Image.open(hr_dir / lr_path.name).convert("RGB")
        scores.append(image_mae(np.asarray(pred, dtype=np.uint8), np.asarray(target, dtype=np.uint8)))
    return float(np.mean(scores))
