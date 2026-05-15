from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from leafsr.utils import load_rgb, to_tensor, to_uint8


def make_bicubic_tensor(lr_path: str | Path, scale: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lr_np = load_rgb(lr_path)
    height, width = lr_np.shape[:2]
    bicubic_np = np.asarray(
        Image.open(lr_path).convert("RGB").resize((width * scale, height * scale), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    lr = to_tensor(lr_np).unsqueeze(0).to(device)
    bicubic = to_tensor(bicubic_np).unsqueeze(0).to(device)
    return lr, bicubic


def infer_with_tta(
    generator: nn.Module,
    lr_tensor: torch.Tensor,
    bicubic_tensor: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    variants = [
        (lr_tensor, bicubic_tensor, lambda x: x),
        (torch.flip(lr_tensor, dims=[3]), torch.flip(bicubic_tensor, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (torch.flip(lr_tensor, dims=[2]), torch.flip(bicubic_tensor, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (
            torch.flip(lr_tensor, dims=[2, 3]),
            torch.flip(bicubic_tensor, dims=[2, 3]),
            lambda x: torch.flip(x, dims=[2, 3]),
        ),
        (
            torch.rot90(lr_tensor, 1, dims=[2, 3]),
            torch.rot90(bicubic_tensor, 1, dims=[2, 3]),
            lambda x: torch.rot90(x, -1, dims=[2, 3]),
        ),
        (
            torch.rot90(lr_tensor, 2, dims=[2, 3]),
            torch.rot90(bicubic_tensor, 2, dims=[2, 3]),
            lambda x: torch.rot90(x, -2, dims=[2, 3]),
        ),
        (
            torch.rot90(lr_tensor, 3, dims=[2, 3]),
            torch.rot90(bicubic_tensor, 3, dims=[2, 3]),
            lambda x: torch.rot90(x, -3, dims=[2, 3]),
        ),
        (
            torch.flip(torch.rot90(lr_tensor, 1, dims=[2, 3]), dims=[3]),
            torch.flip(torch.rot90(bicubic_tensor, 1, dims=[2, 3]), dims=[3]),
            lambda x: torch.rot90(torch.flip(x, dims=[3]), -1, dims=[2, 3]),
        ),
    ]
    preds = []
    generator.eval()
    with torch.no_grad():
        for lr_inp, bicubic_inp, inverse in variants:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds.append(inverse(generator(lr_inp, bicubic_inp)))
    return torch.stack(preds, dim=0).mean(dim=0)


def tensor_to_image(pred: torch.Tensor) -> np.ndarray:
    return to_uint8(pred.squeeze(0))
