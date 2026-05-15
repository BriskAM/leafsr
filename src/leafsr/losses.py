from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class VGGPerceptualLoss(nn.Module):
    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"VGG weights not found at {weights_path}. "
                "Download/provide vgg19_weights.pth or disable perceptual loss."
            )
        vgg = models.vgg19(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu")
        vgg.load_state_dict(state_dict)
        self.features = nn.Sequential(*list(vgg.features.children())[:18]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.features(pred_n), self.features(target_n))
