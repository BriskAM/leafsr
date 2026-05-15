from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def init_ema(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def update_ema(model: nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        model_state = model.state_dict()
        for key, value in model_state.items():
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def load_state_copy(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.clone() for key, value in state.items()})


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module | None,
    val_mae: float,
    epoch: int,
    config: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "G": generator.state_dict(),
        "val_mae": val_mae,
        "epoch": epoch,
        "config": config,
    }
    if discriminator is not None:
        payload["D"] = discriminator.state_dict()
    torch.save(payload, path)
