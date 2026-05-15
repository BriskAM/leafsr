from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_data_paths(config: dict[str, Any], data_root: str | Path | None = None) -> dict[str, Path]:
    data_cfg = config["data"]
    root = Path(data_root or data_cfg["root"])
    return {
        "root": root,
        "train_lr": root / data_cfg["train_lr"],
        "train_hr": root / data_cfg["train_hr"],
        "test_lr": root / data_cfg["test_lr"],
        "sample_submission": root / data_cfg["sample_submission"],
        "vgg_weights": root / data_cfg["vgg_weights"],
    }


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
