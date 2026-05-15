#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leafsr.config import load_config
from leafsr.inference import infer_with_tta, make_bicubic_tensor, tensor_to_image
from leafsr.model import build_generator
from leafsr.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LR/bicubic/prediction/HR comparison grid.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--output", default="assets/comparison_grid.png")
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def add_label(image: Image.Image, label: str) -> Image.Image:
    labeled = Image.new("RGB", (image.width, image.height + 24), "white")
    labeled.paste(image, (0, 24))
    draw = ImageDraw.Draw(labeled)
    draw.text((6, 5), label, fill="black")
    return labeled


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator = build_generator(config, scale=config["scale"]).to(device)
    generator.load_state_dict(checkpoint["G"])
    generator.eval()

    lr_dir = Path(args.lr_dir)
    hr_dir = Path(args.hr_dir)
    rows = []
    for lr_path in sorted(lr_dir.glob("*.png"))[: args.limit]:
        hr_path = hr_dir / lr_path.name
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        bicubic_img = lr_img.resize(hr_img.size, Image.Resampling.BICUBIC)
        lr, bicubic = make_bicubic_tensor(lr_path, config["scale"], device)
        pred = infer_with_tta(generator, lr, bicubic, device, use_amp=device.type == "cuda")
        pred_img = Image.fromarray(tensor_to_image(pred))

        cells = [
            add_label(lr_img.resize(hr_img.size, Image.Resampling.NEAREST), "low-res"),
            add_label(bicubic_img, "bicubic"),
            add_label(pred_img, "LeafSR"),
            add_label(hr_img, "target"),
        ]
        row = Image.new("RGB", (sum(cell.width for cell in cells), cells[0].height), "white")
        x = 0
        for cell in cells:
            row.paste(cell, (x, 0))
            x += cell.width
        rows.append(row)

    if not rows:
        raise ValueError("No PNG files found for comparison grid.")

    grid = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
