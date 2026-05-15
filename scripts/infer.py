#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leafsr.config import ensure_dir, load_config
from leafsr.inference import infer_with_tta, make_bicubic_tensor, tensor_to_image
from leafsr.model import build_generator
from leafsr.utils import get_device, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LeafSR inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    use_amp = device.type == "cuda"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator = build_generator(config, scale=config["scale"]).to(device)
    generator.load_state_dict(checkpoint["G"])
    generator.eval()

    input_dir = Path(args.input_dir)
    image_names = []
    if args.sample_submission:
        with Path(args.sample_submission).open("r", newline="", encoding="utf-8") as handle:
            image_names = [row["Id"] for row in csv.DictReader(handle)]
    else:
        image_names = [path.name for path in sorted(input_dir.glob("*.png"))]

    output_dir = ensure_dir(args.output_dir) if args.output_dir else None
    writer = None
    submission_handle = None
    if args.submission:
        Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
        submission_handle = Path(args.submission).open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(submission_handle, fieldnames=["Id", "Pixels"])
        writer.writeheader()

    try:
        for image_name in image_names:
            lr_path = input_dir / image_name
            lr, bicubic = make_bicubic_tensor(lr_path, config["scale"], device)
            with torch.no_grad():
                if args.no_tta:
                    pred = generator(lr, bicubic)
                else:
                    pred = infer_with_tta(generator, lr, bicubic, device, use_amp=use_amp)
            pred_u8 = tensor_to_image(pred)
            if output_dir:
                save_image(pred_u8, output_dir / image_name)
            if writer:
                pixels = " ".join(map(str, pred_u8.reshape(-1).tolist()))
                writer.writerow({"Id": image_name, "Pixels": pixels})
    finally:
        if submission_handle:
            submission_handle.close()


if __name__ == "__main__":
    main()
