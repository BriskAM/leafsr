#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXPECTED_DIRS = {
    "train_Low_Resolution",
    "train_High_Resolution",
    "test_Low_Resolution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unpack the LeafSR competition dataset.")
    parser.add_argument("zip_path", help="Path to plant-leaves-super-resolution-challenge.zip")
    parser.add_argument("--out-dir", default="data", help="Output directory for extracted data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    out_dir = Path(args.out_dir)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        top_level = {name.split("/", 1)[0] for name in names if name}
        missing = EXPECTED_DIRS - top_level
        if missing:
            raise ValueError(f"Dataset ZIP is missing expected folders: {sorted(missing)}")
        archive.extractall(out_dir)

    print(f"Extracted dataset to {out_dir.resolve()}")
    for folder in sorted(EXPECTED_DIRS):
        count = len(list((out_dir / folder).glob("*.png")))
        print(f"{folder}: {count} png files")


if __name__ == "__main__":
    main()
