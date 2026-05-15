# Project Notes

## Why This Is More Than a Notebook

The original notebook proved the modeling approach. This repository separates that work into reusable components:

- `leafsr.data` handles paired LR/HR image loading and patch augmentation.
- `leafsr.model` contains the generator and discriminator.
- `leafsr.losses` collects perceptual, frequency, and robust reconstruction losses.
- `scripts/train.py` runs a configurable training loop with validation and checkpointing.
- `scripts/infer.py` supports PNG outputs and Kaggle-style CSV submissions.

## Next Improvements

- Add sample images when the dataset license allows it.
- Save generated comparison grids in `assets/`.
- Add a small public toy dataset for CI-level model-forward tests.
- Track experiments with a table of hyperparameters and validation MAE.
- Export a lightweight CPU inference checkpoint if the trained model is small enough.
