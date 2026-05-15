# LeafSR

GAN-based 4x super-resolution for plant leaf images.

LeafSR packages a Kaggle/coursework experiment into a reproducible PyTorch project. It trains an ESRGAN-lite conditional GAN that predicts a high-resolution residual over a bicubic upsampled leaf image.

Project site: <https://briskam.github.io/leafsr/>

## Results

Validation was measured with mean absolute error on held-out training images.

| Method | Validation MAE |
| --- | ---: |
| Bicubic baseline | 18.20 |
| LeafSR ESRGAN-lite | 16.89 |

The original run used a Tesla T4, mixed precision, EMA-smoothed generator weights, top-k checkpoint averaging, and 8-way test-time augmentation.

## Highlights

- ESRGAN-lite generator with Residual-in-Residual Dense Blocks.
- Conditional discriminator over bicubic image + target/generated HR image.
- Combined Charbonnier, residual, perceptual, FFT, and adversarial losses.
- CUDA AMP support, cosine schedule, gradient clipping, EMA checkpointing, and early stopping.
- Submission/inference path with test-time augmentation and optional top-k checkpoint ensemble.

## Repository Layout

```text
configs/default.yaml        Training and inference configuration
src/leafsr/                 Reusable Python package
scripts/train.py            Train and validate the model
scripts/infer.py            Generate image outputs or Kaggle submission CSV
scripts/make_comparison.py  Build LR/bicubic/prediction/HR comparison grids
notebooks/                  Original notebook archive
tests/                      Lightweight smoke tests
```

## Data Layout

The dataset is not committed. Place it like this, or pass custom paths in the config:

```text
data/
  train_Low_Resolution/
    image_001.png
  train_High_Resolution/
    image_001.png
  test_Low_Resolution/
    test_001.png
  sample_submission.csv
  vgg19_weights.pth
```

Low-resolution and high-resolution training images must share filenames. The default scale factor is `4`.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```sh
python scripts/train.py --config configs/default.yaml
```

Useful overrides:

```sh
python scripts/train.py \
  --config configs/default.yaml \
  --data-root /path/to/dataset \
  --out-dir runs/leafsr
```

## Inference

Generate enhanced PNG files:

```sh
python scripts/infer.py \
  --checkpoint runs/leafsr/best_leafsr.pt \
  --input-dir data/test_Low_Resolution \
  --output-dir runs/leafsr/predictions
```

Generate a Kaggle-style submission CSV:

```sh
python scripts/infer.py \
  --checkpoint runs/leafsr/best_leafsr.pt \
  --input-dir data/test_Low_Resolution \
  --sample-submission data/sample_submission.csv \
  --submission runs/leafsr/submission.csv
```

## Visual Comparisons

```sh
python scripts/make_comparison.py \
  --checkpoint runs/leafsr/best_leafsr.pt \
  --lr-dir data/train_Low_Resolution \
  --hr-dir data/train_High_Resolution \
  --output assets/comparison_grid.png
```

## Resume Bullet

Built **LeafSR**, a PyTorch super-resolution pipeline for plant leaf imagery using an **ESRGAN-lite conditional GAN**, **VGG perceptual loss**, **EMA checkpointing**, mixed precision training, and 8-way test-time augmentation, improving validation MAE from **18.20** bicubic baseline to **16.89**.

## Notes

The original dataset and VGG weights are expected to come from the competition/course environment and are intentionally excluded from git.
