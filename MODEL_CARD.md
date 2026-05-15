# Model Card: LeafSR

## Overview

LeafSR is an ESRGAN-lite conditional GAN for 4x plant leaf image super-resolution. It reconstructs high-resolution leaf images from low-resolution inputs by predicting a residual over bicubic interpolation.

## Intended Use

- Educational computer vision project.
- Super-resolution experiments on plant leaf imagery.
- Kaggle-style batch inference and submission generation.

It is not validated for scientific measurement, disease diagnosis, agricultural decisions, or production image restoration.

## Training Data

The original experiment used a private/course competition dataset with:

- `train_Low_Resolution`
- `train_High_Resolution`
- `test_Low_Resolution`
- `sample_submission.csv`
- `vgg19_weights.pth`

The dataset is not redistributed in this repository.

## Metrics

Validation used image-level mean absolute error.

| Method | Validation MAE |
| --- | ---: |
| Bicubic baseline | 18.20 |
| LeafSR ESRGAN-lite | 16.89 |

## Architecture

- RRDB-style generator.
- Conditional discriminator receiving bicubic HR image and target/generated HR image.
- VGG19 perceptual loss.
- Charbonnier pixel and residual losses.
- FFT magnitude loss.
- Low-weight adversarial objective after warmup.

## Limitations

- Performance is only measured on the original validation split.
- Model quality depends heavily on the competition dataset distribution.
- Super-resolution can hallucinate fine texture and should not be used as evidence of real leaf structure.
- No pretrained checkpoint is committed because model weights can be large and dataset-specific.
