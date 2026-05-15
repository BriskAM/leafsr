#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leafsr.checkpoint import clone_state, init_ema, load_state_copy, save_checkpoint, update_ema
from leafsr.config import ensure_dir, load_config, resolve_data_paths
from leafsr.data import SuperResolutionPatchDataset, list_pngs, split_paths
from leafsr.evaluate import bicubic_score, evaluate_generator
from leafsr.losses import VGGPerceptualLoss, charbonnier_loss, fft_loss
from leafsr.model import build_discriminator, build_generator
from leafsr.utils import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeafSR ESRGAN-lite model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def lr_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    paths = resolve_data_paths(config, args.data_root)
    out_dir = ensure_dir(args.out_dir or config["output"]["dir"])
    checkpoint_path = out_dir / config["output"]["checkpoint"]
    metrics_path = out_dir / config["output"]["metrics"]

    device = get_device()
    use_amp = device.type == "cuda"
    print(f"device={device} amp={use_amp}")

    train_lr_dir = paths["train_lr"]
    train_hr_dir = paths["train_hr"]
    if not train_lr_dir.exists() or not train_hr_dir.exists():
        raise FileNotFoundError(f"Expected train dirs at {train_lr_dir} and {train_hr_dir}")

    all_lr_paths = list_pngs(train_lr_dir)
    train_lr_paths, val_lr_paths = split_paths(all_lr_paths, config["data"]["val_fraction"])
    print(f"train_images={len(train_lr_paths)} val_images={len(val_lr_paths)}")

    training_cfg = config["training"]
    scale = config["scale"]
    train_ds = SuperResolutionPatchDataset(
        train_lr_paths,
        hr_dir=train_hr_dir,
        scale=scale,
        patch_size=training_cfg["patch_size"],
        patches_per_image=training_cfg["patches_per_image"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    generator = build_generator(config, scale=scale).to(device)
    discriminator = build_discriminator(config).to(device)
    perceptual_loss = VGGPerceptualLoss(paths["vgg_weights"]).to(device)
    adversarial_loss = nn.BCEWithLogitsLoss()

    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=training_cfg["generator_lr"],
        betas=(0.9, 0.99),
        weight_decay=training_cfg["weight_decay"],
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=training_cfg["discriminator_lr"],
        betas=(0.9, 0.99),
        weight_decay=training_cfg["weight_decay"],
    )
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        opt_g,
        lr_lambda=lambda epoch: lr_lambda(epoch, training_cfg["warmup_epochs"], training_cfg["epochs"]),
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        opt_d,
        lr_lambda=lambda epoch: lr_lambda(epoch, training_cfg["warmup_epochs"], training_cfg["epochs"]),
    )

    ema_state = init_ema(generator)
    best_ckpts: list[tuple[float, dict[str, torch.Tensor]]] = []
    best_val = float("inf")
    epochs_without_improvement = 0
    baseline = bicubic_score(val_lr_paths, train_hr_dir, scale=scale)
    print(f"val_bicubic_mae={baseline:.6f}")

    history = [{"epoch": 0, "val_bicubic_mae": baseline}]
    loss_cfg = config["loss"]

    for epoch in range(1, training_cfg["epochs"] + 1):
        generator.train()
        discriminator.train()
        g_losses: list[float] = []
        d_losses: list[float] = []

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{training_cfg['epochs']}")
        for step_idx, (lr_patch, bicubic_hr, hr_patch, residual_target) in enumerate(progress, start=1):
            lr_patch = lr_patch.to(device, non_blocking=True)
            bicubic_hr = bicubic_hr.to(device, non_blocking=True)
            hr_patch = hr_patch.to(device, non_blocking=True)
            residual_target = residual_target.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_hr = generator(lr_patch, bicubic_hr)

            if epoch > training_cfg["warmup_epochs"]:
                opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    real_logits = discriminator(bicubic_hr, hr_patch)
                    fake_logits = discriminator(bicubic_hr, fake_hr.detach())
                    d_real = adversarial_loss(real_logits, torch.ones_like(real_logits))
                    d_fake = adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (d_real + d_fake)
                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=training_cfg["grad_clip"])
                scaler_d.step(opt_d)
                scaler_d.update()
                d_losses.append(float(d_loss.item()))

            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_hr = generator(lr_patch, bicubic_hr)
                residual_pred = fake_hr - bicubic_hr
                l_pix = charbonnier_loss(fake_hr, hr_patch)
                l_res = charbonnier_loss(residual_pred, residual_target)
                l_perc = perceptual_loss(fake_hr, hr_patch)
                l_freq = fft_loss(fake_hr, hr_patch)
                if epoch > training_cfg["warmup_epochs"]:
                    logits = discriminator(bicubic_hr, fake_hr)
                    l_adv = adversarial_loss(logits, torch.ones_like(logits))
                else:
                    l_adv = torch.tensor(0.0, device=device)
                g_loss = (
                    loss_cfg["pixel"] * l_pix
                    + loss_cfg["residual"] * l_res
                    + loss_cfg["perceptual"] * l_perc
                    + loss_cfg["fft"] * l_freq
                    + loss_cfg["adversarial"] * l_adv
                )
            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=training_cfg["grad_clip"])
            scaler_g.step(opt_g)
            scaler_g.update()
            if step_idx % training_cfg["ema_every"] == 0:
                update_ema(generator, ema_state, training_cfg["ema_decay"])
            g_losses.append(float(g_loss.item()))
            progress.set_postfix(g_loss=np.mean(g_losses), d_loss=np.mean(d_losses) if d_losses else 0.0)

        scheduler_g.step()
        if epoch > training_cfg["warmup_epochs"]:
            scheduler_d.step()

        raw_state = clone_state(generator)
        load_state_copy(generator, ema_state)
        val_mae = evaluate_generator(generator, val_lr_paths, train_hr_dir, device, scale=scale, use_amp=use_amp)
        generator.load_state_dict(raw_state)

        row = {
            "epoch": epoch,
            "lr_g": opt_g.param_groups[0]["lr"],
            "g_loss": float(np.mean(g_losses)),
            "d_loss": float(np.mean(d_losses)) if d_losses else 0.0,
            "val_mae": val_mae,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_mae < best_val - training_cfg["early_stopping_min_delta"]:
            best_val = val_mae
            epochs_without_improvement = 0
            raw_state = clone_state(generator)
            load_state_copy(generator, ema_state)
            save_checkpoint(checkpoint_path, generator, discriminator, best_val, epoch, config)
            best_ckpts.append((best_val, clone_state(generator)))
            best_ckpts = sorted(best_ckpts, key=lambda item: item[0])[: training_cfg["top_k"]]
            generator.load_state_dict(raw_state)
            print(f"saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"no significant improvement for {epochs_without_improvement} epoch(s)")

        metrics_path.write_text(json.dumps({"best_val_mae": best_val, "history": history}, indent=2), encoding="utf-8")

        if epoch > training_cfg["warmup_epochs"] and epochs_without_improvement >= training_cfg["early_stopping_patience"]:
            print("early stopping triggered")
            break


if __name__ == "__main__":
    main()
