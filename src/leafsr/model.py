from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int = 64, growth: int = 24) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + growth * 2, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + growth * 3, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + growth * 4, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, channels: int = 64, growth: int = 24) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth)
        self.rdb2 = ResidualDenseBlock(channels, growth)
        self.rdb3 = ResidualDenseBlock(channels, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class Generator(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        num_rrdb: int = 4,
        growth: int = 24,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if scale <= 0 or math.log2(scale) % 1 != 0:
            raise ValueError("scale must be a positive power of two")
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[RRDB(channels, growth) for _ in range(num_rrdb)])
        self.body_conv = nn.Conv2d(channels, channels, 3, padding=1)
        up_layers: list[nn.Module] = []
        for _ in range(int(math.log2(scale))):
            up_layers += [
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.up = nn.Sequential(*up_layers)
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, lr: torch.Tensor, bicubic_hr: torch.Tensor) -> torch.Tensor:
        feat = self.head(lr)
        body = self.body_conv(self.body(feat)) + feat
        residual = self.tail(self.up(body))
        return bicubic_hr + residual


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 24) -> None:
        super().__init__()

        def block(cin: int, cout: int, stride: int = 1, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(cin, cout, 3, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base, norm=False),
            *block(base, base, stride=2),
            *block(base, base * 2),
            *block(base * 2, base * 2, stride=2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 4, stride=2),
            nn.Conv2d(base * 4, 1, 3, padding=1),
        )

    def forward(self, cond_hr: torch.Tensor, target_hr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond_hr, target_hr], dim=1))


def build_generator(config: dict, scale: int) -> Generator:
    model_cfg = config["model"]
    return Generator(
        channels=model_cfg["channels"],
        growth=model_cfg["growth"],
        num_rrdb=model_cfg["num_rrdb"],
        scale=scale,
    )


def build_discriminator(config: dict) -> Discriminator:
    return Discriminator(base=config["model"]["discriminator_base"])
