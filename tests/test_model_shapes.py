import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leafsr.model import Discriminator, Generator


def test_generator_scales_by_four():
    model = Generator(channels=8, growth=4, num_rrdb=1, scale=4)
    lr = torch.rand(2, 3, 8, 8)
    bicubic = torch.rand(2, 3, 32, 32)
    out = model(lr, bicubic)
    assert out.shape == bicubic.shape


def test_discriminator_accepts_condition_and_target():
    model = Discriminator(base=8)
    cond = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    out = model(cond, target)
    assert out.shape[0] == 2
    assert out.shape[1] == 1
