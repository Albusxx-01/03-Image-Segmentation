"""Segmentation losses: BCE, Dice, and a combined objective.

All losses consume raw logits with shape ``(B, C, H, W)`` and binary targets
with either the same shape or ``(B, H, W)`` when ``C == 1``. Targets are cast
to the logits' device and dtype, so ``{0, 1}`` and ``{0, 255}`` masks must be
normalized to the unit interval before they reach these losses.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.config import DEFAULT_CONFIG_PATH, load_config


def _as_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Validate the target shape and return it as a float tensor on ``logits``."""

    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must be torch tensors")

    if targets.shape == logits.shape:
        return targets.to(device=logits.device, dtype=logits.dtype)

    if logits.dim() == 4 and logits.size(1) == 1 and targets.dim() == 3:
        expected = (logits.size(0), logits.size(2), logits.size(3))
        if tuple(targets.shape) == expected:
            return targets.unsqueeze(1).to(device=logits.device, dtype=logits.dtype)

    raise ValueError(
        f"target shape {tuple(targets.shape)} does not match "
        f"logits shape {tuple(logits.shape)}"
    )


class BCELoss(nn.Module):
    """Binary cross-entropy over per-pixel logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, _as_target(logits, targets))


class DiceLoss(nn.Module):
    """Soft Dice loss: ``1 - (2|A n B| + smooth) / (|A| + |B| + smooth)``."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target = _as_target(logits, targets)
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.dim()))
        intersection = (probabilities * target).sum(dims)
        denominator = probabilities.sum(dims) + target.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of BCE and Dice losses."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Construct the loss selected by ``train.loss``."""

    train = config["train"]
    name = str(train.get("loss", "bce_dice")).lower()
    if name == "bce":
        return BCELoss()
    if name == "dice":
        return DiceLoss()
    if name in ("bce_dice", "dice_bce", "combined"):
        return CombinedLoss(
            bce_weight=float(train.get("bce_weight", 0.5)),
            dice_weight=float(train.get("dice_weight", 0.5)),
        )
    raise ValueError(f"unknown loss: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.losses",
        description="Inspect the configured segmentation loss.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run a forward pass on random logits and print the loss value.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    loss = build_loss(config)
    print(loss)
    if args.sanity:
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        targets = (torch.rand(2, 1, 8, 8) > 0.5).float()
        value = loss(logits, targets)
        value.backward()
        print(f"loss: {value.item():.6f}")
        print(f"grad finite: {bool(torch.isfinite(logits.grad).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
