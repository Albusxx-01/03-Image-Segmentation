"""Segmentation losses: BCE, Dice, and a combined objective.

Implementations land in commit 8; this scaffold exposes the intended API.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from src.config import DEFAULT_CONFIG_PATH, load_config


class BCELoss(nn.Module):
    """Binary cross-entropy over per-pixel logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("BCE loss lands in commit 8")


class DiceLoss(nn.Module):
    """Soft Dice loss: ``1 - (2|A n B| + smooth) / (|A| + |B| + smooth)``."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Dice loss lands in commit 8")


class CombinedLoss(nn.Module):
    """Weighted sum of BCE and Dice losses."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("combined loss lands in commit 8")


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
    args = parser.parse_args(argv)
    config = load_config(args.config)
    print(build_loss(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
