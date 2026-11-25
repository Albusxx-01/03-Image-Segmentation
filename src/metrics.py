"""Segmentation metrics: IoU, Dice coefficient, and pixel accuracy.

Implementations land in commit 9; this scaffold exposes the intended API.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from src.config import DEFAULT_CONFIG_PATH, load_config


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection-over-union for a binary mask batch."""

    raise NotImplementedError("IoU lands in commit 9")


def dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Dice coefficient for a binary mask batch."""

    raise NotImplementedError("Dice coefficient lands in commit 9")


def pixel_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> float:
    """Fraction of pixels whose binarized label matches the target."""

    raise NotImplementedError("pixel accuracy lands in commit 9")


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Return all metrics under stable keys used by evaluation artifacts."""

    return {
        "iou": iou(logits, targets, threshold),
        "dice": dice(logits, targets, threshold),
        "pixel_accuracy": pixel_accuracy(logits, targets, threshold),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.metrics",
        description="Report the metric keys produced by the evaluation stage.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    args = parser.parse_args(argv)
    load_config(args.config)
    print("metric keys: iou, dice, pixel_accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
