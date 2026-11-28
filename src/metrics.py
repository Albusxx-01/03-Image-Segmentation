"""Segmentation metrics: IoU, Dice coefficient, and pixel accuracy.

Metrics consume raw logits and binary targets, binarize both, and aggregate over
every element of the input tensor (micro-average across the batch). Empty
foreground and prediction sets count as a perfect match.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from src.config import DEFAULT_CONFIG_PATH, load_config


def _as_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Validate the target shape and return it aligned with ``logits``."""

    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must be torch tensors")

    if targets.shape == logits.shape:
        return targets

    if logits.dim() == 4 and logits.size(1) == 1 and targets.dim() == 3:
        expected = (logits.size(0), logits.size(2), logits.size(3))
        if tuple(targets.shape) == expected:
            return targets.unsqueeze(1)

    raise ValueError(
        f"target shape {tuple(targets.shape)} does not match "
        f"logits shape {tuple(logits.shape)}"
    )


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def _prepare(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    target = (_as_target(logits, targets) >= 0.5).float()
    return _binarize(logits, threshold), target


def iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection-over-union for a binary mask batch."""

    predictions, target = _prepare(logits, targets, threshold)
    intersection = (predictions * target).sum()
    union = predictions.sum() + target.sum() - intersection
    if union.item() == 0:
        return 1.0
    return (intersection / union).item()


def dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Dice coefficient for a binary mask batch."""

    predictions, target = _prepare(logits, targets, threshold)
    intersection = (predictions * target).sum()
    denominator = predictions.sum() + target.sum()
    if denominator.item() == 0:
        return 1.0
    return (2 * intersection / denominator).item()


def pixel_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> float:
    """Fraction of pixels whose binarized label matches the target."""

    predictions, target = _prepare(logits, targets, threshold)
    return (predictions == target).float().mean().item()


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
