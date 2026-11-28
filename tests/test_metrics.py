"""Tests for IoU, Dice, and pixel-accuracy metrics."""

from __future__ import annotations

import pytest
import torch

from src.metrics import compute_metrics, dice, iou, pixel_accuracy


def logits_from_predictions(predictions: torch.Tensor) -> torch.Tensor:
    return torch.where(predictions > 0.5, 10.0, -10.0)


def test_perfect_prediction_scores_one() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    metrics = compute_metrics(logits_from_predictions(targets), targets)

    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_half_overlap_matches_hand_computed_values() -> None:
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    predictions = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    metrics = compute_metrics(logits_from_predictions(predictions), targets)

    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["pixel_accuracy"] == pytest.approx(0.5)


def test_disjoint_prediction_scores_zero() -> None:
    targets = torch.ones(1, 1, 2, 2)
    predictions = torch.zeros_like(targets)

    metrics = compute_metrics(logits_from_predictions(predictions), targets)

    assert metrics["iou"] == pytest.approx(0.0)
    assert metrics["dice"] == pytest.approx(0.0)
    assert metrics["pixel_accuracy"] == pytest.approx(0.0)


def test_empty_prediction_and_target_is_perfect() -> None:
    targets = torch.zeros(1, 1, 4, 4)

    metrics = compute_metrics(logits_from_predictions(targets), targets)

    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_metrics_aggregate_over_batch() -> None:
    targets = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ]
    )
    predictions = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ]
    )

    metrics = compute_metrics(logits_from_predictions(predictions), targets)

    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["pixel_accuracy"] == pytest.approx(7 / 8)


def test_threshold_controls_binarization() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.ones(1, 1, 2, 2)

    assert iou(logits, targets, threshold=0.5) == pytest.approx(1.0)
    assert iou(logits, targets, threshold=0.6) == pytest.approx(0.0)


def test_uint8_style_targets_are_binarized() -> None:
    targets = torch.tensor([[[[255.0, 0.0], [255.0, 0.0]]]])
    predictions = (targets > 0.5).float()

    metrics = compute_metrics(logits_from_predictions(predictions), targets)

    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_3d_targets_are_accepted() -> None:
    targets = torch.ones(2, 4, 4)
    logits = torch.full((2, 1, 4, 4), 10.0)

    assert iou(logits, targets) == pytest.approx(1.0)
    assert dice(logits, targets) == pytest.approx(1.0)
    assert pixel_accuracy(logits, targets) == pytest.approx(1.0)


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="target shape"):
        iou(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 3, 3))


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold"):
        iou(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), threshold=1.5)


def test_metric_keys_are_stable() -> None:
    metrics = compute_metrics(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))

    assert list(metrics) == ["iou", "dice", "pixel_accuracy"]
