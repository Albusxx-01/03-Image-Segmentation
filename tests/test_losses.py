"""Tests for BCE, Dice, and combined segmentation losses."""

from __future__ import annotations

import math

import pytest
import torch

from src.config import load_config
from src.losses import BCELoss, CombinedLoss, DiceLoss, build_loss


def test_bce_matches_manual_half_probability() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

    assert BCELoss()(logits, targets).item() == pytest.approx(math.log(2.0))


def test_bce_prefers_confident_correct_predictions() -> None:
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    confident = torch.where(targets > 0.5, 10.0, -10.0)

    assert BCELoss()(confident, targets).item() < 0.01
    assert BCELoss()(-confident, targets).item() > 1.0


def test_dice_perfect_overlap_is_zero() -> None:
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    logits = torch.where(targets > 0.5, 20.0, -20.0)

    assert DiceLoss()(logits, targets).item() == pytest.approx(0.0, abs=1e-6)


def test_dice_half_probability_matches_hand_computed_value() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.ones(1, 1, 2, 2)

    # p=0.5, |A n B|=2, |A|=2, |B|=4, smooth=1
    expected = 1 - (2 * 2 + 1) / (2 + 4 + 1)
    assert DiceLoss()(logits, targets).item() == pytest.approx(expected)


def test_dice_fully_wrong_prediction_is_near_one() -> None:
    targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    logits = torch.full((1, 1, 2, 2), -20.0)

    # No intersection and only the smooth term remains in the numerator.
    expected = 1 - 1 / (4 + 1)
    assert DiceLoss()(logits, targets).item() == pytest.approx(expected)


def test_targets_accept_3d_masks() -> None:
    logits = torch.zeros(2, 1, 4, 4)
    targets = torch.ones(2, 4, 4)

    assert torch.isfinite(BCELoss()(logits, targets))
    assert torch.isfinite(DiceLoss()(logits, targets))


def test_combined_loss_is_weighted_sum() -> None:
    logits = torch.randn(2, 1, 4, 4)
    targets = (torch.rand(2, 1, 4, 4) > 0.5).float()

    bce = BCELoss()(logits, targets)
    dice_loss = DiceLoss()(logits, targets)
    combined = CombinedLoss(bce_weight=0.25, dice_weight=0.75)(logits, targets)

    assert combined.item() == pytest.approx((0.25 * bce + 0.75 * dice_loss).item())


def test_losses_are_differentiable_and_finite() -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    targets = (torch.rand(2, 1, 8, 8) > 0.5).float()

    loss = CombinedLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())


def test_build_loss_selects_configured_loss() -> None:
    config = load_config()
    assert isinstance(build_loss(config), CombinedLoss)

    config["train"]["loss"] = "bce"
    assert isinstance(build_loss(config), BCELoss)

    config["train"]["loss"] = "dice"
    assert isinstance(build_loss(config), DiceLoss)


def test_unknown_loss_raises() -> None:
    config = load_config()
    config["train"]["loss"] = "focal"

    with pytest.raises(ValueError, match="unknown loss"):
        build_loss(config)


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="target shape"):
        BCELoss()(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 3, 3))


def test_invalid_smooth_raises() -> None:
    with pytest.raises(ValueError, match="smooth"):
        DiceLoss(smooth=0.0)
