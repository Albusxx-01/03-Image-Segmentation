"""Tests for the U-Net forward pass, shape contract, and depth guards."""

from __future__ import annotations

import pytest
import torch

from src.config import load_config
from src.model import UNet, build_model


def test_forward_pass_returns_binary_logits() -> None:
    model = UNet(in_channels=3, out_channels=1, base_filters=4, depth=2).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 16, 16))

    assert output.shape == (2, 1, 16, 16)
    assert torch.is_floating_point(output)


def test_forward_pass_supports_multiclass_output() -> None:
    model = UNet(in_channels=1, out_channels=3, base_filters=4, depth=2).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 16, 16))

    assert output.shape == (2, 3, 16, 16)


def test_build_model_matches_config_and_sanity_shape() -> None:
    config = load_config()
    model = build_model(config)

    assert model.in_channels == config["model"]["in_channels"]
    assert model.out_channels == config["model"]["out_channels"]
    assert model.base_filters == config["model"]["base_filters"]
    assert model.depth == config["model"]["depth"]

    height, width = config["data"]["image_size"]
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, model.in_channels, height, width))
    assert output.shape == (1, model.out_channels, height, width)


def test_indivisible_spatial_size_raises() -> None:
    model = UNet(in_channels=3, out_channels=1, base_filters=4, depth=3).eval()
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 20, 20))


def test_input_channel_mismatch_raises() -> None:
    model = UNet(in_channels=3, out_channels=1, base_filters=4, depth=2).eval()
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(1, 1, 16, 16))


def test_non_4d_input_raises() -> None:
    model = UNet(in_channels=3, out_channels=1, base_filters=4, depth=2).eval()
    with pytest.raises(ValueError, match="4D"):
        model(torch.randn(3, 16, 16))


def test_invalid_constructor_arguments_raise() -> None:
    with pytest.raises(ValueError, match="depth"):
        UNet(depth=0)
    with pytest.raises(ValueError, match="base_filters"):
        UNet(base_filters=0)


def test_model_has_trainable_parameters() -> None:
    model = UNet(in_channels=3, out_channels=1, base_filters=4, depth=2)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
