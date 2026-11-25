"""U-Net encoder-decoder with concatenative skip connections.

The full architecture lands in commit 5. This scaffold defines the class and
builder so downstream modules can import stable names.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from src.config import DEFAULT_CONFIG_PATH, load_config


class UNet(nn.Module):
    """U-Net for binary or multi-class semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("U-Net forward pass lands in commit 5")


def build_model(config: dict[str, Any]) -> UNet:
    """Construct a U-Net from the ``model`` section of a loaded config."""

    model = config["model"]
    return UNet(
        in_channels=model["in_channels"],
        out_channels=model["out_channels"],
        base_filters=model["base_filters"],
        depth=model["depth"],
        dropout=model.get("dropout", 0.0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.model",
        description="Build the U-Net and run a forward-pass sanity check.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run a forward pass on a random tensor and print the output shape.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model = build_model(config)
    if args.sanity:
        size = config["data"]["image_size"]
        dummy = torch.zeros(1, model.in_channels, size[0], size[1])
        output = model(dummy)
        print(f"output shape: {tuple(output.shape)}")
    else:
        print(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
