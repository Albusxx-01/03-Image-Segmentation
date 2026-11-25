"""U-Net encoder-decoder with concatenative skip connections.

The network returns per-pixel logits with shape ``(B, out_channels, H, W)``.
Spatial dimensions must be divisible by ``2 ** depth`` so encoder and decoder
feature maps line up at every skip connection.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from src.config import DEFAULT_CONFIG_PATH, load_config


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if base_filters <= 0:
            raise ValueError("base_filters must be positive")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.dropout = dropout

        self.encoders = nn.ModuleList()
        encoder_channels: list[int] = []
        channels = in_channels
        for level in range(depth):
            level_channels = base_filters * (2**level)
            self.encoders.append(ConvBlock(channels, level_channels, dropout))
            encoder_channels.append(level_channels)
            channels = level_channels

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels, base_filters * (2**depth), dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in reversed(range(depth)):
            skip_channels = encoder_channels[level]
            self.upconvs.append(
                nn.ConvTranspose2d(base_filters * (2 ** (level + 1)), skip_channels, 2, stride=2)
            )
            self.decoders.append(ConvBlock(skip_channels * 2, skip_channels, dropout))

        self.head = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected 4D input (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        factor = 2**self.depth
        height, width = x.shape[-2], x.shape[-1]
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"spatial size {(height, width)} must be divisible by 2**depth={factor}"
            )

        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = torch.cat((skip, x), dim=1)
            x = decoder(x)

        return self.head(x)


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
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        print(f"output shape: {tuple(output.shape)}")
    else:
        print(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
