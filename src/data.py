"""Synthetic image/mask dataset generation and loading.

Day 2 implements deterministic pair generation, split manifests, and
PyTorch dataloaders. This scaffold only defines the public surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import DEFAULT_CONFIG_PATH, load_config, resolve_path

SPLITS = ("train", "val", "test")


class DataError(RuntimeError):
    """Raised when dataset generation or loading fails."""


def split_dir(config: dict[str, Any], split: str, kind: str = "images") -> Path:
    """Return ``<data.root>/<kind>/<split>`` as an absolute path."""

    if split not in SPLITS:
        raise DataError(f"unknown split: {split}")
    if kind not in ("images", "masks"):
        raise DataError(f"unknown data kind: {kind}")
    root = resolve_path(config["data"]["root"], config["_meta"]["root"])
    return root / kind / split


def generate_synthetic_dataset(
    config: dict[str, Any], force: bool = False
) -> dict[str, int]:
    """Generate deterministic image/mask pairs for every split."""

    raise NotImplementedError("synthetic dataset generation lands in commit 2")


def build_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """Build train/val/test dataloaders from the generated dataset."""

    raise NotImplementedError("dataloaders land with the training pipeline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data",
        description="Generate or inspect the synthetic segmentation dataset.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the dataset even if files already exist.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    generate_synthetic_dataset(config, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
