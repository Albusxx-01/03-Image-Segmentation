"""Seeded training loop with checkpoints and history.

The training loop lands in commits 12-13; this scaffold exposes the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import DEFAULT_CONFIG_PATH, load_config, seed_from_config, set_seed


def train(config: dict[str, Any], resume: str | Path | None = None) -> dict[str, Any]:
    """Train a U-Net and write checkpoints plus ``train_history.json``."""

    raise NotImplementedError("training loop lands in commit 12")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train the U-Net on image/mask pairs.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint to resume training from.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    set_seed(seed_from_config(config))
    train(config, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
