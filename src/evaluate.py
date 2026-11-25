"""Checkpoint evaluation with metrics JSON and overlay rendering.

The evaluation pipeline lands in commits 16-17; this scaffold exposes the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import DEFAULT_CONFIG_PATH, load_config


def evaluate(
    config: dict[str, Any],
    checkpoint: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a checkpoint and write ``metrics.json`` plus overlays."""

    raise NotImplementedError("evaluation pipeline lands in commit 16")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate a segmentation checkpoint and render overlays.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: models/best.pt from the config).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate (default: test).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    evaluate(config, checkpoint=args.checkpoint, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
