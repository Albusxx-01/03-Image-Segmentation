"""Single-image inference: mask and overlay generation.

The inference pipeline lands in commit 20; this scaffold exposes the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import DEFAULT_CONFIG_PATH, load_config


def predict_image(
    config: dict[str, Any],
    image_path: str | Path,
    checkpoint: str | Path | None = None,
) -> dict[str, str]:
    """Predict a mask for one image and write ``_mask.png``/``_overlay.png``."""

    raise NotImplementedError("single-image inference lands in commit 20")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Predict a segmentation mask for a single image.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: models/best.pt from the config).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    predict_image(config, args.image, checkpoint=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
