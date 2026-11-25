"""Static HTML prediction gallery.

The gallery builder lands in commit 23; this scaffold exposes the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import DEFAULT_CONFIG_PATH, load_config


def build_gallery(
    config: dict[str, Any],
    results_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Render a static gallery of input/ground-truth/prediction images."""

    raise NotImplementedError("HTML gallery lands in commit 23")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.app",
        description="Build a static HTML gallery of segmentation results.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing prediction artifacts (default: from config).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: from config).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    build_gallery(config, results_dir=args.results_dir, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
