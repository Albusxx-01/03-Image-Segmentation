"""Configuration loading, validation, and reproducibility helpers.

All stage modules (data, model, train, evaluate, predict, app) obtain their
settings through :func:`load_config` so that a single YAML file drives the
whole pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "unet.yaml"

REQUIRED_SECTIONS = ("project", "data", "model", "train", "eval", "infer", "app")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or invalid."""


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{section}.{key}'")
    return mapping[key]


def _validate(config: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"missing required section(s): {', '.join(missing)}")

    for section in REQUIRED_SECTIONS:
        if not isinstance(config[section], dict):
            raise ConfigError(f"section '{section}' must be a mapping")

    project = config["project"]
    seed = _require(project, "seed", "project")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("project.seed must be a non-negative integer")

    data = config["data"]
    splits = _require(data, "splits", "data")
    if not isinstance(splits, dict) or not splits:
        raise ConfigError("data.splits must be a non-empty mapping")
    for name, count in splits.items():
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ConfigError(f"data.splits.{name} must be a positive integer")

    size = _require(data, "image_size", "data")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)
    ):
        raise ConfigError("data.image_size must be a pair of positive integers")

    classes = _require(data, "classes", "data")
    if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
        raise ConfigError("data.classes must be an integer >= 2")

    channels = _require(data, "channels", "data")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise ConfigError("data.channels must be a positive integer")

    model = config["model"]
    depth = _require(model, "depth", "model")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 2:
        raise ConfigError("model.depth must be an integer >= 2")

    base_filters = _require(model, "base_filters", "model")
    if not isinstance(base_filters, int) or isinstance(base_filters, bool) or base_filters <= 0:
        raise ConfigError("model.base_filters must be a positive integer")

    min_side = 2**depth
    if min(size) < min_side:
        raise ConfigError(
            f"data.image_size {tuple(size)} is too small for model.depth={depth} "
            f"(each side must be >= {min_side})"
        )

    train = config["train"]
    epochs = _require(train, "epochs", "train")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ConfigError("train.epochs must be a positive integer")
    batch_size = _require(train, "batch_size", "train")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ConfigError("train.batch_size must be a positive integer")
    lr = _require(train, "lr", "train")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
        raise ConfigError("train.lr must be a positive number")


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_CONFIG_PATH
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Relative paths are resolved against the repository root. The returned
    mapping includes a ``_meta`` entry with the resolved config path and project
    root for downstream path resolution.
    """

    config_path = _resolve_config_path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {config_path}")

    _validate(raw)
    raw["_meta"] = {"path": str(config_path), "root": str(PROJECT_ROOT)}
    return raw


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a config path value against the repository root."""

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    base = Path(root) if root is not None else PROJECT_ROOT
    return (base / candidate).resolve()


def config_path(config: dict[str, Any], dotted_key: str) -> Path:
    """Return an absolute path for a dotted config key, e.g. ``train.lr``."""

    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"unknown config key: {dotted_key}")
        value = value[part]
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"config key '{dotted_key}' is not a path")
    root = config.get("_meta", {}).get("root")
    return resolve_path(value, root)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_from_config(config: dict[str, Any]) -> int:
    """Return the validated global seed from a loaded config."""

    return int(config["project"]["seed"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.config",
        description="Load and validate the segmentation YAML configuration.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the validated configuration as JSON.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"config ok: {config['_meta']['path']}")
    print(f"project root: {config['_meta']['root']}")
    if args.show:
        printable = {k: v for k, v in config.items() if k != "_meta"}
        print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
