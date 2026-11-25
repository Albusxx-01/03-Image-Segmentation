"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    ConfigError,
    config_path,
    load_config,
    resolve_path,
    seed_from_config,
)


def test_default_config_loads() -> None:
    config = load_config()
    assert config["project"]["name"] == "03-image-segmentation"
    assert config["data"]["splits"] == {"train": 64, "val": 16, "test": 16}
    assert config["model"]["depth"] == 4
    assert config["_meta"]["path"] == str(DEFAULT_CONFIG_PATH)


def test_default_config_exists() -> None:
    assert DEFAULT_CONFIG_PATH.is_file()


def test_seed_is_available() -> None:
    config = load_config()
    assert seed_from_config(config) == 42


def test_resolve_path_relative_to_project_root() -> None:
    assert resolve_path("data/images") == (PROJECT_ROOT / "data" / "images").resolve()
    absolute = PROJECT_ROOT / "models"
    assert resolve_path(absolute) == absolute


def test_config_path_helper() -> None:
    config = load_config()
    assert config_path(config, "train.checkpoint_dir") == (PROJECT_ROOT / "models").resolve()
    with pytest.raises(ConfigError):
        config_path(config, "train.missing_key")


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("config/does-not-exist.yaml")


def test_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(empty)


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)


def test_missing_section_raises(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del config["train"]
    partial = tmp_path / "partial.yaml"
    partial.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required section"):
        load_config(partial)


def test_image_size_must_fit_depth(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["data"]["image_size"] = [8, 8]
    config["model"]["depth"] = 4
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="too small"):
        load_config(tiny)


def test_invalid_split_raises(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["data"]["splits"]["train"] = 0
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="positive integer"):
        load_config(broken)
