"""Tests for deterministic synthetic dataset generation and pairing."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.data import (
    DataError,
    generate_synthetic_dataset,
    list_pairs,
    pair_stem,
    split_dir,
)


def make_config(
    tmp_path: Path,
    splits: dict[str, int] | None = None,
    size: tuple[int, int] = (32, 32),
    channels: int = 3,
    seed: int = 42,
) -> dict:
    config = copy.deepcopy(load_config())
    config["data"]["root"] = str(tmp_path / "data")
    config["data"]["splits"] = splits if splits is not None else {"train": 3, "val": 2, "test": 2}
    config["data"]["image_size"] = list(size)
    config["data"]["channels"] = channels
    config["project"]["seed"] = seed
    return config


def test_generates_expected_counts_and_layout(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    counts = generate_synthetic_dataset(config, force=True)

    assert counts == config["data"]["splits"]
    for split, count in config["data"]["splits"].items():
        image_names = {p.name for p in split_dir(config, split, "images").glob("*.png")}
        mask_names = {p.name for p in split_dir(config, split, "masks").glob("*.png")}
        expected = {f"{pair_stem(index)}.png" for index in range(count)}
        assert image_names == expected
        assert mask_names == expected
        assert len(list_pairs(config, split)) == count


def test_pairs_have_matching_shapes_and_binary_masks(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    generate_synthetic_dataset(config, force=True)

    for split in ("train", "val", "test"):
        image_path, mask_path = list_pairs(config, split)[0]
        image = np.asarray(Image.open(image_path))
        mask = np.asarray(Image.open(mask_path))
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert mask.shape == (32, 32)
        assert set(np.unique(mask).tolist()) <= {0, 255}


def test_mask_marks_the_drawn_foreground(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    generate_synthetic_dataset(config, force=True)

    image_path, mask_path = list_pairs(config, "train")[0]
    image = np.asarray(Image.open(image_path)).astype(float)
    foreground = np.asarray(Image.open(mask_path)) > 0

    assert 0.0 < foreground.mean() < 1.0
    assert image[foreground].mean() > image[~foreground].mean() + 50


def test_generation_is_deterministic(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    generate_synthetic_dataset(config, force=True)
    first = {
        (split, kind, path.name): path.read_bytes()
        for split in ("train", "val", "test")
        for kind in ("images", "masks")
        for path in split_dir(config, split, kind).glob("*.png")
    }

    generate_synthetic_dataset(config, force=True)
    second = {
        (split, kind, path.name): path.read_bytes()
        for split in ("train", "val", "test")
        for kind in ("images", "masks")
        for path in split_dir(config, split, kind).glob("*.png")
    }

    assert first == second


def test_force_removes_stale_pairs(tmp_path: Path) -> None:
    config = make_config(tmp_path, splits={"train": 3, "val": 1, "test": 1})
    generate_synthetic_dataset(config, force=True)
    assert (split_dir(config, "train", "images") / f"{pair_stem(2)}.png").is_file()

    config["data"]["splits"]["train"] = 2
    generate_synthetic_dataset(config, force=True)

    stale = split_dir(config, "train", "images") / f"{pair_stem(2)}.png"
    assert not stale.is_file()


def test_missing_mask_raises(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    generate_synthetic_dataset(config, force=True)

    _, mask_path = list_pairs(config, "train")[0]
    mask_path.unlink()

    with pytest.raises(DataError, match="missing mask"):
        list_pairs(config, "train")


def test_unknown_split_raises(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(DataError, match="unknown split"):
        split_dir(config, "holdout", "images")


def test_grayscale_generation(tmp_path: Path) -> None:
    config = make_config(tmp_path, channels=1)
    generate_synthetic_dataset(config, force=True)

    image_path, mask_path = list_pairs(config, "train")[0]
    assert np.asarray(Image.open(image_path)).shape == (32, 32)
    assert np.asarray(Image.open(mask_path)).shape == (32, 32)


def test_unsupported_channels_raise(tmp_path: Path) -> None:
    config = make_config(tmp_path, channels=2)
    with pytest.raises(DataError, match="channels"):
        generate_synthetic_dataset(config, force=True)
