"""Tests for single-image inference outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data import generate_synthetic_dataset, list_pairs
from src.predict import predict_image
from src.train import train
from tests.helpers import make_toy_config


def test_predict_writes_mask_and_overlay(tmp_path: Path) -> None:
    config = make_toy_config(tmp_path)
    generate_synthetic_dataset(config, force=True)
    train(config)

    image_path, _ = list_pairs(config, "test")[0]
    result = predict_image(config, image_path)

    mask_path = Path(result["mask"])
    overlay_path = Path(result["overlay"])
    assert mask_path.is_file()
    assert overlay_path.is_file()

    mask = np.asarray(Image.open(mask_path))
    assert mask.shape == (16, 16)
    assert set(np.unique(mask).tolist()) <= {0, 255}

    overlay = np.asarray(Image.open(overlay_path))
    assert overlay.shape == (16, 16, 3)
