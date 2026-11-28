"""Smoke test: train one tiny CPU epoch and verify artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.data import generate_synthetic_dataset
from src.train import train
from tests.helpers import make_toy_config


def test_smoke_train_writes_artifacts(tmp_path: Path) -> None:
    config = make_toy_config(tmp_path)
    generate_synthetic_dataset(config, force=True)

    result = train(config)

    assert result["epochs_run"] == 1
    assert result["device"] == "cpu"
    assert (tmp_path / "models" / "best.pt").is_file()
    assert (tmp_path / "models" / "last.pt").is_file()

    history = json.loads((tmp_path / "results" / "train_history.json").read_text())
    assert len(history) == 1
    for key in (
        "train_loss",
        "val_loss",
        "val_iou",
        "val_dice",
        "val_pixel_accuracy",
        "lr",
    ):
        assert key in history[0]
        assert math.isfinite(history[0][key])
