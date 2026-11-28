"""Tests for checkpoint evaluation, metrics JSON, and overlays."""

from __future__ import annotations

import json
from pathlib import Path

from src.data import generate_synthetic_dataset
from src.evaluate import evaluate
from src.train import train
from tests.helpers import make_toy_config


def test_evaluate_writes_metrics_and_overlays(tmp_path: Path) -> None:
    config = make_toy_config(tmp_path)
    generate_synthetic_dataset(config, force=True)
    train(config)

    result = evaluate(config, split="test")

    assert result["split"] == "test"
    assert result["num_samples"] == 1
    for key in ("iou", "dice", "pixel_accuracy"):
        assert key in result
        assert 0.0 <= result[key] <= 1.0

    metrics_path = Path(config["eval"]["metrics_path"])
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text())
    assert {"split", "checkpoint", "num_samples", "iou", "dice", "pixel_accuracy"} <= set(
        payload
    )

    overlays = list(Path(config["eval"]["output_dir"]).glob("*_overlay.png"))
    assert len(overlays) == 1
