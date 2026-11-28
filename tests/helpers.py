"""Shared helpers for pipeline tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.config import load_config


def make_toy_config(tmp_path: Path) -> dict[str, Any]:
    config = copy.deepcopy(load_config())
    config["data"]["root"] = str(tmp_path / "data")
    config["data"]["splits"] = {"train": 2, "val": 1, "test": 1}
    config["data"]["image_size"] = [16, 16]
    config["data"]["num_workers"] = 0
    config["model"]["base_filters"] = 4
    config["model"]["depth"] = 2
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 1
    config["train"]["patience"] = 2
    config["train"]["checkpoint_dir"] = str(tmp_path / "models")
    config["train"]["history_path"] = str(tmp_path / "results" / "train_history.json")
    config["eval"]["batch_size"] = 1
    config["eval"]["num_overlays"] = 1
    config["eval"]["output_dir"] = str(tmp_path / "results" / "eval")
    config["eval"]["metrics_path"] = str(tmp_path / "results" / "eval" / "metrics.json")
    config["infer"]["output_dir"] = str(tmp_path / "results" / "predictions")
    config["project"]["device"] = "cpu"
    return config
