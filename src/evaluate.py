"""Checkpoint evaluation with metrics JSON and overlay rendering.

Evaluation loads a checkpoint, scores the selected split, writes
``results/eval/metrics.json``, and renders input/ground-truth/prediction
triptychs for the first ``eval.num_overlays`` samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.config import (
    DEFAULT_CONFIG_PATH,
    config_path,
    load_config,
    resolve_device,
    resolve_path,
)
from src.data import build_dataloader, image_to_display
from src.metrics import compute_metrics
from src.model import load_checkpoint

_COLORS = {
    "ground_truth": (0, 200, 0),
    "prediction": (220, 0, 0),
}


def _checkpoint_path(config: dict[str, Any], checkpoint: str | Path | None) -> Path:
    if checkpoint is None:
        return config_path(config, "train.checkpoint_dir") / str(
            config["train"]["best_checkpoint"]
        )
    return resolve_path(checkpoint, config["_meta"]["root"])


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.float32)
    selected = mask.astype(bool)
    base[selected] = 0.5 * base[selected] + 0.5 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(base.astype(np.uint8), mode="RGB")


def _label(panel: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (panel.width, panel.height + 20), (255, 255, 255))
    canvas.paste(panel, (0, 20))
    ImageDraw.Draw(canvas).text((4, 4), text, fill=(0, 0, 0))
    return canvas


def _save_triplet(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: Path,
) -> None:
    display = image_to_display(image)
    target_mask = target.squeeze(0).numpy()
    prediction_mask = prediction.squeeze(0).numpy()
    panels = [
        _label(_to_pil(display), "input"),
        _label(_overlay(display, target_mask, _COLORS["ground_truth"]), "ground truth"),
        _label(_overlay(display, prediction_mask, _COLORS["prediction"]), "prediction"),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    canvas.save(path)


def evaluate(
    config: dict[str, Any],
    checkpoint: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a checkpoint and write ``metrics.json`` plus overlays."""

    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split: {split}")

    device = torch.device(resolve_device(config))
    checkpoint_path = _checkpoint_path(config, checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model = load_checkpoint(checkpoint_path, config, device)
    loader = build_dataloader(
        config,
        split,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
    )
    threshold = float(config["eval"].get("threshold", 0.5))

    image_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    logit_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            image_chunks.append(images.cpu())
            target_chunks.append(targets.cpu())
            logit_chunks.append(logits.cpu())

    all_images = torch.cat(image_chunks)
    all_targets = torch.cat(target_chunks)
    all_logits = torch.cat(logit_chunks)
    metrics = compute_metrics(all_logits, all_targets, threshold)

    output_dir = config_path(config, "eval.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (torch.sigmoid(all_logits) >= threshold).float()
    dataset = loader.dataset
    overlay_count = min(int(config["eval"].get("num_overlays", 8)), len(dataset))
    overlay_paths: list[str] = []
    for index in range(overlay_count):
        stem = dataset.pairs[index][0].stem
        overlay_path = output_dir / f"{stem}_overlay.png"
        _save_triplet(
            all_images[index],
            all_targets[index],
            predictions[index],
            overlay_path,
        )
        overlay_paths.append(str(overlay_path))

    metrics_path = config_path(config, "eval.metrics_path")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "num_samples": len(dataset),
        **metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"split: {split}")
    print(f"samples: {len(dataset)}")
    print(f"iou: {metrics['iou']:.4f}")
    print(f"dice: {metrics['dice']:.4f}")
    print(f"pixel_accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"metrics: {metrics_path}")
    return {**payload, "metrics_path": str(metrics_path), "overlays": overlay_paths}


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
