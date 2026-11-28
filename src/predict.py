"""Single-image inference: mask and overlay generation.

Inference loads a checkpoint, resizes the input to ``data.image_size``, applies
the same normalization used for training, and writes ``<stem>_mask.png`` plus a
red ``<stem>_overlay.png`` into ``infer.output_dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.config import (
    DEFAULT_CONFIG_PATH,
    config_path,
    load_config,
    resolve_device,
    resolve_path,
)
from src.model import load_checkpoint


def _checkpoint_path(config: dict[str, Any], checkpoint: str | Path | None) -> Path:
    if checkpoint is None:
        return config_path(config, "train.checkpoint_dir") / str(
            config["train"]["best_checkpoint"]
        )
    return resolve_path(checkpoint, config["_meta"]["root"])


def predict_image(
    config: dict[str, Any],
    image_path: str | Path,
    checkpoint: str | Path | None = None,
) -> dict[str, str]:
    """Predict a mask for one image and write ``_mask.png``/``_overlay.png``."""

    device = torch.device(resolve_device(config))
    checkpoint_path = _checkpoint_path(config, checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    path = resolve_path(image_path, config["_meta"]["root"])
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")

    data = config["data"]
    height, width = (int(value) for value in data["image_size"])
    channels = int(data["channels"])
    mode = "RGB" if channels == 3 else "L"
    with Image.open(path) as raw_image:
        image = raw_image.convert(mode).resize((width, height), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 3:
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        tensor = torch.from_numpy(array).unsqueeze(0)
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)

    model = load_checkpoint(checkpoint_path, config, device)
    threshold = float(config["infer"].get("threshold", 0.5))
    with torch.no_grad():
        logits = model(tensor)
    probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probabilities >= threshold).astype(np.uint8) * 255

    output_dir = config_path(config, "infer.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{path.stem}_mask.png"
    overlay_path = output_dir / f"{path.stem}_overlay.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    overlay = rgb.copy()
    selected = mask > 0
    overlay[selected] = 0.5 * rgb[selected] + 0.5 * np.asarray([1.0, 0.0, 0.0])
    Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB").save(
        overlay_path
    )

    print(f"mask: {mask_path}")
    print(f"overlay: {overlay_path}")
    return {"mask": str(mask_path), "overlay": str(overlay_path)}


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
