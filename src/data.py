"""Synthetic image/mask dataset generation, preprocessing, and dataloaders.

The generator is deterministic: every pair is rendered from a seed derived from
``project.seed`` plus a per-split offset and the pair index. Images and masks are
written under separate trees with identical filenames so they stay aligned
pixel-for-pixel. Dataloaders resize to ``data.image_size``, normalize images to
``[-1, 1]``, binarize masks, and apply flip augmentation to the training split.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Dataset

from src.config import DEFAULT_CONFIG_PATH, load_config, resolve_path

SPLITS = ("train", "val", "test")
_KINDS = ("images", "masks")
_SPLIT_SEED_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000}
_SUPPORTED_CHANNELS = (1, 3)
_NORMALIZE_MEAN = 0.5
_NORMALIZE_STD = 0.5


class DataError(RuntimeError):
    """Raised when dataset generation or loading fails."""


def split_dir(config: dict[str, Any], split: str, kind: str = "images") -> Path:
    """Return ``<data.root>/<kind>/<split>`` as an absolute path."""

    if split not in SPLITS:
        raise DataError(f"unknown split: {split}")
    if kind not in _KINDS:
        raise DataError(f"unknown data kind: {kind}")
    data = config["data"]
    subdir = data.get("image_dir", "images") if kind == "images" else data.get("mask_dir", "masks")
    root = resolve_path(data["root"], config["_meta"]["root"])
    return root / subdir / split


def pair_stem(index: int) -> str:
    """Return the shared filename stem for the pair at ``index``."""

    return f"img_{index:03d}"


def _rng(base_seed: int, split: str, index: int) -> np.random.Generator:
    offset = _SPLIT_SEED_OFFSET[split]
    return np.random.default_rng(base_seed + offset + index)


def _color(rng: np.random.Generator, low: int, high: int, channels: int) -> Any:
    values = rng.integers(low, high, size=channels)
    if channels == 1:
        return int(values[0])
    return tuple(int(value) for value in values)


def _synthetic_pair(
    rng: np.random.Generator, size: tuple[int, int], channels: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = size
    mode = "RGB" if channels == 3 else "L"
    image = Image.new(mode, (width, height), _color(rng, 0, 64, channels))
    mask = Image.new("L", (width, height), 0)
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    for _ in range(int(rng.integers(1, 4))):
        radius_x = int(rng.integers(max(2, width // 12), max(3, width // 3)))
        radius_y = int(rng.integers(max(2, height // 12), max(3, height // 3)))
        center_x = int(rng.integers(0, width))
        center_y = int(rng.integers(0, height))
        box = (center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y)
        color = _color(rng, 120, 256, channels)
        if int(rng.integers(0, 2)) == 0:
            image_draw.ellipse(box, fill=color)
            mask_draw.ellipse(box, fill=255)
        else:
            image_draw.rectangle(box, fill=color)
            mask_draw.rectangle(box, fill=255)

    image_array = np.asarray(image, dtype=np.uint8).copy()
    noise = rng.integers(-8, 9, size=image_array.shape, dtype=np.int16)
    image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    mask_array = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return image_array, mask_array


def generate_synthetic_dataset(
    config: dict[str, Any], force: bool = False
) -> dict[str, int]:
    """Generate deterministic image/mask pairs for every configured split."""

    data = config["data"]
    size = tuple(int(value) for value in data["image_size"])
    channels = int(data["channels"])
    if channels not in _SUPPORTED_CHANNELS:
        raise DataError(
            f"data.channels must be one of {_SUPPORTED_CHANNELS} for synthetic pairs"
        )
    base_seed = int(config["project"]["seed"])

    counts: dict[str, int] = {}
    for split, raw_count in data["splits"].items():
        if split not in SPLITS:
            raise DataError(f"unknown split: {split}")
        count = int(raw_count)
        image_dir = split_dir(config, split, "images")
        mask_dir = split_dir(config, split, "masks")
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        expected = [f"{pair_stem(index)}.png" for index in range(count)]
        complete = all(
            (image_dir / name).is_file() and (mask_dir / name).is_file()
            for name in expected
        )
        if not force and complete:
            counts[split] = count
            continue

        for stale in list(image_dir.glob("*.png")) + list(mask_dir.glob("*.png")):
            stale.unlink()

        for index in range(count):
            image_array, mask_array = _synthetic_pair(_rng(base_seed, split, index), size, channels)
            name = f"{pair_stem(index)}.png"
            Image.fromarray(image_array).save(image_dir / name)
            Image.fromarray(mask_array, mode="L").save(mask_dir / name)
        counts[split] = count

    return counts


def list_pairs(config: dict[str, Any], split: str) -> list[tuple[Path, Path]]:
    """Return sorted ``(image, mask)`` path pairs for ``split``.

    Raises :class:`DataError` if a mask is missing for an image, which would
    break the pixel-for-pixel alignment invariant.
    """

    image_dir = split_dir(config, split, "images")
    mask_dir = split_dir(config, split, "masks")
    if not image_dir.is_dir():
        raise DataError(f"missing image directory: {image_dir}")

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            raise DataError(f"missing mask for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))

    if not pairs:
        raise DataError(f"no image/mask pairs found in {image_dir}")
    return pairs


class SegmentationDataset(Dataset):
    """Image/mask pairs resized, normalized, and optionally flipped."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        image_size: tuple[int, int],
        channels: int,
        augment: bool = False,
    ) -> None:
        if channels not in _SUPPORTED_CHANNELS:
            raise DataError(f"data.channels must be one of {_SUPPORTED_CHANNELS}")
        self.pairs = list(pairs)
        self.image_size = (int(image_size[1]), int(image_size[0]))
        self.channels = channels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        mode = "RGB" if self.channels == 3 else "L"

        with Image.open(image_path) as raw_image:
            image = raw_image.convert(mode).resize(self.image_size, Image.BILINEAR)
        with Image.open(mask_path) as raw_mask:
            mask = raw_mask.convert("L").resize(self.image_size, Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            if random.random() < 0.5:
                image = ImageOps.flip(image)
                mask = ImageOps.flip(mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)

        if self.channels == 3:
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        else:
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        image_tensor = (image_tensor - _NORMALIZE_MEAN) / _NORMALIZE_STD
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor


def image_to_display(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized ``(C, H, W)`` tensor to an ``(H, W, 3)`` RGB array."""

    array = image.detach().cpu().float().clamp(-1.0, 1.0)
    array = (array + 1.0) / 2.0
    array = array.permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return np.clip(array, 0.0, 1.0)


def build_dataloader(
    config: dict[str, Any],
    split: str,
    batch_size: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """Build a dataloader for one split with the configured preprocessing."""

    if split not in SPLITS:
        raise DataError(f"unknown split: {split}")

    data = config["data"]
    image_size = tuple(int(value) for value in data["image_size"])
    channels = int(data["channels"])
    if channels not in _SUPPORTED_CHANNELS:
        raise DataError(f"data.channels must be one of {_SUPPORTED_CHANNELS}")

    dataset = SegmentationDataset(
        list_pairs(config, split),
        image_size=image_size,
        channels=channels,
        augment=split == "train",
    )
    if batch_size is None:
        batch_size = int(config["train"]["batch_size"])
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def build_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from the generated dataset."""

    return {split: build_dataloader(config, split) for split in SPLITS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data",
        description="Generate or inspect the synthetic segmentation dataset.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file (default: config/unet.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the dataset even if files already exist.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    counts = generate_synthetic_dataset(config, force=args.force)
    for split in SPLITS:
        if split in counts:
            print(f"{split}: {counts[split]} pairs")
    print(f"dataset root: {split_dir(config, 'train', 'images').parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
