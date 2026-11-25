# Contracts

Stable interfaces for every pipeline stage. This file grows with the release;
each stage documents its CLI flags, inputs, outputs, and invariants here.

> Dataset contracts are filled in by Day 2. Model, loss/metric, training,
> evaluation, inference, and app contracts are filled in by their respective
> commits.

## Configuration

- `config/unet.yaml` is the single source of truth.
- Paths are resolved relative to the repository root.
- Validation fails fast with `ConfigError` for missing sections or invalid
  values.

## Data

Generate the deterministic synthetic dataset with:

```bash
python -m src.data --config config/unet.yaml --force
```

- Root: `data.root` from the config (`data/` by default), resolved against the
  repository root.
- Layout: `data/images/{train,val,test}/img_NNN.png` and
  `data/masks/{train,val,test}/img_NNN.png`.
- Pairing: images and masks share the identical filename and index, so a mask is
  always found at the same relative path under `masks/`. Missing masks raise
  `DataError` instead of silently misaligning.
- Counts: `data.splits` (`64/16/16` for `v0.1.0`) defines pairs per split.
- Images: `data.channels` (1 grayscale or 3 RGB) x `data.image_size`, `uint8`.
- Masks: single-channel `uint8` with values `{0, 255}`; foreground is `255`.
- Determinism: each pair is rendered from `project.seed` plus a fixed per-split
  offset and the pair index. Re-running with `--force` reproduces identical
  bytes; the same seed always yields the same pairs.
- `--force` clears stale `*.png` files in the affected split directories before
  regenerating, so a changed count never leaves orphan pairs.
- Python API: `generate_synthetic_dataset(config, force=False)` returns a
  `{split: count}` mapping; `list_pairs(config, split)` returns sorted
  `(image_path, mask_path)` tuples.

## Planned contracts

| Stage | Command | Primary artifacts |
|-------|---------|-------------------|
| Data | `python -m src.data --config ... --force` | `data/{images,masks}/{train,val,test}/` |
| Model | `python -m src.model --config ... --sanity` | `(B, 1, H, W)` logits |
| Train | `python -m src.train --config ...` | `models/best.pt`, `models/last.pt`, `results/train_history.json` |
| Evaluate | `python -m src.evaluate --config ...` | `results/eval/metrics.json`, overlays |
| Predict | `python -m src.predict --config ... --image ...` | `<stem>_mask.png`, `<stem>_overlay.png` |
| App | `python -m src.app --config ...` | static HTML gallery |

Metric keys: `iou`, `dice`, `pixel_accuracy`.
