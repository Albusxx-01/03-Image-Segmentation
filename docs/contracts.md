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

## Model

`UNet` in `src/model.py` is a symmetric encoder-decoder with concatenative skip
connections.

- Encoder: `model.depth` levels. Each level is two 3x3 convolutions with batch
  norm and ReLU, followed by 2x2 max pooling. Level `l` (0-based) uses
  `model.base_filters * 2**l` channels.
- Bottleneck: one `ConvBlock` at `model.base_filters * 2**model.depth`.
- Decoder: one `ConvTranspose2d(kernel=2, stride=2)` per level. The matching
  encoder feature map is concatenated before a `ConvBlock` that returns to the
  skip channel count.
- Head: 1x1 convolution mapping to `model.out_channels` logits.

Invariants:

- Input is `(B, model.in_channels, H, W)`; output is
  `(B, model.out_channels, H, W)` with raw logits (no sigmoid).
- `H` and `W` must be divisible by `2 ** model.depth`; otherwise the forward pass
  raises `ValueError`.
- `depth >= 1`, `base_filters > 0`, and channel counts must be positive.
- `model.dropout > 0` appends `Dropout2d` to every `ConvBlock`.

```bash
python -m src.model --config config/unet.yaml --sanity
```

With `--sanity`, the CLI runs a forward pass on a zero tensor of
`data.image_size` and prints the output shape; without it, it prints the module.

| Key | Meaning | v0.1.0 |
|-----|---------|--------|
| `model.in_channels` | input image channels | `3` |
| `model.out_channels` | logit channels (`1` for binary) | `1` |
| `model.base_filters` | first-level channel count | `32` |
| `model.depth` | encoder/decoder levels | `4` |
| `model.dropout` | spatial dropout after each block | `0.0` |

## Metrics

`src/metrics.py` scores raw logits against binary targets. Both sides are
binarized: predictions use `sigmoid(logits) >= eval.threshold` (default `0.5`)
and targets use `>= 0.5`, so `{0, 1}` and `{0, 255}` masks are both accepted.
Every metric aggregates over all elements of the batch (micro-average).

Let `P` be the binarized prediction set and `T` the binarized target set:

| Key | Formula | Empty case (`|P| = |T| = 0`) |
|-----|---------|------------------------------|
| `iou` | `|P n T| / |P u T|` | `1.0` |
| `dice` | `2|P n T| / (|P| + |T|)` | `1.0` |
| `pixel_accuracy` | `|P = T| / N` | `1.0` |

- `compute_metrics(logits, targets, threshold)` returns exactly these three keys;
  they are the stable schema written to `results/eval/metrics.json`.
- Targets may be `(B, C, H, W)` or `(B, H, W)` when `C == 1`; any other mismatch
  raises `ValueError`.
- `threshold` must be in `[0, 1]`; out-of-range values raise `ValueError`.

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
