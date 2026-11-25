# Contracts

Stable interfaces for every pipeline stage. This file grows with the release;
each stage documents its CLI flags, inputs, outputs, and invariants here.

> Scaffold stub. Dataset, model, loss/metric, training, evaluation, inference,
> and app contracts are filled in by their respective commits.

## Configuration

- `config/unet.yaml` is the single source of truth.
- Paths are resolved relative to the repository root.
- Validation fails fast with `ConfigError` for missing sections or invalid
  values.

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
