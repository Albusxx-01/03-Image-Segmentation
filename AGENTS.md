# AGENTS.md

Project: `03-Image-Segmentation` — U-Net binary segmentation, release `v0.1.0`.

## Environment

- Python 3.13, PyTorch 2.x CPU locally; Colab GPU only for full training.
- Install: `python -m pip install -r requirements-dev.txt`
- Config is the single source of truth: `config/unet.yaml`.

## Commands

```bash
python -m pytest -q                                  # full suite
python -m pytest tests/test_config.py -q             # fast config check
python -m compileall -q src                          # syntax check
python -m src.config --show                          # validate config
python -m src.data --config config/unet.yaml --force
python -m src.train --config config/unet.yaml
python -m src.evaluate --config config/unet.yaml
python -m src.predict --config config/unet.yaml --image <path>
python -m src.app --config config/unet.yaml
```

Run the check listed for the current commit in `docs/commit-plan.md` before
committing. Never commit with a failing check.

## Conventions

- Python files `snake_case.py`; functions/variables `snake_case`; constants
  `UPPER_SNAKE_CASE`; classes `PascalCase`.
- Commit format: `<type>(<scope>): <imperative summary>`.
  Types: `feat`, `fix`, `docs`, `test`, `refactor`, `exp`, `chore`.
  Scopes: `data`, `model`, `train`, `eval`, `infer`, `app`, `notebook`,
  `config`, `deps`, `repo`, `release`.
- One logical change per commit; no mixed commits. Work lands directly on `main`.
- Config values are read through `src.config.load_config`; paths resolve against
  the repository root.
- No comments in code unless they explain non-obvious constraints.

## Never commit

- `data/`, `models/`, `results/` run outputs (only curated `results/v0.1.0/`).
- Weights/checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.onnx`).
- Caches, notebooks checkpoints, videos, secrets or `.env` files.
- Backdated commits; use actual author dates.

## Architecture guardrails

- Images and masks use identical filenames/spatial transforms; masks align
  pixel-for-pixel.
- U-Net output is `(B, 1, H, W)` logits for binary segmentation.
- Metrics keys are stable: `iou`, `dice`, `pixel_accuracy`.
- Evaluation writes `results/eval/metrics.json`; prediction writes
  `<stem>_mask.png` and `<stem>_overlay.png`.
