# 03-Image-Segmentation (U-Net)

Binary semantic segmentation with a U-Net encoder-decoder. Local reference runs
use deterministic synthetic image/mask pairs; the Colab notebook trains on
CVC-ClinicDB.

## Status

Training, evaluation, and single-image inference are implemented for `v0.1.0`.
The static gallery remains planned. The Colab notebook
[`notebooks/03-colab-train.ipynb`](notebooks/03-colab-train.ipynb) downloads
CVC-ClinicDB from Hugging Face and runs the full data -> train -> evaluate ->
predict -> export pipeline on a GPU runtime.

## Quickstart

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_config.py -q
python -m src.config --show
```

Generate data, train, evaluate, and predict:

```bash
python -m src.data --config config/unet.yaml --force
python -m src.train --config config/unet.yaml
python -m src.evaluate --config config/unet.yaml
python -m src.predict --config config/unet.yaml --image data/images/test/img_000.png
```

## Colab

Open [`notebooks/03-colab-train.ipynb`](notebooks/03-colab-train.ipynb) in Colab,
select a GPU runtime, and run the cells top to bottom. It downloads
`Aeoo/CVC-ClinicDB`, converts it to the project layout, then trains, evaluates,
predicts, and exports a zip of weights and results.

## Dataset

The Colab notebook uses CVC-ClinicDB (binary polyp segmentation, 612 pairs) from
Hugging Face:
[`Aeoo/CVC-ClinicDB`](https://huggingface.co/datasets/Aeoo/CVC-ClinicDB).
Local runs can continue with the deterministic synthetic pairs.

## Pipeline

```
image/mask pairs -> U-Net (skip connections) -> BCE/Dice loss
      -> train/val with checkpoints -> test IoU/Dice -> masks + overlays
      -> static HTML gallery
```

## Layout

```
config/unet.yaml        paths, seeds, model/training hyperparameters
src/config.py           config loading, validation, seeding
src/data.py             pair generation and dataloaders
src/model.py            U-Net
src/losses.py           BCE, Dice, combined loss
src/metrics.py          IoU, Dice, pixel accuracy
src/train.py            training loop, checkpoints, early stopping
src/evaluate.py         metrics JSON and overlays
src/predict.py          single-image inference
src/app.py              static gallery
tests/                  unit and smoke tests
docs/                   contracts, reproducibility, troubleshooting, plans
```

Only curated artifacts under `results/v0.1.0/` are tracked. Datasets, weights,
and run folders never enter git.

## Documentation

- [`docs/commit-plan.md`](docs/commit-plan.md) — schedule and release checklist
- [`docs/contracts.md`](docs/contracts.md) — stable CLI and artifact contracts
- [`docs/reproducibility.md`](docs/reproducibility.md) — versions, seeds, run order
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common failures
- [`learnings/README.md`](learnings/README.md) — concept notes
