# Reproducibility

## Reference environment

Local smoke runs (CPU):

| Component | Version |
|-----------|---------|
| Python | 3.13.0 |
| torch | 2.14.1+cpu |
| numpy | 2.3.5 |
| PyYAML | 6.0.3 |
| Pillow | 12.1.0 |
| matplotlib | 3.10.8 |

Colab GPU runs use the runtime's preinstalled torch/CUDA stack and the
`datasets` package installed by the notebook. Record the exact Colab versions in
the release notes when they differ from the table above.

## Dataset

- Local reference runs use the deterministic synthetic pairs from `src.data`.
- The Colab notebook downloads `Aeoo/CVC-ClinicDB` (binary polyp segmentation,
  612 pairs) from Hugging Face. The dataset's fixed `train`/`validation`/`test`
  splits are preserved; pairs are resized to `data.image_size` and masks are
  binarized to `{0, 255}`.
- No authentication is required for the dataset download.

## Seeds and determinism

- `project.seed` (default `42`) seeds Python, NumPy, and torch through
  `src.config.set_seed`.
- Synthetic pairs derive their RNG from `project.seed` plus a fixed per-split
  offset and the pair index, so regeneration is byte-for-byte deterministic.
- CVC-ClinicDB pairs are downloaded once per Colab runtime and written in a
  fixed order; the notebook clears stale PNGs before writing.
- CUDA training enables cuDNN deterministic mode. AMP and GPU kernel scheduling
  can still produce small metric differences between runs and machines.

## Run order

```bash
python -m src.data --config config/unet.yaml --force
python -m src.train --config config/unet.yaml
python -m src.evaluate --config config/unet.yaml --split test
python -m src.predict --config config/unet.yaml --image data/images/test/img_000.png
```

Resume training with `python -m src.train --config config/unet.yaml --resume
models/last.pt`.

## Expected artifacts

| Path | Producer | Contents |
|------|----------|----------|
| `data/{images,masks}/{train,val,test}/` | `src.data` or Colab notebook | deterministic synthetic pairs or CVC-ClinicDB pairs |
| `models/best.pt` | `src.train` | best validation-loss checkpoint |
| `models/last.pt` | `src.train` | latest checkpoint |
| `results/train_history.json` | `src.train` | per-epoch losses, metrics, LR |
| `results/eval/metrics.json` | `src.evaluate` | split, checkpoint, sample count, `iou`, `dice`, `pixel_accuracy` |
| `results/eval/<stem>_overlay.png` | `src.evaluate` | input/ground-truth/prediction triptychs |
| `results/predictions/<stem>_mask.png` | `src.predict` | binary mask |
| `results/predictions/<stem>_overlay.png` | `src.predict` | red foreground overlay |

Only curated `results/v0.1.0/` artifacts are tracked. Datasets, weights, and
run folders stay out of git.

## Colab workflow

1. Open `notebooks/03-colab-train.ipynb` and select a GPU runtime.
2. Set `REPO_URL` or upload the project zip.
3. Run the CVC-ClinicDB download/preprocess cell (Hugging Face, no auth).
4. Run the config cell, then the train/evaluate/predict cells.
5. Export the generated zip or download it.

The notebook writes `/content/colab_unet.yaml` and passes it to every CLI, so
the run is reproducible without editing the tracked config. It overrides
`train.epochs`, `train.batch_size`, `train.patience`, and `eval.num_overlays`
for GPU runs; adjust those values for your hardware.

## Deviations to record

- Colab torch/CUDA versions, `datasets` version, and GPU model.
- CVC-ClinicDB revision and the number of pairs written per split.
- Epochs run before early stopping and the best validation loss.
- Any difference between local CPU smoke metrics and Colab GPU metrics.
