# Troubleshooting

## Common checks

- Validate config: `python -m src.config --show`
- Syntax check: `python -m compileall -q src`
- Fast tests: `python -m pytest tests/test_config.py -q`
- Full suite: `python -m pytest -q`

## Environment and dependencies

- Install runtime and dev tools with `python -m pip install -r requirements-dev.txt`.
- `torch` must be importable before running any stage; `python -m src.config --show`
  only needs PyYAML.
- On Windows, quote paths that contain spaces.

## Dataset generation and alignment

- Images and masks must share the same filename stem under `data/images/` and
  `data/masks/`; a missing mask raises `DataError`.
- `python -m src.data --force` clears stale `*.png` files in the affected split
  directories before regenerating.
- The Colab notebook downloads `Aeoo/CVC-ClinicDB` from Hugging Face. No auth is
  required; retry the download cell if the Hub request fails.
- CVC-ClinicDB masks may be `{0, 1}` or `{0, 255}`; the notebook binarizes both
  to `{0, 255}` and clears stale PNGs before writing.

## Training

- `data.image_size` must be divisible by `2 ** model.depth` (default `16`).
- BatchNorm needs at least one spatial element per sample; use `image_size` and
  `batch_size` that are not both tiny.
- CUDA out of memory: lower `train.batch_size` or `model.base_filters`.
- Early stopping before `train.epochs` is expected when validation loss stalls
  for `train.patience` epochs.
- Resume from `models/last.pt`; checkpoints include optimizer and scheduler
  state, so the LR schedule continues correctly.
- If a run is interrupted, `models/last.pt` and `results/train_history.json`
  contain the latest completed epoch.

## Evaluation and inference

- Evaluation defaults to `models/best.pt`; pass `--checkpoint` to score another
  file.
- `results/eval/metrics.json` contains `split`, `checkpoint`, `num_samples`, and
  the stable metric keys `iou`, `dice`, `pixel_accuracy`.
- Overlay triptychs are written for the first `eval.num_overlays` samples.
- Prediction writes `<stem>_mask.png` and `<stem>_overlay.png` to
  `infer.output_dir`; relative image paths resolve against the repository root.
- A missing checkpoint or image raises `FileNotFoundError` before inference.

## Colab runtime

- Select `Runtime -> Change runtime type -> T4 GPU`; CPU runs are supported but
  slower.
- If the repository is not found, set `REPO_URL` or upload a project zip to
  `/content`.
- The notebook downloads CVC-ClinicDB into `/content` and writes
  `/content/colab_unet.yaml`; edit the config cell rather than the tracked
  `config/unet.yaml`.
- Export the results zip before the runtime disconnects; weights are not tracked
  in git.
