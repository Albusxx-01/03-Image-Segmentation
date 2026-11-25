# Reproducibility

> Scaffold stub. Versions, seeds, run order, and expected artifacts are recorded
> here before the `v0.1.0` release.

## Principles

- Fixed seed from `project.seed` in `config/unet.yaml`.
- Deterministic synthetic data generation.
- Local CPU runs are the reference for smoke checks; Colab GPU runs use the same
  config and CLIs.
- Track only curated artifacts under `results/v0.1.0/`.

## To document during the release

- Python and package versions (`torch`, `numpy`, `PyYAML`, `Pillow`, `matplotlib`).
- Exact command order: data -> train -> evaluate -> predict -> app.
- Expected artifacts and their locations.
- Any deviation between local and Colab results.
