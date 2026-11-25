# Commit Plan — 03-Image-Segmentation v0.1.0

Status: planned
Window: Mon 2026-11-30 through Sun 2026-12-06
Target release: `v0.1.0`
Primary stack: Python 3.13 local (PyTorch 2.14 CPU), Colab GPU for full training
Dataset for v0.1.0: deterministic synthetic image/mask pairs

This plan is a schedule, not a quota. Commit author dates must be the actual dates.
Never backdate commits. If a day slips, move the remaining commits forward; never
combine unrelated changes to catch up.

## Trunk model

- `main` is the only long-lived branch and the integration trunk.
- All work lands directly on `main`; there is no `develop` branch.
- Release is an annotated tag on `main`.
- No routine merge commits. Optional short-lived `exp/<name>` branches may be used
  for risky experiments, but the planned commits below go straight to `main`.
- Never force-push or rewrite published history; fix forward.

## Weekly summary

| Day | Date | Theme | Commits | Milestone |
|-----|------|-------|---------|-----------|
| Mon | 2026-11-30 | Scaffold | 1 | M0, M1 |
| Tue | 2026-12-01 | Data + model core | 5 | M2, M3 |
| Wed | 2026-12-02 | Model close + losses/metrics | 5 | M3, M4 |
| Thu | 2026-12-03 | Training | 4 | M5 |
| Fri | 2026-12-04 | Evaluation + inference | 7 | M6, M7 |
| Sat | 2026-12-05 | App + smoke + docs | 7 | M8, M9, M10 |
| Sun | 2026-12-06 | Notebooks + Colab + release | 5 + tag | M10, M11, M12 |
| | | **Total** | **34 + tag** | |

## Commit rules

- Format: `<type>(<scope>): <imperative summary>`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `exp`, `chore`
- Scopes: `data`, `model`, `train`, `eval`, `infer`, `app`, `notebook`, `config`, `deps`, `repo`, `release`
- One logical change per commit. No mixed commits. No `fix stuff`.
- Order within a feature: smallest working slice → next slice → tests/smoke → docs.
- Never commit `data/`, weights/checkpoints, run folders, videos, caches, or secrets.
- Only curated `results/v0.1.0/` artifacts may be tracked.
- Run the listed check before each commit; do not commit with a failing check.

## Bootstrap commands (before commit 1)

```bash
git init -b main
```

The scaffold commit is the first commit on `main`.

## Day 1 — Mon 2026-11-30 — Scaffold

| # | Branch | Exact commit message | Files | Checks |
|---|--------|----------------------|-------|--------|
| 1 | `main` | `chore(repo): initialize 03-image-segmentation scaffold` | `.gitignore`, `.gitattributes`, `README.md`, `CHANGELOG.md`, `VERSION`, `AGENTS.md`, `requirements.txt`, `requirements-dev.txt`, `config/unet.yaml`, `src/__init__.py`, `src/config.py`, `src/data.py`, `src/model.py`, `src/losses.py`, `src/metrics.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`, `src/app.py`, `tests/__init__.py`, `tests/test_config.py`, `docs/commit-plan.md`, `docs/contracts.md` (stub), `docs/reproducibility.md` (stub), `docs/troubleshooting.md` (stub), `docs/pre-build/*`, `learnings/*`, `notebooks/.gitkeep`, `data/.gitkeep`, `models/.gitkeep`, `results/.gitkeep`, `results/README.md` | `python -m pytest tests/test_config.py -q`; `python -m compileall -q src`; `git status --short` |

Definition of done: config loads, all stage modules import, no data/weights tracked,
CLI stubs expose `main()` and `--help`.

## Day 2 — Tue 2026-12-01 — Data + model core

| # | Branch | Exact commit message | Files | Checks |
|---|--------|----------------------|-------|--------|
| 2 | `main` | `feat(data): generate synthetic image-mask pairs` | `src/data.py` | `python -m src.data --config config/unet.yaml --force` |
| 3 | `main` | `test(data): verify image-mask pairing and splits` | `tests/test_data.py` | `python -m pytest tests/test_data.py -q` |
| 4 | `main` | `docs(data): document dataset layout and generation` | `docs/contracts.md` | Markdown review |
| 5 | `main` | `feat(model): implement U-Net encoder-decoder with skip connections` | `src/model.py` | `python -m src.model --config config/unet.yaml --sanity` |
| 6 | `main` | `test(model): check forward-pass shapes and depth guards` | `tests/test_model.py` | `python -m pytest tests/test_model.py -q` |

Definition of done: 64/16/16 synthetic pairs align pixel-for-pixel; model forward
returns `(B, 1, H, W)`.

## Day 3 — Wed 2026-12-02 — Model close + losses/metrics

| # | Branch | Exact commit message | Files | Checks |
|---|--------|----------------------|-------|--------|
| 7 | `main` | `docs(model): document architecture and config knobs` | `docs/contracts.md` | Markdown review |
| 8 | `main` | `feat(model): add BCE, Dice, and combined losses` | `src/losses.py` | `python -m compileall -q src`; import check |
| 9 | `main` | `feat(eval): add IoU, Dice, and pixel-accuracy metrics` | `src/metrics.py` | `python -m compileall -q src`; import check |
| 10 | `main` | `test(model): verify losses and metrics on hand-computed fixtures` | `tests/test_losses.py`, `tests/test_metrics.py` | `python -m pytest tests/test_losses.py tests/test_metrics.py -q` |
| 11 | `main` | `docs(eval): document metric keys and formulas` | `docs/contracts.md` | Markdown review |

Definition of done: perfect/disjoint/half-overlap fixtures match hand-computed
values; losses are finite and differentiable.

## Day 4 — Thu 2026-12-03 — Training

| # | Branch | Exact commit message | Files | Checks |
|---|--------|----------------------|-------|--------|
| 12 | `main` | `feat(train): add seeded training loop with checkpoints` | `src/train.py` | `python -m compileall -q src`; `python -m src.train --help` |
| 13 | `main` | `feat(train): add LR scheduling and early stopping` | `src/train.py` | `python -m src.train --help` |
| 14 | `main` | `test(train): smoke-train one epoch on toy data` | `tests/test_smoke_pipeline.py` | `python -m pytest tests/test_smoke_pipeline.py -q` |
| 15 | `main` | `docs(train): document training CLI and artifacts` | `docs/contracts.md` | Markdown review |

Definition of done: one tiny CPU epoch writes `models/best.pt`,
`models/last.pt`, and `results/train_history.json`; loss is finite.

## Day 5 — Fri 2026-12-04 — Evaluation + inference

| # | Branch | Exact commit message | Files | Checks |
|---|--------|----------------------|-------|--------|
| 16 | `main` | `feat(eval): evaluate checkpoints and write metrics JSON` | `src/evaluate.py` | `python -m compileall -q src`; `python -m src.evaluate --help` |
| 17 | `main` | `feat(eval): render input-groundtruth-prediction overlays` | `src/evaluate.py` | `python -m src.evaluate --help` |
| 18 | `main` | `test(eval): assert metrics schema and overlay outputs` | `tests/test_eval.py` | `python -m pytest tests/test_eval.py -q` |
| 19 | `main` | `docs(eval): document evaluation contract` | `docs/contracts.md` | Markdown review |
| 20 | `main` | `feat(infer): predict masks and overlays from checkpoints` | `src/predict.py` | `python -m compileall -q src`; `python -m src.predict --help` |
| 21 | `main` | `test(infer): smoke-test single-image inference` | `tests/test_infer.py` | `python -m pytest tests/test_infer.py -q` |
| 22 | `main` | `docs(infer): document inference flags and outputs` | `docs/contracts.md` | Markdown review |

Definition of done: `results/eval/metrics.json` and overlays exist; a checkpoint
produces `<stem>_mask.png` and `<stem>_overlay.png`.

## Day 6 — Sat 2026-12-05 — App + smoke + docs

| # | Branch | Exact commit message | Files | Checks |
|---|--------|----------------------|-------|--------|
| 23 | `main` | `feat(app): build static HTML prediction gallery` | `src/app.py` | `python -m src.app --help` |
| 24 | `main` | `docs(app): document gallery usage` | `docs/contracts.md` | Markdown review |
| 25 | `main` | `test(repo): add end-to-end smoke script` | `scripts/smoke.py` | `python scripts/smoke.py` |
| 26 | `main` | `docs(repo): document project checks` | `README.md`, `docs/reproducibility.md` | Markdown review |
| 27 | `main` | `docs(repo): write README overview, quickstart, and pipeline` | `README.md` | Markdown review |
| 28 | `main` | `docs(repo): add reproducibility and troubleshooting guides` | `docs/reproducibility.md`, `docs/troubleshooting.md` | Markdown review |
| 29 | `main` | `docs(repo): add contracts reference` | `docs/contracts.md` | Markdown review |

Definition of done: static gallery renders prediction pairs; smoke runs
data → train → eval → predict → app on CPU; README quickstart is complete.

## Day 7 — Sun 2026-12-06 — Notebooks + Colab + release

| # | Branch | Exact commit message | Files | Checks |
|---|--------|----------------------|-------|--------|
| 30 | `main` | `docs(notebook): add data exploration and error analysis notebooks` | `notebooks/01-data-exploration.ipynb`, `notebooks/02-error-analysis.ipynb` | `python -m json.tool` on both notebooks |
| 31 | `main` | `feat(notebook): add Colab training notebook` | `notebooks/03-colab-train.ipynb` | `python -m json.tool` on notebook |
| 32 | `main` | `docs(notebook): document Colab workflow` | `docs/reproducibility.md`, `README.md` | Markdown review |
| 33 | `main` | `chore(release): bump version to 0.1.0` | `VERSION` | `python -m pytest -q` |
| 34 | `main` | `docs(release): update changelog for v0.1.0` | `CHANGELOG.md`, `results/v0.1.0/metrics.json`, curated `results/v0.1.0/*.png` | `python scripts/smoke.py`; artifact audit |
| 35 | `main` | tag `v0.1.0` (annotated) | none | `git tag -n`; `git log --oneline --graph -20` |

Definition of done: tag `v0.1.0` points at commit 34, CHANGELOG has `[0.1.0]`,
and a fresh clone can reproduce the toy run.

## Checks by day

| Day | Required checks |
|-----|-----------------|
| 1 | `python -m pytest tests/test_config.py -q`; `python -m compileall -q src`; `git status --short` |
| 2 | `python -m pytest tests/test_data.py tests/test_model.py -q`; `python -m src.data --force` |
| 3 | `python -m pytest tests/test_losses.py tests/test_metrics.py -q` |
| 4 | `python -m pytest tests/test_smoke_pipeline.py -q`; `python -m src.train --help` |
| 5 | `python -m pytest tests/test_eval.py tests/test_infer.py -q` |
| 6 | `python scripts/smoke.py`; `python -m pytest -q`; notebook JSON validation |
| 7 | `python scripts/smoke.py`; `python -m pytest -q`; release checklist; artifact audit |

## Colab dependency

`notebooks/03-colab-train.ipynb` is built and syntax-validated locally, but Colab
execution is performed by the project owner on a GPU runtime. The notebook only
invokes CLIs; it never imports `src/`.

Decision point before commit 34:

- **Option A (default):** tag `v0.1.0` using local CPU smoke metrics. Record the
  real Colab run afterward as `docs(repo): record Colab v0.1.0 results` and release
  `v0.1.1` if metrics need to be part of a tagged release.
- **Option B:** pause before commit 34 until the Colab run produces
  `results/v0.1.0/metrics.json`; then include those metrics in the v0.1.0 changelog.

## Release checklist

- [ ] `VERSION` is `0.1.0`
- [ ] `CHANGELOG.md` has a dated `[0.1.0]` entry (Keep a Changelog + SemVer)
- [ ] `docs/reproducibility.md` records versions, seeds, run order, expected artifacts
- [ ] `python scripts/smoke.py` passes
- [ ] `python -m pytest -q` passes
- [ ] Final results recorded in `README.md` and `CHANGELOG.md`
- [ ] Artifact audit: no data, weights, run dirs, videos, caches, or secrets tracked
- [ ] Annotated tag `v0.1.0` created on commit 34
- [ ] `git status --short` is clean

## Slip policy

1. Finish the current commit, run its check, then stop for the day.
2. Move unfinished commits to the next day in the same order.
3. Never commit to `main` with failing checks; fix forward on the current branch.
4. Never rewrite published history; add a new `fix(...)` commit instead.
5. If Colab is unavailable, use Option A and keep the release moving.

## Progress tracker

- [ ] Day 1 — scaffold
- [ ] Day 2 — data + model core
- [ ] Day 3 — model close + losses/metrics
- [ ] Day 4 — training
- [ ] Day 5 — evaluation + inference
- [ ] Day 6 — app + smoke + docs
- [ ] Day 7 — notebooks + Colab + release
