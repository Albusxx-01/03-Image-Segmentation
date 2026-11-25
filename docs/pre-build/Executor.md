# Image Segmentation (U-Net) — Executor

## Naming Rules
- **Python files**: `snake_case.py` (e.g., `metrics.py`, `train.py`)
- **Functions/variables**: `snake_case` (e.g., `dice_coefficient()`, `val_iou`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `INPUT_SIZE`, `N_CLASSES`)
- **Classes**: `PascalCase` (e.g., `UNetBlock`, `DiceLoss`)
- **Mask files**: same base name as image + `_mask` suffix (`img_001.png`, `mask_001.png`) OR separate masks/ tree
- **Configs**: lowercase yaml (`unet.yaml`)
- **Checkpoints**: `unet_ep<epoch>_iou<val>.pt` e.g., `unet_ep50_iou0.79.pt`
- **Results plots**: `train_history.png`, `overlay_batch.png`, `class_iou.csv`
- **Notebooks**: zero-padded prefixes (`02-error-analysis.ipynb`)

## Milestones (bite-sized tasks)

### Milestone 0 — Setup (0.5-1 day)
- [ ] Create folder structure (data, config, src, notebooks, models, results)
- [ ] Virtual environment + `requirements.txt`
- [ ] Choose framework (PyTorch or Keras)
- **Definition of done**: skeleton imports and empty config load without errors.

### Milestone 1 — Dataset (2-3 days)
- [ ] Acquire dataset (Carvana, medical, or satellite imagery with masks)
- [ ] Inspect masks: value distribution, resolution, class balance
- [ ] Write `dataset.py` returning (image, mask) pairs with identical resize
- [ ] Split into train/val/test
- [ ] Visualize sample pairs (notebook)
- **Definition of done**: image and mask align pixel-for-pixel in overlays.

### Milestone 2 — Model (2-3 days)
- [ ] Write `model.py`: UNet encoder (4× Conv Block + MaxPool), bottleneck, decoder (4× UpConv + skip concat)
- [ ] Output: 1x1 Conv + Sigmoid (binary) / Softmax (multi-class)
- [ ] Sanity test: forward pass produces `(H, W, n_classes)` output
- **Definition of done**: model compiles and trains on a tiny toy batch.

### Milestone 3 — Losses + metrics (1-2 days)
- [ ] Implement `loss.py`: DiceLoss, BCE, CombinedLoss
- [ ] Implement `metrics.py`: IoU, Dice, pixel accuracy
- [ ] Verify on a fake binary mask pair
- **Definition of done**: metrics match hand-computed values on unit fixtures.

### Milestone 4 — Training (3-4 days)
- [ ] Augmentation: rotation, flip, elastic deformation (apply identically to image+mask)
- [ ] Compile/train with Adam lr=0.001 + LR scheduling
- [ ] Checkpoint + early stopping
- [ ] Plot loss/metric curves per epoch
- **Definition of done**: validation IoU steadily improves across epochs.

### Milestone 5 — Evaluation (2-3 days)
- [ ] Compute IoU, Dice, pixel accuracy on test set
- [ ] Side-by-side visualization: input vs GT vs prediction
- [ ] Error analysis: worst images, per-class metrics
- **Definition of done**: metrics targets reached (IoU 70-85%, Dice 0.75-0.90).

### Milestone 6 — Optimization (optional) + finalize (2-3 days)
- [ ] Optional: attention gates, residual connections, deep supervision, TTA
- [ ] Update `README.md` with setup/training/eval instructions
- [ ] Final run with fixed seed; commit reproducible config
- **Definition of done**: one command reproduces training + evaluation end-to-end.