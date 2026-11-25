# Training & Preprocessing

Data preparation and training strategies specific to segmentation tasks.

---

## Resize & Alignment

Images and masks must share **identical spatial transforms** to maintain pixel-level correspondence.

- Both image and mask resized to the same dimensions (e.g., 256×256, 512×512)
- Any spatial transform applied to the image must be identically applied to the mask
- Interpolation: Bilinear for images, Nearest Neighbor for masks (preserves label values)

**Official Documentation:**
- [Albumentations: Spatial Transforms](https://albumentations.ai/docs/api_reference/spatial_transforms/)
- [PyTorch: Transforms](https://pytorch.org/vision/stable/transforms.html)

---

## Augmentation

Random transformations applied identically to image + mask to expand the training dataset.

| Transform | Effect on Segmentation |
|-----------|----------------------|
| Random horizontal flip | Mirror invariance |
| Random rotation (±45°) | Rotation invariance |
| Random shift/scale | Scale & position invariance |
| Elastic deformation | Simulates tissue deformation (medical) |
| Random brightness/contrast | Lighting invariance |

- **Critical**: Apply the same random seed to both image and mask
- Albumentations library handles image+mask synchronization automatically

**Official Documentation:**
- [Albumentations: Segmentation Examples](https://albumentations.ai/docs/examples/example-targets/)
- [TensorFlow: Data Augmentation for Segmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

---

## Learning Rate Scheduling

Reducing the learning rate during training to refine convergence in later epochs.

| Strategy | Description |
|----------|-------------|
| **Step decay** | Reduce LR by factor every N epochs |
| **ReduceOnPlateau** | Reduce LR when validation metric plateaus |
| **Cosine annealing** | Smoothly decrease LR following cosine curve |
| **Warmup + decay** | Start low, increase, then decay |

**Official Documentation:**
- [PyTorch: LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [TensorFlow: LearningRateSchedule](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules)

---

## Early Stopping

Halt training when the validation loss or metric (e.g., IoU, Dice) plateaus.

- **Patience**: Number of epochs to wait for improvement (typically 10–20)
- Restore the best model weights from the epoch with best validation metric
- Prevents overfitting and saves compute

**Official Documentation:**
- [TensorFlow: EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
