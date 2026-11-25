# Loss Functions

Loss functions designed for pixel-wise prediction tasks, addressing class imbalance and overlap quality.

---

## Binary Cross-Entropy (BCE)

Pixel-wise classification loss (log loss) for binary masks. Each pixel is independently classified.

```
L_BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

- Simple and stable, but doesn't directly optimize overlap quality
- Sensitive to class imbalance (background dominates foreground)

**Official Documentation:**
- [TensorFlow: BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)
- [PyTorch: BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

---

## Dice Loss

Based on the Dice coefficient — directly optimizes the overlap between predicted and ground truth masks.

```
L_Dice = 1 - (2·|ŷ ∩ y| + smooth) / (|ŷ| + |y| + smooth)
```

- **Robust to class imbalance**: Focuses on overlap ratio, not pixel counts
- **smooth** term (e.g., 1.0) prevents division by zero
- Target Dice coefficient: 0.75–0.90

**Papers:**
- [Milletari et al., 2016 — V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)

---

## Combined BCE + Dice Loss

Hybrid loss combining the stability of BCE with the overlap optimization of Dice.

```
L_total = α · L_BCE + (1 - α) · L_Dice
```

- **BCE**: Provides stable, per-pixel gradients
- **Dice**: Optimizes the actual metric of interest (overlap)
- Common choice: `α = 0.5` (equal weighting)

**Official Documentation:**
- [PyTorch: Combined Loss Examples](https://github.com/usuyama/pytorch-unet)

---

## Class Imbalance

Background pixels vastly outnumber foreground pixels in most segmentation masks (e.g., 95% background, 5% foreground).

| Strategy | Mechanism |
|----------|-----------|
| **Dice loss** | Inherently handles imbalance via ratio-based loss |
| **Class weighting** | Assign higher weight to minority class pixels |
| **Oversampling** | Crop patches where foreground is present |
| **Focal loss** | Down-weights easy (background) pixels |

**Papers:**
- [Lin et al., 2017 — Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
