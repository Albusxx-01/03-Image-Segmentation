# Advanced Options

Techniques to improve U-Net performance beyond the basic architecture.

---

## Attention Gates

Mechanisms that suppress irrelevant regions and focus on salient structures in skip connections.

- Learn to weight skip connection features based on the decoder's context
- Reduce false positives from irrelevant background regions
- Lightweight addition with minimal computational overhead

```
Skip features × Attention coefficients = Focused features
```

**Papers:**
- [Oktay et al., 2018 — Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)

---

## Residual Connections

Skip connections within encoder/decoder blocks (not to be confused with U-Net's cross-level skip connections).

- Stabilize deep networks by allowing gradients to flow through identity shortcuts
- Enable training of deeper U-Net variants
- Used in ResUNet and Attention-ResUNet

**Papers:**
- [He et al., 2016 — Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)

---

## Deep Supervision

Adding auxiliary loss functions at multiple decoder levels to provide gradient signals earlier in training.

```
L_total = L_output + α₁·L_level3 + α₂·L_level2 + α₃·L_level1
```

- Faster convergence by supervising intermediate predictions
- Each auxiliary output is upsampled to match the final output resolution
- Auxiliary losses typically weighted with decreasing α

**Papers:**
- [Dou et al., 2017 — Cascaded Deep Video Deblurring Using Temporal Sharpness Prior](https://arxiv.org/abs/1707.09230)

---

## Test-Time Augmentation (TTA)

Averaging predictions across multiple augmented versions of the input at inference time.

| Augmentation | Purpose |
|-------------|---------|
| Original | Baseline prediction |
| Horizontal flip | Capture mirrored features |
| Multiple scales (0.8×, 1.0×, 1.2×) | Scale robustness |
| Rotations (0°, 90°, 180°, 270°) | Rotation invariance |

- Typically improves IoU by 1–4% at the cost of N× inference time

---

## Transposed Convolution / Upsampling

Methods for increasing feature map resolution in the decoder.

| Method | Description | Pros/Cons |
|--------|-------------|-----------|
| **Transposed Conv** | Learnable upsampling | Can create checkerboard artifacts |
| **Bilinear Upsampling** | Fixed interpolation + Conv | Smoother, no artifacts |
| **PixelShuffle** | Sub-pixel convolution | Efficient, learned upsampling |

**Official Documentation:**
- [PyTorch: ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
- [PyTorch: Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
