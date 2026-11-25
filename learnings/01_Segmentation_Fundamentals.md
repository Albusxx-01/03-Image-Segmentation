# Segmentation Fundamentals

Understanding the different types of image segmentation and how they differ from classification and detection.

---

## Semantic Segmentation

Assigning a class label to **every pixel** in the image — pixel-wise classification.

- Output: Same spatial dimensions as input, with each pixel assigned a class
- All pixels of the same class are treated identically (no distinction between instances)

```
Input:  Image (H × W × 3)
Output: Mask  (H × W × num_classes)
```

**Papers:**
- [Long et al., 2015 — Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

**Official Documentation:**
- [PyTorch: Semantic Segmentation Models](https://pytorch.org/vision/stable/models.html#semantic-segmentation)

---

## Instance Segmentation

Distinguishing **individual object instances** within each class — each object gets its own mask.

- More complex than semantic segmentation
- Requires detecting objects AND segmenting each one
- Architecture: Mask R-CNN (detection + mask prediction)

**Papers:**
- [He et al., 2017 — Mask R-CNN](https://arxiv.org/abs/1703.06870)

---

## Pixel-Level vs Image-Level vs Box-Level Prediction

| Prediction Level | Output Density | Task |
|-----------------|---------------|------|
| **Image-level** | 1 label per image | Classification |
| **Box-level** | 1 box + label per object | Object Detection |
| **Pixel-level** | 1 label per pixel | Segmentation |

Segmentation has the densest output space — `H × W` predictions vs 1 or a handful.

**Official Documentation:**
- [CS231n: Segmentation (Stanford)](https://cs231n.github.io/understanding-cnn/)
