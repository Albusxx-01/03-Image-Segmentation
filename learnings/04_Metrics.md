# Metrics

Quantitative measures for evaluating segmentation quality at the pixel level.

---

## IoU (Jaccard Index)

The primary segmentation metric. Measures overlap between predicted and ground truth masks.

```
IoU = |A ∩ B| / |A ∪ B|
```

| IoU Value | Interpretation |
|-----------|---------------|
| 1.0 | Perfect overlap |
| 0.7 – 0.85 | Good segmentation (target range) |
| < 0.5 | Poor |

- **mIoU**: Mean IoU averaged across all classes
- Most widely used metric in segmentation benchmarks

**Official Documentation:**
- [Torchvision: MeanIoU](https://pytorch.org/vision/stable/generated/torchvision.metrics.functional.mean_iou.html)
- [Scikit-learn: Jaccard Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)

**Papers:**
- [Pascal VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)

---

## Dice Coefficient (F1 Score for Segmentation)

Harmonic mean of precision and recall at the pixel level:

```
Dice = 2·|A ∩ B| / (|A| + |B|)
```

| Dice Value | Interpretation |
|-----------|---------------|
| 1.0 | Perfect overlap |
| 0.75 – 0.90 | Target range |
| < 0.5 | Poor |

- Closely related to IoU: `Dice = 2·IoU / (1 + IoU)`
- Preferred in medical imaging due to its sensitivity to overlap quality

**Official Documentation:**
- [TensorFlow: Dice Metric](https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/DiceCoefficient)

---

## Pixel Accuracy

Fraction of correctly classified pixels across the entire image.

```
Pixel Accuracy = Correct Pixels / Total Pixels
```

- **Can be misleading**: A model predicting all-background achieves 95% accuracy if only 5% is foreground
- Useful only when classes are roughly balanced
- Prefer IoU/Dice for imbalanced datasets

**Official Documentation:**
- [Scikit-learn: Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
