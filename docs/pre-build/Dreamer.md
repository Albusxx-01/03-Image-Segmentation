# Image Segmentation (U-Net) — Dreamer

## Problem Statement
Many applications (medical imaging, self-driving cars, agriculture) need a class label for *every pixel* of an image, not just a class per image or per box. Weak boundaries, small structures, and imbalanced class sizes make this hard. This project implements the U-Net architecture for semantic segmentation, producing dense pixel-wise masks from images such as medical scans or satellite/car imagery.

## User Stories
- As a medical researcher, I want pixel-accurate masks for structures like tumours or organs so that I can quantify them automatically.
- As a developer, I want the U-Net encoder-decoder architecture with skip connections so that spatial detail is preserved.
- As a data scientist, I want a mask-compatible pipeline (image ↔ mask pairs) so that I can train on datasets like Carvana or medical scans.
- As an evaluator, I want IoU and Dice coefficient per class so that I can measure segmentation quality robustly.
- As a user, I want side-by-side visualizations of input, ground-truth mask, and predicted mask so that I can sanity-check the model.
- As a doctor, I want segmentations that handle imbalanced classes via Dice/BCE loss so that small structures are not ignored.

## Features
- Dataset of image + pixel-mask pairs (Carvana, medical, or satellite imagery).
- U-Net architecture: 4-level encoder, bottleneck, 4-level decoder with concatenative skip connections.
- Loss: binary cross-entropy, Dice loss, or combined BCE + Dice.
- Metrics: IoU, Dice coefficient, pixel accuracy.
- Evaluation: side-by-side input vs ground truth vs prediction visualization.
- Data augmentation: rotation, flips, elastic deformation.
- Optional optimization: attention gates, residual connections, deep supervision, test-time augmentation.

## Success Metrics
- IoU score: 70-85%
- Dice coefficient: 0.75-0.90
- Visual masks closely matching ground truth on held-out test images.
- Stable training without class imbalance dominating the loss.
- Reproducible training pipeline with a fixed seed and saved checkpoints.