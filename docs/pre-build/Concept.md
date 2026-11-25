# Image Segmentation (U-Net) — Technical Concepts

## Segmentation Fundamentals
- **Semantic segmentation**: Assigning a class label to every pixel (pixel-wise classification).
- **Instance segmentation**: Distinguishing individual object instances (beyond scope; uses Mask R-CNN).
- **Pixel-level vs image-level/box-level prediction**: Density of the output space.

## U-Net Architecture
- **Encoder (contracting path)**: Repeated Conv Blocks (2× Conv + ReLU) + MaxPool 2×2; captures context at multiple scales.
- **Bottleneck**: Deepest, most abstract feature representation.
- **Decoder (expanding path)**: UpConv 2×2 upsampling + concatenation of encoder features via skip connections + Conv Blocks for refinement.
- **Skip connections**: Concatenating encoder features to decoder at same resolution to preserve fine spatial detail lost in pooling.
- **Output layer**: 1×1 Conv + Sigmoid (binary) or Softmax (multi-class).
- **Feature maps**: Visualization of learned activations for interpretability.

## Loss Functions
- **Binary cross-entropy (BCE)**: Pixel-wise classification loss (log loss) for binary masks.
- **Dice loss**: `1 - Dice`; robust to class imbalance, focus on overlap.
- **Combined BCE + Dice loss**: Hybrid for stability (BCE) + overlap quality (Dice).
- **Class imbalance**: Background dominates foreground in most masks; Dice/sampling counter this.

## Metrics
- **IoU (Jaccard)**: `|A∩B| / |A∪B|` per class, averaged; target 70-85%.
- **Dice coefficient**: `2·|A∩B| / (|A| + |B|)`; target 0.75-0.90.
- **Pixel accuracy**: Fraction of correctly classified pixels (can be misleading with imbalance).

## Training & Preprocessing
- **Resize/alignment**: Images and masks must share identical spatial transforms.
- **Augmentation**: Rotation, flips, shifts, elastic deformation — applied identically to image+mask to keep them aligned.
- **Learning rate scheduling**: Reducing LR to refine late-training convergence.
- **Early stopping**: Halt when validation loss/metric plateaus.

## Advanced Options
- **Attention gates**: Suppress irrelevant regions in skip connections.
- **Residual connections**: Stabilize deep encoder/decoder blocks.
- **Deep supervision**: Auxiliary losses at multiple decoder levels for faster convergence.
- **Test-time augmentation (TTA)**: Averaging predictions across augmented versions of the input.
- **Transposed convolution / upsampling**: Methods for increasing feature map resolution.

## Applications
- Medical imaging: tumor/+organ segmentation, radiology analysis.
- Autonomous driving: road, lane, obstacle/segment parsing.
- Satellite imagery: land cover classification.
- Agriculture: crop disease region detection.