# Image Segmentation (U-Net) — Planner

## Technical Architecture
```
[Input Image] ──► Encoder                        Decoder ──► [1x1 Conv + Sigmoid]
                   │ Conv Block ×4 (2×ConvReLU)      ▲                 └► mask
                   │ MaxPool (2×2) downsample        │
                   ▼                                 │
                 Bottleneck (deepest features) ──────┘
                 (Conv Block, 2×ConvReLU)       upsampling UpConv(2×2)
                                               + skip concat + Conv Block
```

Pipeline stages:
1. Data + masks: create/obtain pixel-level masks per image.
2. Preprocessing: resize image+mask pairs, augment (rotation, flip, elastic deformation).
3. Model building: U-Net with concatenative skip connections.
4. Training: BCE, Dice, or BCE+Dice loss; Adam (lr=0.001); LR scheduling.
5. Evaluation: IoU, Dice coefficient, pixel accuracy.
6. Visualization: side-by-side input / ground truth / prediction.

## Tech Stack
- **Language**: Python 3.9+
- **Framework**: PyTorch or TensorFlow/Keras
- **Data handling**: NumPy, OpenCV / PIL
- **Augmentation**: Albumentations, imgaug (rotation, flip, elastic)
- **Visualization**: Matplotlib
- **Metrics**: custom IoU/Dice implementations
- **Version control**: Git

## File and Folder Design
```
03-Image-Segmentation/
├── pre-build docs/
├── data/
│   ├── images/train,val,test/
│   └── masks/train,val,test/     # same filenames as images
├── config/
│   └── unet.yaml                 # paths, input size, n_classes, hyperparams
├── src/
│   ├── dataset.py                # image+mask loader, transforms
│   ├── model.py                  # UNet builder (encoder/bottleneck/decoder)
│   ├── loss.py                   # DiceLoss, BCE, CombinedLoss
│   ├── metrics.py                # IoU, Dice, pixel accuracy
│   ├── train.py                  # training loop, callbacks
│   ├── evaluate.py               # metrics + side-by-side plots
│   └── predict.py                # single-image inference + overlay
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   └── 02-error-analysis.ipynb
├── models/                       # saved checkpoints
├── results/                      # metric tables, overlay visualizations
└── requirements.txt
```

## Data Model / Schemas
**Image**
- Tensor: `(batch, H, W, channels)` uint8 → normalized 0.0-1.0 (RGB) or 0.0-1.0 (grayscale).
- Typical input sizes: 256x256 or 512x512 (resized consistently).

**Mask (ground truth)**
- Binary segmentation: `(batch, H, W, 1)` values {0, 1}, float.
- Multi-class: `(batch, H, W, n_classes)` one-hot encoded.
- Masks and images MUST share filenames (e.g., `img_001.png` ↔ `mask_001.png`).

**Class map**
- `classes.json`: `{0: "background", 1: "object"}`
- For multi-class: index = mask pixel value.

**Loss**
- Dice loss: `1 - (2·|A∩B| + smooth) / (|A| + |B| + smooth)`
- Combined loss: `BCE + λ · DiceLoss` (λ configurable).

**Metrics**
- IoU per class: `|A∩B| / |A∪B|` averaged over classes.
- Dice coefficient: `2·|A∩B| / (|A| + |B|)`.
- Pixel accuracy: fraction of correctly predicted pixels.

**Checkpoint**
- `.pt` (PyTorch state_dict) or `.h5` (Keras); include `unet.yaml` for exact config.