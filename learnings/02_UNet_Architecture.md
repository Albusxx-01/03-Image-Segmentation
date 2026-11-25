# U-Net Architecture

The encoder-decoder architecture with skip connections designed for biomedical image segmentation.

---

## Encoder (Contracting Path)

Progressively captures context through repeated blocks of convolution + ReLU followed by MaxPooling 2×2.

```
Input (572×572) → Conv(64) → Conv(64) → MaxPool
                 → Conv(128) → Conv(128) → MaxPool
                 → Conv(256) → Conv(256) → MaxPool
                 → Conv(512) → Conv(512) → MaxPool
                 → Conv(1024) [Bottleneck]
```

- Each level: 2× Conv(3×3) + ReLU → MaxPool(2×2)
- Feature channels double at each level (64 → 128 → 256 → 512 → 1024)
- Spatial resolution halves at each level

---

## Bottleneck

The deepest, most abstract feature representation. Maximum context, minimum spatial detail.

- Highest number of feature channels (typically 1024)
- Captures the most semantic information

---

## Decoder (Expanding Path)

Progressively recovers spatial detail through upsampling and concatenation with encoder features.

```
Bottleneck → UpConv(512) + Skip(C5) → Conv(512) → Conv(512)
           → UpConv(256) + Skip(C4) → Conv(256) → Conv(256)
           → UpConv(128) + Skip(C3) → Conv(128) → Conv(128)
           → UpConv(64)  + Skip(C2) → Conv(64)  → Conv(64)
           → 1×1 Conv → Output Mask
```

- Each level: UpConv(2×2) → Concatenate with encoder feature map → 2× Conv(3×3) + ReLU

---

## Skip Connections

Concatenating encoder feature maps to the decoder at the same resolution level.

- **Purpose**: Preserve fine spatial detail lost during pooling
- **Mechanism**: Channel-wise concatenation (doubles channels at that level)
- **Result**: Decoder has both high-level semantics (from bottleneck) AND low-level spatial detail (from skip connections)

**Papers:**
- [Ronneberger et al., 2015 — U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## Output Layer

- **Binary segmentation**: 1×1 Conv → Sigmoid (1 channel, threshold at 0.5)
- **Multi-class segmentation**: 1×1 Conv → Softmax (C channels, argmax per pixel)

```
Last Conv(64) → 1×1 Conv(num_classes) → Softmax → Predicted Mask (H × W × C)
```

---

## Feature Maps

Visualization of learned activations at each encoder/decoder level reveals what the network has learned:

- **Early encoder layers**: Edges, textures
- **Deep encoder layers**: Object parts, shapes
- **Decoder layers**: Refined spatial predictions

**Official Documentation:**
- [PyTorch: U-Net Implementation Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
