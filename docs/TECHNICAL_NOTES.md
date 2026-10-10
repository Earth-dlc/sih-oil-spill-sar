# Technical Notes — Architecture & Design Decisions

This document explains *why* specific technical choices were made, so judges and
future contributors can evaluate the reasoning rather than just the code.

---

## Why U-Net with ResNet34?

**U-Net** was chosen because:
1. It's the de facto standard for semantic segmentation with limited training data
2. The encoder-decoder architecture with skip connections preserves fine spatial
   detail — critical for delineating spill boundaries in SAR imagery
3. It's well-supported by `segmentation-models-pytorch`, which provides pretrained
   encoders out of the box

**ResNet34** (not ResNet50/101) was chosen because:
- Our GPU budget is an 8 GB laptop GPU (RTX 4060)
- ResNet34 fits comfortably in VRAM at batch size 8 with AMP enabled
- Deeper encoders showed diminishing returns on this dataset size (~1,100 images)
- ResNet34's ImageNet pretraining transfers surprisingly well to SAR imagery
  despite the domain gap (optical RGB → 2-channel SAR)

---

## Why Dice Loss Instead of Cross-Entropy?

Cross-Entropy Loss treats every pixel equally. In SAR oil spill imagery, the vast
majority of pixels are **sea surface** (class 0). A model that predicts "sea
everywhere" would achieve >90% pixel accuracy while detecting zero oil spills.

**Dice Loss** directly optimizes the overlap (intersection over union) between
predicted and ground-truth regions, which inherently handles class imbalance
without explicit class weighting. During development, switching from
`CrossEntropyLoss` to `DiceLoss(mode='multiclass')` was one of the most impactful
changes — training loss stopped getting stuck at 1.0.

---

## Why Per-Image Max-Scaling (Not Global Normalization)?

Standard deep learning practice normalizes images using dataset-wide mean and
standard deviation. We deliberately chose **per-image max-scaling** instead:

```python
img_max = image.max()
if img_max > 0:
    image = image / img_max
```

**Rationale**: SAR imagery has fundamentally different statistics from optical
photos. Radar backscatter values depend on:
- Incidence angle (varies across the swath)
- Sea state and wind speed at acquisition time
- Sensor calibration differences between orbits

Global normalization would crush the relative backscatter ratios within each scene
— exactly the signal that distinguishes oil (low backscatter) from sea (high
backscatter). Per-image scaling preserves these ratios.

**Known risk**: A single extremely bright pixel (e.g., a metallic ship) can
compress the dynamic range of the entire image. Future work could clip at the 98th
percentile before scaling.

---

## Why 70/15/15 Train/Val/Test Split?

- **70% train** provides sufficient data for a ResNet34-based U-Net (~21M parameters)
- **15% validation** is used for early stopping (best model checkpoint)
- **15% test** (~180 images) is held out completely and used only for final evaluation

The split is **seeded** (`set_seed(42)`) and saved to `split_manifest.json`,
ensuring exact reproducibility across runs.

---

## Why Automatic Mixed Precision (AMP)?

AMP uses float16 for forward/backward passes and float32 for parameter updates,
providing:
- ~40-50% memory reduction (allows batch size 8 on 8 GB VRAM)
- ~20-30% training speedup (Tensor Core acceleration on RTX GPUs)
- No measurable accuracy loss for this task

Implementation uses PyTorch 2.x API (`torch.amp.autocast`, `torch.amp.GradScaler`)
rather than the deprecated `torch.cuda.amp` module.

---

## Why 256×256 Input Resolution?

- The Krestenitis dataset images are not uniformly sized
- PyTorch's `DataLoader` requires all images in a batch to have identical dimensions
- 256×256 balances spatial detail against memory consumption
- Bilinear interpolation for images, nearest-neighbor for masks (to avoid creating
  fractional class labels)

**Trade-off**: Resizing may distort non-square images and lose fine boundary detail.
A sliding-window or padding approach would preserve resolution but adds complexity
beyond hackathon scope.

---

## Why Connected-Component Noise Filtering?

After argmax prediction, the raw segmentation mask often contains isolated small
clusters of "oil" pixels caused by SAR speckle noise. These are false positives
that would generate hundreds of tiny, meaningless GeoJSON polygons.

We apply OpenCV's `connectedComponentsWithStats` with a minimum area threshold
(50 pixels by default) to remove these clusters before converting to GeoJSON.
This is a simple but effective post-processing step that dramatically improves
output quality.

---

## Checkpoint Design

The training checkpoint saves not just model weights, but also:
- Optimizer state (Adam momentum terms)
- GradScaler state (AMP loss scaling factor)
- Current epoch number
- Best validation loss so far

This allows training to be **resumed from any interruption** without losing
progress or corrupting the best-model tracking logic.
