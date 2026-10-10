# Data Preparation Guide

This guide explains how to download, organize, and verify the Krestenitis SAR
oil spill dataset for use with this project.

---

## 1. Download the Dataset

The dataset is hosted on Zenodo by Krestenitis et al.:

**📥 Download Link**: [https://zenodo.org/records/8346860](https://zenodo.org/records/8346860)

Download the full archive (~4.5 GB). It contains Sentinel-1 GRD (Ground Range
Detected) SAR imagery with pixel-level segmentation masks for 5 classes:

| Class | Index | Description |
|---|---|---|
| Sea Surface | 0 | Open water (majority class) |
| Oil Spill | 1 | Confirmed oil contamination |
| Look-Alike | 2 | Biogenic films, calm water, algal blooms |
| Ship | 3 | Vessel signatures |
| Land | 4 | Coastal land areas |

---

## 2. Organize the Files

After downloading, place the data in the following structure relative to the
project root:

```
dataset/
├── images/           ← 2-band SAR GeoTIFFs (VV + VH polarization)
│   ├── 00001.tif
│   ├── 00002.tif
│   └── ...          (~1,112 files, each ~4 MB)
│
└── mask oil/         ← Corresponding segmentation masks
    ├── 00001.tif
    └── ...          (same filenames as images/)
```

> **Important**: The mask folder is named `mask oil` (with a space). Do not rename
> it — the data loading scripts expect this exact name.

---

## 3. Verify the Dataset

Run a quick sanity check:

```bash
# Count images and masks (should be equal, ~1,112 each)
python -c "import glob; print('Images:', len(glob.glob('dataset/images/*.tif'))); print('Masks:', len(glob.glob('dataset/mask oil/*.tif')))"
```

Expected output:
```
Images: 1112
Masks: 1112
```

Each image file should be:
- **Format**: GeoTIFF
- **Bands**: 2 (VV and VH polarization)
- **Size**: ~4 MB each
- **Pixel type**: Float32

Each mask file should be:
- **Format**: TIFF (single-band)
- **Values**: Integer class labels (0–4)

---

## 4. Generate the Split Manifest

Once the dataset is in place, generate the deterministic train/val/test split:

```bash
python main.py prepare
```

This creates `split_manifest.json` with a seeded 70/15/15 split:
- **~778 training** images
- **~167 validation** images
- **~180 test** images (held-out, never used during training)

The random seed is fixed at 42, so the split is identical across runs.

---

## 5. SAR Image Details

### What are VV and VH?

Sentinel-1 transmits and receives radar pulses in two polarization modes:

- **VV** (vertical transmit, vertical receive): Stronger response from rough sea
  surfaces. Oil spills dampen this signal.
- **VH** (vertical transmit, horizontal receive): Sensitive to volume scattering.
  Useful for discriminating oil from look-alikes.

Using both bands together (2-channel input) provides the model with complementary
information about surface roughness and composition.

### Why GeoTIFFs?

GeoTIFF files embed geographic metadata (coordinate reference system, affine
transform) that allows the model's pixel-level predictions to be converted into
real-world latitude/longitude coordinates. This is critical for the downstream
vessel attribution workflow.

---

## Notes

- The dataset is provided under the terms specified on the Zenodo record page
- Some preprocessed versions of the dataset may have stripped CRS metadata;
  use the original files from Zenodo for georeferenced inference
- Total storage requirement: ~5 GB for the dataset + ~400 MB for model weights
