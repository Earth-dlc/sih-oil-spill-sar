# 🛢️ SAR Oil Spill Detection & Vessel Attribution

**Smart India Hackathon 2026 — Problem Statement #26143**

> A deep learning pipeline that detects oil spills in Sentinel-1 SAR (Synthetic Aperture Radar) satellite imagery and outputs geospatial polygons for downstream vessel attribution. Trained on the Krestenitis dataset (1,112 labeled SAR scenes), the model achieves **83.63% Mean IoU** and **80.42% Oil Spill Dice** on a 180-image held-out test set, enabling rapid triage of satellite imagery into actionable investigation leads.

---

## 🧭 The Problem

When an oil spill is spotted at sea by satellite, the critical question is: **which vessel caused it?**

Current approaches rely on manual visual inspection of SAR imagery by trained analysts — slow, subjective, and impossible to scale across the thousands of Sentinel-1 scenes captured daily. This project automates the first step of that workflow: **detecting and localizing oil spills in SAR imagery**, outputting precise geographic coordinates (GeoJSON polygons) that can be fed into drift-back models (e.g., OpenDrift) and cross-referenced against vessel tracking data (AIS) to identify responsible ships.

### Why SAR?

Synthetic Aperture Radar works through clouds and at night, making it the only reliable way to monitor oceans 24/7. Oil spills appear as dark patches in SAR imagery because oil dampens the radar backscatter from ocean waves. However, **many natural phenomena look similar** (algal blooms, low-wind zones, upwelling) — which is exactly why we need machine learning to distinguish real spills from look-alikes.

---

## 🏗️ Architecture

| Component | Specification |
|---|---|
| **Model** | U-Net (fully convolutional encoder-decoder) |
| **Encoder** | ResNet34, pretrained on ImageNet |
| **Input** | 2-channel SAR (VV + VH dual-polarization) |
| **Output** | 5-class segmentation (sea, oil spill, look-alike, ship, land) |
| **Resolution** | 256 × 256 pixels |
| **Loss** | Dice Loss (multiclass) — chosen for minority-class handling |
| **Optimizer** | Adam (lr=1e-4) |
| **Training** | 20 epochs, batch size 8, AMP (Automatic Mixed Precision) |
| **Post-processing** | Connected-component noise removal → GeoJSON polygon extraction |

---

## 📊 Results

Evaluated on **180 held-out test images** never seen during training:

| Metric | Value | Meaning |
|---|---|---|
| **Mean Image-wise IoU** | 83.63% | Average overlap across all 5 classes per image |
| **Oil Spill Dice** | 80.42% | Harmonic mean of precision/recall for oil class |
| **Global Micro IoU (Oil)** | 65.71% | Cumulative oil-pixel overlap across all test images |
| **Average Dice Loss** | 0.0392 | Training loss metric (lower = better) |

For a detailed breakdown including per-metric explanations, comparison to published benchmarks, and known issues, see [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md).

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10+ (tested on 3.12)
- **GPU**: NVIDIA GPU with ≥6 GB VRAM (tested on RTX 4060, 8 GB)
- **CUDA**: 12.1 (matching PyTorch 2.5.1+cu121)
- **OS**: Windows 10/11 or Linux (macOS untested)

> **CPU-only fallback**: The code will detect and fall back to CPU automatically. Training will be significantly slower (~10x), but inference is feasible.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-team>/sih-oil-spill-sar.git
cd sih-oil-spill-sar

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Download the Krestenitis oil spill dataset from [Zenodo](https://zenodo.org/records/8346860) and place it in the `dataset/` folder:

```
dataset/
├── images/       ← SAR GeoTIFF files (2-band: VV + VH)
└── mask oil/     ← Corresponding pixel-level mask TIFFs
```

See [DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for detailed download and verification instructions.

### Run the Pipeline

```bash
# Step 1: Generate train/val/test split (70/15/15, seeded)
python main.py prepare

# Step 2: (Optional) Smoke-test the model architecture
python main.py smoke-test

# Step 3: Train the model (~4 hours on RTX 4060)
python main.py train

# Step 4: Evaluate on held-out test set
python main.py evaluate

# Step 5: Run inference on test images → GeoJSON output
python main.py predict
```

---

## 📁 Repository Structure

```
├── README.md                   ← You are here
├── MODEL_PERFORMANCE.md        ← Detailed evaluation report with metrics
├── requirements.txt            ← Pinned Python dependencies
├── config.yaml                 ← Centralized hyperparameters and paths
├── LICENSE                     ← MIT License
├── .gitignore
│
├── main.py                     ← Main CLI entry point (run train, evaluate, predict)
├── src/
│   ├── config.py               ← Configuration loader
│   ├── dataset.py              ← Data loading and augmentation
│   ├── model.py                ← U-Net initialization
│   └── utils.py                ← Random seeding and logging
│
├── scripts/
│   ├── prepare_data.py         ← Generate train/val/test split manifest
│   ├── train.py                ← Training loop with AMP + checkpointing
│   ├── evaluate.py             ← Full holdout test evaluation (IoU, Dice)
│   ├── predict.py              ← Batch inference → noise filter → GeoJSON
│   └── smoke_test.py           ← Verify model forward pass
│
├── docs/
│   ├── TECHNICAL_NOTES.md      ← Architecture decisions explained
│   ├── DATA_PREPARATION.md     ← Dataset download and setup guide
│   └── INFERENCE_GUIDE.md      ← Using predict.py on new SAR data
│
├── dataset/                    ← Krestenitis SAR images + masks (not tracked)
├── reports/                    ← Generated GeoJSON oil-spill polygons
└── split_manifest.json         ← Deterministic data split (generated)
```

---

## 🖥️ Hardware Requirements

### Training

| Resource | Minimum | Tested On |
|---|---|---|
| GPU VRAM | 6 GB | 8 GB (RTX 4060 Laptop) |
| System RAM | 16 GB | 16 GB |
| Storage | ~20 GB | (dataset + checkpoints) |
| Time | ~4 hours | 20 epochs on RTX 4060 |

### Inference

- **GPU**: Optional (CPU fallback adds ~5s/image)
- **RAM**: < 2 GB
- **Time**: < 1 second/image (GPU) or ~5 seconds (CPU)

---

## ⚠️ Known Limitations

1. **All-positive test set**: Every image in the Krestenitis dataset contains oil. False positive rate on clean ocean scenes is **unknown**.
2. **Geographic bias**: Dataset covers Mediterranean, North Sea, and Atlantic. Performance on tropical waters (Gulf of Mexico, Southeast Asia) is untested.
3. **Single-timepoint detection**: The model sees one SAR acquisition. Real spills drift over hours; temporal tracking requires multi-temporal data.
4. **256×256 resize**: All images are resized to 256×256, which may distort aspect ratios and lose fine-grained boundary detail.
5. **No uncertainty quantification**: Outputs are argmax predictions, not calibrated probabilities.

---

## 🔮 Future Work

- [ ] Evaluate false positive rate on clean ocean scenes
- [ ] Test geographic generalization beyond Mediterranean/Atlantic
- [ ] Integrate temporal multi-frame spill tracking
- [ ] Uncertainty quantification (Bayesian or ensemble methods)
- [ ] Lightweight encoder (MobileNet) for edge deployment
- [ ] End-to-end pipeline: SAR detection → OpenDrift backward drift → AIS vessel matching

---

## 📚 Citations & Acknowledgments

**Dataset**:
Krestenitis, M., et al. (2020). *Oil-spill detection in SAR imagery using convolutional neural networks.* Remote Sensing, 12(11), 1762.
Zenodo: [https://zenodo.org/records/8346860](https://zenodo.org/records/8346860)

**Key Libraries**:
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) — U-Net implementation
- [PyTorch](https://pytorch.org) — Deep learning framework
- [Rasterio](https://rasterio.readthedocs.io) — Geospatial raster I/O
- [OpenCV](https://opencv.org) — Connected-component noise filtering

**Hackathon**:
Smart India Hackathon 2026, Problem Statement #26143:
*"Tracing Oil Spills Back to Responsible Vessels via Satellite Imagery and Maritime Data"*

---

## 👥 Team

| Name | Role | Institution |
|---|---|---|
| Pruthviraj Abhijeet Pethe | Lead Developer | Sinhgad Institute of Technology |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
