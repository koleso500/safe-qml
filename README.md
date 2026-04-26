# SAFE Quantum Machine Learning with Variational Quantum Classifiers

Code for the paper **"SAFE Quantum Machine Learning with Variational Quantum Classifiers"** (under review at ICML 2026).

---

## Overview

This repository implements a hybrid classical–quantum framework for brain tumor MRI classification. The pipeline consists of:

- Feature extraction using ResNet-18 pretrained on ImageNet
- Classification with Variational Quantum Circuit (VQC) and classical baselines
- Evaluation using SAFE-AI metrics based on Cramér–von Mises divergence

Models are evaluated across three key dimensions:
- Predictive accuracy
- Robustness to noise perturbations
- Explainability robustness via Grad-CAM occlusions

---

## Installation

### Option 1: Standard Installation (CPU or Auto-detect)
```bash
pip install -r requirements.txt
```

### Option 2: CUDA 11.8 Installation (GPU)
```bash
pip install -r requirements-cuda.txt
```

### Option 3: Custom PyTorch Installation
1. First, install PyTorch with your preferred configuration from [pytorch.org](https://pytorch.org)
2. Then install remaining dependencies:
```bash
pip install -r requirements.txt --no-deps torch torchaudio torchvision
```

---

## Dataset

Download the publicly available **Brain Cancer MRI Dataset** (Rahman, 2024):

- **DOI:** https://doi.org/10.17632/mk56jw9rns.1
- **Size:** 6,056 MRI images
- **Classes:** 3 (glioma, meningioma, tumor)

**Expected directory structure:**
```
data/
└── Brain_Cancer/
    ├── glioma/
    ├── meningioma/
    └── tumor/
```

---

## Usage

Run the complete experimental pipeline:

```bash
python main.py
```

**The script executes the following steps:**
1. Preprocesses MRI images and normalizes inputs
2. Extracts ResNet-18 features 
3. Trains five classifiers with 5-fold cross-validation
4. Computes SAFE metrics
5. Saves results to `figures/` and `tables/`

---

## Runtime

**GPU (NVIDIA GeForce RTX 3060 Laptop):** ~50-60 minutes  
**CPU:** significantly slower, not extensively tested

**Note:** While the code is fully compatible with CPU-only environments, consider reducing experimental complexity
(e.g., cross-validation splits or training epochs) when running without GPU acceleration.

---

## Configuration

Core parameters are configured in `config.yaml`. Some key defaults:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_SPLITS` | 5 | K-Fold Cross-Validation splits |
| `N_QUBITS` | 9 | Number of Qubits |
| `N_LAYERS` | 1 | Strongly Entangling Layers in VQC |

---

## Repository Structure

```
.
├── main.py                 # Main experimental pipeline
├── config.yaml             # Configuration parameters
├── requirements.txt        # Python dependencies (CPU/auto)
├── requirements-cuda.txt   # Python dependencies (CUDA 11.8)
│
├── safe/                   # SAFE-AI metrics implementation
│   ├── cramer.py           # Cramér-von Mises divergence
│   ├── rga.py              # RGA (accuracy) metric
│   ├── rgr.py              # RGR (robustness) metric
│   ├── rge.py              # RGE (explainability) metric
│   └── utils.py            # Grad-CAM, preprocessing, helpers
│
├── data/                   # Dataset directory
│   └── Brain_Cancer/       # MRI images (download separately)
│       ├── glioma/
│       ├── meningioma/
│       └── tumor/
│
├── figures/                # Generated plots (created at runtime)
└── tables/                 # Generated CSV results (created at runtime)
```

---

## Reproducibility

All experiments use fixed random seed (`SEED = 42`) for NumPy, PyTorch and SAFE-AI perturbations.

---


**Status:** Preliminary research code under review at ICML 2026. Don't distribute