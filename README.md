# Defect Detection in Textiles — Assignment 3

## Computer Vision · Technische Hochschule Deggendorf · Summer 2026

**Student:** Mohamed Eldemellawy

**Supervisor:** Prof. Marcus Barkowsky

**Programme:** Master's — Automotive Software Engineering

---

## Paper Reimplemented

> H. Alimohamadi, A. Ahmadyfard, E. Shojaee,
> "Defect Detection in Textiles Using Morphological Analysis of Optimal Gabor Wavelet Filter Response,"
> *2009 International Conference on Computer and Automation Engineering (ICCAE)*,
> Bangkok, Thailand, 2009, pp. 26–30.
> DOI: [10.1109/ICCAE.2009.43](https://doi.org/10.1109/ICCAE.2009.43)

---

## Project Structure

```
DataSet/
├── Main.py                   # Full pipeline — run this
├── README.md                 # This file
├── defective/                # Defective images used for testing
├── defect_free/              # Defect-free images used for testing
├── output/                   # Output visualisations (generated at runtime)
├── results_hole.txt          # Saved terminal output for the hole-defect run
├── results_stain.txt         # Saved terminal output for the stain-defect run
└── venv/                     # Python virtual environment
```

### What the result files contain

- `results_hole.txt` stores the console output from running the detector on the hole-defect subset.
- `results_stain.txt` stores the console output from running the detector on the stain-defect subset.
- Each file includes the per-image detection summary, the selected Gabor filter, the Fisher score, the adaptive $h$ value, and whether each image was detected or missed.

---

## Algorithm Overview

The pipeline follows the paper exactly in three stages:

**Stage 1 — Gabor Wavelet Filter Bank (Eq. 1–7)**

A bank of M×N complex Gabor kernels at M=4 scales and N=6 orientations (24 filters total).
Parameters: Ul=0.01, Uh=0.4, mask size 33×33.

**Stage 2 — Optimal Filter Selection via Fisher Criterion (Eq. 9–12)**

For each filter, compute mean μ and mean absolute deviation σ of the filter response.
Select the filter with maximum F = μ/σ (Fisher criterion).

**Stage 3 — Morphological Analysis for Adaptive Detection (Eq. 13–16)**

Normalise the input image: N(x,y) = |M − T(x,y)|.
Apply grayscale morphological reconstruction (Vincent 1993) to extract h-domes.
Adaptive h = MAD of the feature image — no manual threshold required.

---

## How to Run

### Setup

```
cd /path/to/DataSet
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python matplotlib scikit-image
```

### Run on your dataset

```
python3 Main.py \
  "/path/to/defective" \
  "/path/to/defect_free" \
  "/path/to/output_results"
```

Optional: add a ground-truth mask directory as a 4th argument for pixel-level evaluation:

```
python3 Main.py \
  "/path/to/defective" \
  "/path/to/defect_free" \
  "/path/to/gt_masks" \
  "/path/to/output_results"
```

### Run smoke test (no dataset needed)

```
python3 Main.py
```

Generates a synthetic textile image with a dark defect and saves the result to `smoke_test_output/`.

---

## Dependencies

Library
Version
Purpose

Python
≥ 3.10
Language

NumPy
≥ 1.24
Array operations, filter math

OpenCV
≥ 4.8
Convolution, morphology, I/O

Matplotlib
≥ 3.7
Result visualisation

> **Note:** Morphological grayscale reconstruction is hand-coded (iterative geodesic dilation).
> `skimage.morphology.reconstruction` is deliberately not used, in accordance with the
> assignment requirement for a hand-coded solution.

---

## Datasets Used

The original paper evaluated on a **private, unpublished 71-image dataset** that is no longer
available.

I used the dataset from:

https://data.mendeley.com/datasets/663j22s43c/3

