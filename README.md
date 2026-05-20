# NeuroFuzzy DR 

> An end-to-end deep learning pipeline for **Diabetic Retinopathy grading** combining EfficientNetB3 feature extraction with pure-NumPy Mamdani fuzzy logic, ordinal regression, and QWK threshold optimization.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-Academic-purple)](LICENSE)

---

##  Overview

NeuroFuzzy DR is a clinical-grade diabetic retinopathy screening system trained on the **APTOS 2019** fundus image dataset. It classifies retinal images into 5 DR severity grades using a neuro-fuzzy pipeline that combines deep learning with interpretable fuzzy logic post-processing.

| Metric | Value |
|--------|-------|
| Best QWK (with TTA) | **0.892** |
| Best QWK (no TTA) | 0.874 |
| Validation Accuracy | 83.4% |
| Best Validation Loss | 0.241 |
| Training Epochs | 34 |
| Input Resolution | 300 × 300 px |

---

##  DR Grades

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No signs of diabetic retinopathy |
| 1 | Mild DR | Microaneurysms only |
| 2 | Moderate DR | Haemorrhages, exudates, cotton-wool spots |
| 3 | Severe DR | Extensive haemorrhages, venous beading |
| 4 | Proliferative DR | Neovascularisation, vitreous haemorrhage |

---

##  Pipeline Architecture

```
Fundus Image
     │
     ▼
CLAHE Preprocessing (LAB colorspace, 8×8 tile)
     │
     ▼
EfficientNetB3 Backbone (ImageNet pretrained)
     │
     ▼
Ordinal Regression Head (Cumulative-link, 4 outputs)
     │
     ▼
QWK Threshold Optimization
     │
     ▼
┌────────────────────────────────────┐
│         Fuzzy Logic Layer          │
│  Module 1: Orientation Correction  │
│  Module 2: Grade Adjustment        │
└────────────────────────────────────┘
     │
     ▼
Final DR Grade (0–4)
```

---

##  Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/janawalid/NeuroFuzzyDR.git
cd NeuroFuzzyDR

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Then open your browser at `http://localhost:5000`

> **Note:** The trained model (`model/efficientnetb3_dr_savedmodel/`) is not included in this repository due to file size. Contact the authors to obtain the model weights.

---

##  Project Structure

```
NeuroFuzzyDR/
│
├── app.py                          # Flask backend
├── requirements.txt
│
├── model/
│   ├── efficientnetb3_dr_savedmodel/   # Trained TF SavedModel (not in repo)
│   ├── thresholds.npy                  # QWK-optimized thresholds
│   ├── fuzzy_adj.py                    # Adjustment fuzzy module
│   └── fuzzy_orient.py                 # Orientation fuzzy module
│
├── static/
│   ├── script.js                   # Frontend logic + Plotly charts
│   ├── style.css                   # Styles
│   └── favicon.svg                 # Site favicon
│
└── templates/
    └── index.html                  # Main UI
```

---

##  Fuzzy Logic Modules

### Module 1 — Orientation Correction
Detects whether a fundus image is flipped/inverted based on optic disc position and dark notch ratio. Uses Mamdani fuzzy inference with centroid defuzzification.

**Inputs:** Optic disc x-position (0–1), Dark notch ratio (0–1)  
**Output:** Inversion confidence (0–1) → flip decision

### Module 2 — Grade Adjustment
Refines the model's raw grade prediction based on prediction confidence and image sharpness (Laplacian variance).

**Inputs:** Prediction confidence (0–1), Image sharpness (0–200)  
**Output:** Grade delta (−1 to +1)

---

##  Results

### Ablation Study

| System | QWK |
|--------|-----|
| EfficientNetB0 Softmax (Baseline) | 0.831 |
| EfficientNetB3 Ordinal (No CLAHE) | 0.856 |
| EfficientNetB3 + CLAHE + Fuzzy (No TTA) | 0.874 |
| **EfficientNetB3 + CLAHE + Fuzzy + TTA (This Work)** | **0.892** |
| APTOS Top-10 Ensemble (Reference) | 0.933 |

---

##  Tech Stack

- **Deep Learning:** TensorFlow 2.x, EfficientNetB3
- **Fuzzy Logic:** Pure NumPy (Mamdani inference, centroid defuzzification)
- **Backend:** Flask (Python 3.11)
- **Frontend:** Vanilla JS, Plotly.js
- **Image Processing:** OpenCV, CLAHE
- **Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)

---

##  Team

| Name | Role |
|------|------|
| **Jana Walid Eleskandrany** | Model Architecture · Fuzzy Logic · Backend |
| **Raneem Mohamed Mabrouk** | Research · Documentation · Testing |
| **Eng. Ibrahim El Gazar** | Teaching Assistant · Technical Guidance |
| **Prof. Mona Nagy El Bedwehy** | Course Instructor · Academic Supervision |

---

##  License

This project is an academic research submission. All rights reserved by the authors.

---

<p align="center">Built with ❤️ for the Soft Computing course · Faculty of Engineering</p>
