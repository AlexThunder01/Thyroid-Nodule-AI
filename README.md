# 🩺 Thyroid Nodule CAD Pipeline: CNNs vs. Vision Transformers

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v12-00FFFF?logo=yolo&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)
![Thesis](https://img.shields.io/badge/Type-Bachelor_Thesis-orange)

> **Official Repository** for the Bachelor's Thesis:  
> *"Sviluppo di una pipeline di Deep Learning per la diagnosi di noduli tiroidei in ecografia: confronto tra architetture CNN e Vision Transformers"* > **Sapienza University of Rome** (2024/2025)

## 📑 Table of Contents
- [Abstract](#-abstract)
- [Key Features](#-key-features)
- [Dataset & Pre-processing](#-dataset--pre-processing)
- [Methodology](#-methodology)
  - [Stage 1: Object Detection](#stage-1-object-detection)
  - [Stage 2: Classification](#stage-2-classification)
- [Results](#-results)
- [Demo App](#-thyroid-ai-assistant-demo)
- [Installation](#-installation)
- [Citation](#-citation)

---

## 📋 Abstract
Thyroid nodules are a common pathology, but accurately distinguishing between benign and malignant cases in ultrasound images is challenging due to speckle noise and operator subjectivity.

This project proposes a **two-stage Deep Learning pipeline** to automate this diagnosis. We conducted a comparative study between traditional **Convolutional Neural Networks (CNNs)** and modern **Vision Transformers (ViTs)** / **Foundation Models**. The study proves that Self-Supervised Learning (SSL) models like **DINOv3** significantly outperform supervised baselines in medical imaging tasks characterized by data scarcity and noise.

## 🚀 Key Features
* **Advanced Architectures:** Comparison of **YOLOv12**, **DINO-DETR**, **Faster R-CNN**, **DINOv3**, **ConvNeXt V2**, and **EfficientNetV2**.
* **Foundation Models:** Utilization of DINOv3 (Self-Supervised) pre-trained on massive datasets (LVD-142M) to improve feature extraction.
* **Robust Data Pipeline:** Automated deduplication via **Perceptual Hashing** (dHash) to prevent data leakage.
* **Image Enhancement:** Implementation of a pre-processing pipeline using **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and **Sharpening**.
* **Explainable AI:** Integration of **Saliency Maps** to validate model decisions against **TI-RADS** clinical criteria (e.g., irregular margins, microcalcifications).
* **GUI Prototype:** A fully functional desktop application ("Thyroid AI Assistant") for real-time inference.

---

## 💾 Dataset & Pre-processing

### Data Sources
The study aggregates data from multiple open-source datasets to ensure heterogeneity:
1.  **TN5000 (Thyroid Nodule 5000):** Biopsy-validated labels and high-quality segmentation masks.
2.  **AUITD:** Additional data to introduce variability in ultrasound devices (Toshiba/Samsung).

**Total volume:** ~7,000 nodules (Split: 80% Train, 10% Val, 10% Test).

### Cleaning & Enhancement
* **Deduplication:** Applied Hamming Distance on Perceptual Hashes to remove near-duplicate frames.
* **Enhancement:**
    * *CLAHE:* Improves local contrast.
    * *Sharpening:* Emphasizes nodule boundaries (crucial for malignancy detection).

---

## 🛠 Methodology

The pipeline consists of two distinct stages:

### Stage 1: Object Detection
**Goal:** Localize the nodule in the ultrasound frame.
* **Models Tested:** YOLOv12 (Small/Medium/Large), DINO-DETR (ResNet/Swin backbones), Faster R-CNN.
* **Technique:** Real-time anchor-free detection vs. Two-stage detection.

### Stage 2: Classification
**Goal:** Classify the localized crop as **Benign** or **Malignant**.
* **Models Tested:** DINOv3 (ViT), ConvNeXt V2, EfficientNetV2, SVM (Radiomics baseline).
* **Technique:** Full Fine-tuning with context padding (10-15% margin) to include the peri-nodular tissue (halo sign).

---

## 📊 Results

### 🏆 Detection Performance
**Winner:** `YOLOv12-Medium` achieved the best balance of speed and recall.

| Model | mAP@50 | Recall | Inference Time |
| :--- | :---: | :---: | :---: |
| **YOLOv12-m** | **95.20%** | **92.97%** | **< 30ms** |
| DINO-DETR (Swin-B) | 90.45% | 85.52% | ~440ms |
| Faster R-CNN | 89.94% | 81.20% | ~120ms |

### 🏆 Classification Performance
**Winner:** `DINOv3-Large` set the new state-of-the-art for this dataset.

| Model | Architecture | AUC-ROC | Sensitivity (Recall) |
| :--- | :--- | :---: | :---: |
| **DINOv3-Large** | **Foundation Model (SSL)** | **0.932** | **94.70%** |
| DINOv3-Base | Foundation Model (SSL) | 0.930 | 91.71% |
| ConvNeXt V2 | Modern CNN | 0.906 | 91.71% |
| EfficientNetV2 | CNN | 0.898 | 92.63% |
| SVM | Hand-crafted Radiomics | 0.814 | 71.26% |

> **Clinical Insight:** DINOv3's attention maps accurately follow irregular margins and microcalcifications, showing a 90.1% alignment with human TI-RADS risk scoring on external validation.

---

## 🖥 Thyroid AI Assistant (Demo)
A custom GUI built with `CustomTkinter` allows for drag-and-drop inference.

**Features:**
* Toggle Image Enhancement (CLAHE).
* Real-time Detection & Classification.
* Risk Probability visualization.

*(Place screenshot here, e.g., `docs/gui_preview.png`)*

---

## ⚙ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/username/thyroid-nodule-detection.git](https://github.com/username/thyroid-nodule-detection.git)
    cd thyroid-nodule-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Pre-trained Weights:**
    * *Note: Due to file size limits, weights are hosted externally.*
    * Download `yolov12_medium.pt` and `dinov3_large_finetuned.pth` from [Release Page](#).

4.  **Run the Demo:**
    ```bash
    python src/gui/app.py
    ```

---

## 📖 Citation

If you use this work, please cite the thesis:

```bibtex
@thesis{catania2025thyroid,
  author       = {Alessandro Catania},
  title        = {Sviluppo di una pipeline di Deep Learning per la diagnosi di noduli tiroidei in ecografia: confronto tra architetture CNN e Vision Transformers},
  school       = {Sapienza Università di Roma},
  year         = {2025},
  type         = {Bachelor's Thesis}
}
