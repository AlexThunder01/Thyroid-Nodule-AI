# Thyroid Nodule CAD Pipeline: CNNs vs. Vision Transformers

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c)
![YOLOv12](https://img.shields.io/badge/YOLO-v12-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-yellow)

## 📋 Abstract
This repository hosts a comprehensive Deep Learning pipeline for the **Computer-Aided Diagnosis (CAD)** of thyroid nodules in ultrasound images. The project investigates the shift from traditional Convolutional Neural Networks (CNNs) to modern **Vision Transformers (ViTs)** and **Foundation Models**.

The pipeline is designed as a two-stage system:
1.  **Real-time Detection:** Localizing nodules using Anchor-Free and Transformer-based detectors.
2.  **Risk Classification:** Stratifying nodules (Benign vs. Malignant) using Self-Supervised Learning (SSL) representations.

The study utilizes a curated dataset of over **7,000 nodules** (TN5000 + AUITD) processed with perceptual hashing deduplication and image enhancement techniques.

## 🚀 Key Features
*   **State-of-the-Art Architectures:** Benchmarking **YOLOv12**, **DINO-DETR**, and **Faster R-CNN** for detection; **DINOv3**, **ConvNeXt V2**, and **EfficientNetV2** for classification.
*   **Foundation Models:** Leveraging **DINOv3 (Self-Supervised Learning)** to handle speckle noise and visual ambiguity better than supervised CNNs.
*   **Robust Pre-processing:** Automated deduplication via **Perceptual Hashing** and image enhancement using **CLAHE** and **Sharpening**.
*   **Explainability:** Integrated **Saliency Maps** and **Attention Maps** to visualize morphological features (margins, microcalcifications) aligned with **TI-RADS** criteria.
*   **GUI Demo:** A functional desktop application ("Thyroid AI Assistant") for real-time inference.

## 🛠️ Methodology & Pipeline

### 1. Data Preparation
- **Datasets:** Aggregated from TN5000 and AUITD.
- **Deduplication:** Hamming distance on Difference Hash (dHash) to prevent data leakage.
- **Enhancement:** Adaptive Histogram Equalization (CLAHE) + Unsharp Masking.

### 2. Stage 1: Object Detection
Comparison of One-Stage vs. Two-Stage detectors.
- **Best Performer:** **YOLOv12 (Medium)**
- **Optimization:** Anchor-free detection with attention modules.
- **Input:** Static Letterboxing (640x640).

### 3. Stage 2: Classification (Benign vs. Malignant)
Crop-based classification with context padding (10-15%).
- **Best Performer:** **DINOv3-Large** (Foundation Model).
- **Training Strategy:** Full Fine-Tuning of pre-trained SSL weights (LVD-142M).
- **Loss:** Weighted Binary Cross-Entropy to handle class imbalance.

## 📊 Experimental Results

### Object Detection (Test Set)
| Model | mAP@50 | Recall | Inference Speed |
| :--- | :---: | :---: | :---: |
| **YOLOv12-m** | **95.20%** | **92.97%** | **< 30ms (Real-time)** |
| DINO-DETR (Swin-B)| 90.45% | 85.52% | High Latency |
| Faster R-CNN | 89.94% | 81.20% | High False Positives |

### Classification (Test Set)
| Model | Architecture | AUC-ROC | Sensitivity (Recall) |
| :--- | :--- | :---: | :---: |
| **DINOv3-Large** | **ViT (SSL)** | **0.932** | **94.70%** |
| DINOv3-Base | ViT (SSL) | 0.930 | 91.71% |
| ConvNeXt V2 | Modern CNN | 0.906 | 91.71% |
| EfficientNetV2 | CNN | 0.898 | 92.63% |
| SVM (Baseline) | Radiomics | 0.814 | 71.26% |

> **Insight:** The Foundation Model (DINOv3) significantly outperforms CNNs in capturing subtle morphological patterns like microlobulated margins, demonstrating high robustness to ultrasound speckle noise.

## 💻 Tech Stack
*   **Language:** Python 3.x
*   **Deep Learning:** PyTorch, Ultralytics (YOLO), Detrex/Detectron2 (Transformers)
*   **Data Processing:** NumPy, OpenCV, scikit-image, ImageHash
*   **Visualization:** Matplotlib, Seaborn, Grad-CAM
*   **GUI:** CustomTkinter

## 📷 Demo Application
*Includes a prototype GUI for clinical workflow simulation.*

![GUI Screenshot](interface.png)

## 📄 Reference
This repository is part of the Master's Thesis: *"Development of a Deep Learning pipeline for thyroid nodule diagnosis in ultrasound: comparison between CNN and Vision Transformers architectures"* (Sapienza University of Rome, 2024/2025).

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
