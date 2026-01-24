# Pretrained Weights — Distribution and Licensing

This document describes the **pretrained and fine-tuned model weights**
associated with the *Thyroid-Nodule-AI* project, together with their
licensing terms and redistribution policies.

⚠️ **Important**  
Only weights that can be legally redistributed for research purposes
are provided. Users are responsible for complying with the original
licenses of all third-party models.

---

## Overview

| Component        | Included | Distribution method | License |
|------------------|----------|---------------------|---------|
| DINOv3 (fine-tuned) | ✅ Yes | Hugging Face Hub | Meta AI DINOv3 License |
| YOLOv12 (detector) | ❌ No  | User-provided only | GNU AGPL-3.0 |
| Datasets (TN5000) | ❌ No  | External access only | See original publication |

---

## DINOv3 Fine-Tuned Weights (Included)

### Description

This project provides **fine-tuned weights of DINOv3-Large**, trained for
**binary thyroid nodule classification (benign vs malignant)** using
ultrasound images.

These weights are a **derivative work** of the original DINOv3 backbone
released by **Meta AI**.

### Files

- `dinov3_large_thyroid_finetuned.pth`

### Distribution

The weights are distributed via:

- **Hugging Face Hub**  
[👉 Model on Hugging Face](https://huggingface.co/<USERNAME>/dinov3-large-thyroid-nodule)





The weights are **not stored directly in the Git repository** to keep
the repository lightweight and version-controlled.

### License

The original DINOv3 model is released by **Meta AI** under the
**DINOv3 License**.

Users of these fine-tuned weights must comply with the original license
terms provided by Meta AI:

👉 https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

These weights are released for:

- ✔️ Research use  
- ✔️ Educational use  

They are **NOT** intended for:

- ❌ Clinical diagnosis  
- ❌ Medical decision making  
- ❌ Commercial deployment  

---

### Attribution requirement

If you use these weights in academic work, you must:

- Cite Meta AI for the original DINOv3 model
- Cite this repository for the fine-tuned model

Example attribution:

> “The classification model is based on DINOv3 (Meta AI) and was fine-tuned
> for thyroid ultrasound analysis using the Thyroid-Nodule-AI pipeline.”

---


## YOLOv12 Detection Weights (Not Included)

### Reason for exclusion

YOLOv12 models are licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

Including YOLOv12 weights in this repository would impose **strong copyleft
obligations** on the entire project.

For this reason:

* ❌ YOLOv12 weights are **not distributed**
* ❌ YOLOv12 checkpoints are **not bundled**

### User responsibility

Users who wish to use YOLOv12 must:

1. Obtain the weights independently
2. Accept the AGPL-3.0 license terms
3. Ensure license compatibility with their intended use

License reference:
👉 [https://roboflow.com/model-licenses/yolov12](https://roboflow.com/model-licenses/yolov12)

---

## Dataset Weights / Files (Not Included)

### TN5000 dataset

The **TN5000 thyroid ultrasound dataset** is **not redistributed** in this repository.

Dataset access and licensing are governed by the original publication:

👉 [https://www.nature.com/articles/s41597-025-05757-4](https://www.nature.com/articles/s41597-025-05757-4)

Users must:

* Request or download the dataset from official sources
* Respect all ethical and legal constraints related to medical data

---

## Summary

✔️ This repository distributes **only** legally redistributable
fine-tuned DINOv3 weights
✔️ All third-party licenses are explicitly documented
✔️ No patient data or restricted datasets are included

For questions regarding weights or licensing, open an issue or contact
the repository maintainer.

---

**Maintainer:** Alessandro Catania
**Project:** Thyroid-Nodule-AI
