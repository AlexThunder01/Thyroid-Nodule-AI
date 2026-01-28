# Model Card — Thyroid Nodule Classification (DINOv3)

## Model Overview

This model is a **fine-tuned DINOv3 Vision Transformer** designed to classify **cropped thyroid nodule ultrasound images** as:

- **Benign**
- **Malignant**

It is part of a research project exploring AI-assisted analysis of thyroid ultrasound imaging with visual explainability.

The model operates **only on pre-cropped nodules** and does not perform detection or segmentation.

---

## Intended Use

### Primary Use
This model is intended for:

- AI research and experimentation  
- Educational purposes  
- Computer vision benchmarking  
- Explainability research (Grad-CAM visualization)

### Users
- Machine learning researchers  
- Students in medical imaging / AI  
- Developers exploring vision transformers in healthcare  

### Out-of-Scope Use ⚠️
This model is **NOT intended for**:
- Clinical diagnosis  
- Medical decision-making  
- Use in real patient care  
- Screening or triage applications  

It is **not a medical device** and has **not been clinically validated**.

---

## Model Architecture

- Backbone: **DINOv3 Vision Transformer (ViT)**
- Input: RGB ultrasound image of a **pre-cropped thyroid nodule**
- Input Size: 224 × 224 pixels
- Output: Binary classification (Benign / Malignant)
- Explainability: Grad-CAM heatmap visualization

---

## Training Data

The model was trained on a dataset of **thyroid ultrasound images** containing manually or semi-automatically cropped nodules.

### Data Characteristics
- Imaging modality: Ultrasound
- Region: Thyroid gland
- Task: Nodule malignancy classification
- Labels: Benign / Malignant (based on ground truth annotations available in the dataset)

### Important Notes
- Dataset size and composition may not represent the full diversity of real-world clinical populations.
- Data may be biased toward specific acquisition devices, imaging protocols, or demographics.

---

## Performance

Performance was evaluated on a held-out test set of cropped nodules.

| Model            | AUC      | Accuracy | Sensitivity | Specificity |
| **DINOv3-Large** | **0.93** | **0.88** | **0.90**    | **0.86**    |

⚠️ These results are **research metrics only** and do not imply clinical reliability.

---

## Explainability

Grad-CAM is used to generate visual heatmaps highlighting image regions that most influence the model’s decision.

These visualizations are intended for:
- Interpretability research  
- Educational demonstrations  

They should **not** be interpreted as clinically validated localization.

---

## Limitations

This model has several important limitations:

- Works **only on pre-cropped nodules** (no automatic detection)
- Performance may degrade on:
  - Different ultrasound machines  
  - Different imaging protocols  
  - Underrepresented patient groups  
- May learn **spurious correlations** unrelated to pathology
- Grad-CAM explanations are approximate and not medically validated
- No robustness guarantees to noise, artifacts, or adversarial perturbations

---

## Ethical Considerations

AI models in medical imaging carry significant risks:

- Potential bias across demographic groups  
- False positives leading to anxiety or unnecessary follow-ups  
- False negatives leading to missed diagnoses  

This project emphasizes **responsible AI research** and explicitly prohibits clinical deployment.

---

## Safety Notice

🚨 **This model must not be used for clinical diagnosis or medical decision-making.**  
It is a research prototype and lacks regulatory approval, clinical validation, and real-world testing.

---

## Reproducibility

To reproduce inference results:

1. Download DINOv3 fine-tuned weights (see `docs/WEIGHTS.md`)
2. Use cropped thyroid nodule images
3. Run:

```bash
python src/inference_dino.py --image path_to_image.png --weights models/dinov3_finetuned.pt
