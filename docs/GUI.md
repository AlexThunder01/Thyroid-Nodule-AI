# GUI — Thyroid AI Diagnosis Pipeline

This document describes the **Graphical User Interface (GUI)** of the
*Thyroid-Nodule-AI* project and explains **how the GUI works internally**,
including **mandatory configuration steps** required to run the real models.

⚠️ **Important**
The GUI will **NOT run real inference** unless **valid model weight paths
are explicitly provided by the user**.  
If model files are missing or misconfigured, the GUI automatically switches
to **Mock (Demo) Mode**.

---

## Purpose of the GUI

The GUI is designed as:
- an **interactive demonstration tool**,
- a **debugging interface** for the CAD pipeline,
- a **qualitative evaluation tool** for thesis experiments.

It integrates:
- detection,
- ultrasound-specific preprocessing,
- classification,
- visual feedback and logging.

⚠️ The GUI is **not a medical device** and must not be used for clinical
decision making.

---

## Mandatory user configuration (VERY IMPORTANT)

Before running the GUI in **real inference mode**, the user **must manually
set the correct paths to the model weights** at the top of the script.

### Required parameters

```python
PREFER_REAL_MODELS = True

YOLO_MODEL_PATH = "yolo12m.pt"
DINO_CHECKPOINT_PATH = "dinov3_large.pt"
DINO_REPO_PATH = "dinov3"
````

### Meaning of each path

| Variable               | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `YOLO_MODEL_PATH`      | Path to YOLOv12 detection weights                |
| `DINO_CHECKPOINT_PATH` | Path to fine-tuned DINOv3 classification weights |
| `DINO_REPO_PATH`       | Local clone of the DINOv3 repository             |

⚠️ **If any of these paths is incorrect or missing, the GUI will not load the models.**

---

## Automatic fallback behavior (Mock Mode)

The GUI is intentionally designed to **never crash**.

If one of the following occurs:

* weight files are not found,
* DINOv3 repository path is invalid,
* a runtime error happens during loading,
* `PREFER_REAL_MODELS = False`,

then the system automatically activates:

> 🟠 **Mock (Demo) Mode**

In this mode:

* detection boxes are simulated,
* classification probabilities are fake,
* the GUI remains fully interactive.

This allows:

* UI testing,
* thesis demonstrations without heavy models,
* development on machines without GPU support.

---

## GUI layout

The interface is divided into two main areas.

### Sidebar (left panel)

Contains:

* application title
* **Load Image** button
* preprocessing ON/OFF switch
* system status indicator:

  * `SYSTEM ONLINE` → real models loaded
  * `DEMO MODE` → mock inference
* real-time log console

### Main panel (right area)

Displays:

* the loaded ultrasound image
* bounding boxes
* diagnosis label
* probability score

---

## Model loading pipeline

Model loading is handled by:

```python
load_models_safely()
```

### Internal logic

1. **Device selection**

   * CUDA if available
   * CPU otherwise

2. **YOLOv12 loading**

   * Model loaded from `YOLO_MODEL_PATH`
   * Used for nodule detection

3. **DINOv3 backbone loading**

   * Architecture dynamically loaded from `DINO_REPO_PATH`
   * Supports ViT-L/16 and ViT-L/14 variants

4. **Classifier head construction**

   * MLP head with dropout
   * Binary output (logit)

5. **Fine-tuned weight loading**

   * Loaded from `DINO_CHECKPOINT_PATH`
   * Automatically removes `module.` prefixes if present

If **any step fails**, the function:

* logs the error,
* switches to Mock Mode,
* keeps the GUI running.

---

## Image preprocessing pipeline

Controlled by the GUI switch:

> **“Apply Filters (CLAHE + Sharpness)”**

### When enabled:

* grayscale conversion
* **CLAHE** contrast normalization
* RGB reconstruction
* **sharpness enhancement**

Parameters:

* CLAHE clip limit: `1.0`
* tile grid: `(7, 7)`
* sharpness factor: `2.0`

### When disabled:

* original image is used
* no enhancement applied

All preprocessing steps are logged in real time.

---

## Detection stage

### Real mode

* YOLOv12 runs on the input image
* all detected bounding boxes are processed

### Mock mode

* a synthetic bounding box is generated
* used to simulate a detected nodule

If no boxes are found, the GUI reports:

```
No nodules detected.
```

---

## Crop extraction

For each detected box:

* a margin is applied (`CROP_MARGIN = 0.15`)
* crop is clipped to image boundaries
* invalid crops are skipped safely

This improves robustness against detection inaccuracies.

---

## Classification stage

For each valid crop:

1. resize to `384 × 384`
2. normalize using ImageNet statistics
3. forward pass through DINOv3 classifier
4. sigmoid activation → probability

### Decision threshold

```python
DINO_THRESHOLD = 0.38
```

| Probability | Diagnosis |
| ----------- | --------- |
| ≥ threshold | MALIGNANT |
| < threshold | BENIGN    |

---

## Visualization

* bounding boxes are drawn on the image
* color coding:

  * **red** → malignant
  * **green** → benign
* probability score is shown next to each box

Rendering uses `CTkImage` to ensure:

* High-DPI support
* no Tkinter warnings

---

## Logging system

The GUI includes a real-time logging console showing:

* model loading status
* preprocessing actions
* inference results
* errors and fallback activation

Logs are printed both:

* inside the GUI
* to the terminal

---

## Supported image formats

The file dialog accepts:

* `.jpg`
* `.jpeg`
* `.png`
* `.bmp`
* `.dicom` (if readable by OpenCV)

---

## Intended use

This GUI is intended for:

* thesis demonstrations
* qualitative result inspection
* debugging and development
* educational purposes

It is **not intended for clinical use**.

---

## Summary

✔️ The GUI provides a full visual interface to the CAD pipeline
✔️ **Model paths must be configured manually by the user**
✔️ Missing weights automatically trigger Mock Mode
✔️ No patient data is included

---

**Project:** Thyroid-Nodule-AI
**Author:** Alessandro Catania