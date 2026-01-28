#!/usr/bin/env python3
"""
CLI inference script for Thyroid-Nodule-AI (DINOv3 classification only).

Usage examples:
  python src/inference_dino.py --image assets/example_nodule.png --weights models/dinov3_finetuned.pt
  python src/inference_dino.py --input-dir ./assets/crops --weights models/dinov3_finetuned.pt --out-dir results/

This script:
 - loads a DINOv3 backbone from a local repo (torch.hub)
 - wraps it with a small classification head (same structure as in your GUI)
 - runs inference on cropped nodules
 - computes a saliency heatmap (tries feature-grad Grad-CAM-like method, falls back to input-gradient)
 - saves overlays and prints predictions
"""

import argparse
import os
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# -------------------- USER DEFAULT CONFIG (tweak if needed) --------------------
DEFAULT_IMG_SIZE = 384
DEFAULT_DINO_REPO = "dinov3"       # local folder containing the dinov3 repo (used with torch.hub)
DEFAULT_MODEL_NAME = "dinov3_vitl16"
DEFAULT_THRESHOLD = 0.38
# ------------------------------------------------------------------------------

CLASS_NAMES = ["Benign", "Malignant"]

class ClassifierModel(nn.Module):
    """Head + (optionally) backbone wrapper (we keep flexible usage)."""
    def __init__(self, backbone, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward_from_feats(self, feats):
        # feats can be: (B, N, C) tokens OR (B, C, H, W) conv-like OR (B, C)
        if feats.ndim == 3:
            # (B, N, C) -> take cls token (first token) if present else mean tokens
            # Many DINO variants return (B, N, C) with a [CLS] token at 0
            cls = feats[:, 0, :]  # try cls token
        elif feats.ndim == 4:
            # (B, C, H, W) -> global average pool
            cls = feats.mean(dim=(2, 3))
        else:
            cls = feats
        return self.head(cls).squeeze(1)  # returns (B,)

# -------------------- HELPERS --------------------
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.5):
    """heatmap: float array [0..1], single channel same spatial size as orig_bgr"""
    heatmap_uint = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------- MODEL LOADING --------------------
def load_dino_backbone(dino_repo_path, model_name, device):
    """
    Load backbone using torch.hub from local repo (same approach as GUI code).
    Returns backbone (callable).
    """
    if dino_repo_path not in (p := list(map(str, sys_path := []))):
        pass  # placeholder to satisfy static analyzer (we modify below)

    import sys
    if dino_repo_path not in sys.path:
        sys.path.append(dino_repo_path)

    try:
        backbone = torch.hub.load(dino_repo_path, model_name, source='local', pretrained=False)
    except Exception as e:
        # fallback: try other model name variants
        alt = "dinov3_vitl14" if model_name.endswith("16") else "dinov3_vitl16"
        backbone = torch.hub.load(dino_repo_path, alt, source='local', pretrained=False)
    backbone.to(device).eval()
    return backbone

def build_full_model(dino_repo, model_name, weights_path, device, img_size):
    # Load backbone
    backbone = load_dino_backbone(dino_repo, model_name, device)

    # Determine feature dimension via a dummy forward
    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        out = backbone(dummy)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        # infer last dim as feature dim (token channels or C)
        if feats.ndim == 3:
            feat_dim = feats.shape[-1]
        elif feats.ndim == 4:
            feat_dim = feats.shape[1]
        else:
            feat_dim = feats.shape[-1] if feats.ndim >= 1 else 1024

    # Create classifier wrapper (we will often call backbone separately for saliency)
    model = ClassifierModel(backbone, feat_dim).to(device)

    # Load checkpoint
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    ck = torch.load(weights_path, map_location='cpu')
    sd = ck.get('model_state_dict', ck)
    # strip 'module.' if present
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

    # Try to load into model.head only (common case: checkpoint saved head weights only)
    loaded = False
    try:
        # If checkpoint keys match entire model, load strict
        model.load_state_dict(sd, strict=True)
        loaded = True
    except Exception:
        # try load only head keys
        head_state = {k.replace('head.', ''): v for k, v in sd.items() if k.startswith('head.')}
        if head_state:
            model.head.load_state_dict(head_state, strict=False)
            loaded = True
    if not loaded:
        # try to load as-is into head or leave as initialized
        print("[WARN] Couldn't strictly load checkpoint into model; head may be partially initialized.")

    model.eval()
    return model

# -------------------- SALIENCY / GRAD-CAM (feature-grad and fallback) --------------------
def compute_feature_gradcam(backbone, head, input_tensor, target_idx=None):
    """
    Try feature-grad Grad-CAM-like approach:
     - run backbone -> feats
     - retain grad on feats, forward head from feats to get logit
     - backward on the logit -> collect feats.grad
     - produce spatial map: if feats are tokens, reshape to sqrt(N) grid; if conv-like, pool channels
    Returns heatmap resized to input spatial size (H,W) float [0..1]
    """
    device = input_tensor.device
    backbone.zero_grad()
    head.zero_grad()

    # 1) forward backbone
    out = backbone(input_tensor)
    feats = out[0] if isinstance(out, (tuple, list)) else out  # e.g. (B, N, C) or (B, C, H, W) or (B, C)
    feats.requires_grad_(True)
    feats.retain_grad()

    # 2) compute logits from feats using head (reuse head forward logic)
    if feats.ndim == 3:
        cls_feats = feats[:, 0, :]  # try CLS token
    elif feats.ndim == 4:
        cls_feats = feats.mean(dim=(2, 3))
    else:
        cls_feats = feats
    logits = head(cls_feats).squeeze(1)  # (B,)
    # take first batch only (we handle single image mostly)
    target_logit = logits[0] if logits.ndim == 0 or logits.shape[0] == 1 else (logits[0] if target_idx is None else logits[0])

    # backward
    backbone.zero_grad()
    head.zero_grad()
    target_logit.backward(retain_graph=False)

    grads = feats.grad  # may be None if graph disconnected
    if grads is None:
        return None

    # generate map
    if feats.ndim == 4:
        # (B, C, H, W)
        grad = grads[0]  # (C,H,W)
        weights = grad.mean(dim=(1,2), keepdim=True)  # (C,1,1)
        cam = (weights * feats[0]).sum(dim=0).cpu().detach().numpy()  # (H,W)
        cam = np.maximum(cam, 0)
    elif feats.ndim == 3:
        # (B, N, C) tokens -> try reshape tokens to grid
        tok = feats[0].cpu().detach().numpy()  # (N,C)
        g = grads[0].cpu().detach().numpy()     # (N,C)
        # weights per token: mean over channels
        token_weights = g.mean(axis=1)  # (N,)
        token_map = token_weights * (tok.mean(axis=1))  # any reasonable aggregation
        N = token_map.shape[0]
        s = int(math.sqrt(N))
        if s * s == N:
            cam = token_map.reshape(s, s)
            cam = np.maximum(cam, 0)
        else:
            # cannot reshape tokens -> abort
            return None
    else:
        return None

    # normalize cam to [0,1] and resize to input size
    cam = cam - cam.min() if cam.max() != cam.min() else cam * 0.0
    cam = cam / (cam.max() + 1e-12)
    cam = cv2.resize(cam.astype(np.float32), (input_tensor.shape[-1], input_tensor.shape[-2]))
    return cam

def compute_input_gradients_saliency(model, input_tensor, target_is_malignant=True):
    """Fallback: compute gradient of logit w.r.t. input image and use absolute gradient map."""
    model.zero_grad()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    # run a forward: we need to call backbone+head to get logit
    # assume model.backbone exists
    out = model.backbone(input_tensor)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    if feats.ndim == 3:
        cls = feats[:, 0, :]
    elif feats.ndim == 4:
        cls = feats.mean(dim=(2,3))
    else:
        cls = feats
    logits = model.head(cls).squeeze(1)
    logit = logits[0]
    # if target_is_malignant True, maximize logit (assume higher means malignant)
    target = logit if target_is_malignant else -logit
    target.backward()
    grads = input_tensor.grad.detach().cpu().numpy()[0]  # (C,H,W)
    saliency = np.mean(np.abs(grads), axis=0)
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-12)
    return saliency

# -------------------- INFERENCE PIPELINE --------------------
def apply_enhancement(img_bgr):
    # 1. Converti in LAB per applicare CLAHE solo sulla luminanza (L)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. Applica CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # 3. Ricomponi l'immagine
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 4. Sharpening (Unsharp Masking)
    # Formula: Enhanced = Original + (Original - Blurred) * Amount
    gaussian = cv2.GaussianBlur(img_eq, (0, 0), 3.0)
    img_sharp = cv2.addWeighted(img_eq, 1.5, gaussian, -0.5, 0)

    return img_sharp

def preprocess_image_cv2(path, img_size):
    # Leggi l'immagine con OpenCV
    img_cv = cv2.imread(path)
    if img_cv is None:
        raise ValueError(f"Image not found: {path}")

    # --- APPLICAZIONE FILTRI (CLAHE + SHARPENING) ---
    img_enhanced = apply_enhancement(img_cv)

    # Converti in RGB per PIL/PyTorch
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    # Trasformazioni standard per il modello (Resize + Normalize)
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensor = tfm(pil).unsqueeze(0)
    
    # Ritorna sia l'immagine PIL (per visualizzazione) che il tensore (per inferenza)
    return pil, tensor

def single_image_inference(model, image_path, device, out_dir, threshold, save_heatmap=True):
    orig_pil, input_tensor = preprocess_image_cv2(image_path, DEFAULT_IMG_SIZE)
    input_tensor = input_tensor.to(device)

    # Try feature-grad cam first
    try:
        backbone = model.backbone
        head = model.head
        cam = compute_feature_gradcam(backbone, head, input_tensor)
    except Exception:
        cam = None

    if cam is None:
        # fallback to input-gradient saliency
        cam = compute_input_gradients_saliency(model, input_tensor, target_is_malignant=True)

    # Forward for prediction
    with torch.no_grad():
        out = model.backbone(input_tensor)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        if feats.ndim == 3:
            cls_feats = feats[:, 0, :]
        elif feats.ndim == 4:
            cls_feats = feats.mean(dim=(2,3))
        else:
            cls_feats = feats
        logits = model.head(cls_feats).squeeze(1)
        prob = torch.sigmoid(logits).item()
        pred_label = "MALIGNANT" if prob >= threshold else "BENIGN"

    # Prepare outputs
    orig_cv2 = pil_to_cv2(orig_pil)
    heatmap_resized = cv2.resize(cam, (orig_cv2.shape[1], orig_cv2.shape[0]))
    overlay = overlay_heatmap_on_image(orig_cv2, heatmap_resized, alpha=0.45)

    # annotate text
    label = f"{pred_label} {prob:.1%}"
    color = (0,0,255) if "MALIGNANT" in pred_label else (0,255,0)
    cv2.putText(overlay, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # save
    out_path = out_dir / (Path(image_path).stem + "_overlay.png")
    cv2.imwrite(str(out_path), overlay)
    print(f"[RESULT] {image_path} -> {out_path}  | Prediction: {pred_label} ({prob:.3f})")
    return out_path, prob, pred_label

# -------------------- CLI ENTRYPOINT --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, help="Path to single image (cropped nodule).")
    p.add_argument("--input-dir", type=str, help="Path to a folder with cropped images.")
    p.add_argument("--weights", type=str, required=True, help="Path to DINO fine-tuned weights (.pt).")
    p.add_argument("--dino-repo", type=str, default=DEFAULT_DINO_REPO, help="Local path to dinov3 repo for torch.hub.")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model name to load from repo (torch.hub).")
    p.add_argument("--out-dir", type=str, default="results", help="Directory to save overlays.")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda. Default auto-detect.")
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Using device: {device}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("[SYSTEM] Loading model (this may take a while)...")
    model = build_full_model(args.dino_repo, args.model_name, args.weights, device, args.img_size)

    # Inference on single image or directory
    images = []
    if args.image:
        images = [args.image]
    elif args.input_dir:
        p = Path(args.input_dir)
        images = [str(p / f) for f in sorted(os.listdir(p)) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
    else:
        raise ValueError("Either --image or --input-dir must be provided.")

    for img_path in images:
        try:
            single_image_inference(model, img_path, device, out_dir, args.threshold)
        except Exception as e:
            print(f"[ERROR] while processing {img_path}: {e}")

if __name__ == "__main__":
    main()
