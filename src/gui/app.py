import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import sys

# --- CONFIGURAZIONE UTENTE ---
PREFER_REAL_MODELS = True 

# 1. PERCORSI MODELLI
YOLO_MODEL_PATH = "yolo12m.pt"
DINO_CHECKPOINT_PATH = "dinov3_large.pt"
DINO_REPO_PATH = "dinov3"

# 2. Parametri Image Processing
IMG_SIZE = 384
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID = (7, 7)
SHARPNESS_FACTOR = 2.0
CROP_MARGIN = 0.15
DINO_THRESHOLD = 0.38

# --- DEFINIZIONE CLASSE MODELLO ---
class ClassifierModel(nn.Module):
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

    def forward(self, x):
        out = self.backbone(x)
        feats = out[0] if isinstance(out, (tuple, list)) else out
        if feats.ndim == 3:
            cls = feats[:, 0, :]
        elif feats.ndim == 4:
            cls = feats.mean(dim=(2, 3))
        else:
            cls = feats
        return self.head(cls)

# --- FUNZIONI DI CARICAMENTO ---
def load_models_safely():
    print("\n" + "="*40)
    print("AVVIO PROCEDURA CARICAMENTO MODELLI")
    print("="*40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Device selezionato: {device}")
    
    if not PREFER_REAL_MODELS:
        print("[CONFIG] PREFER_REAL_MODELS è False -> Attivazione modalità MOCK.")
        return None, None, device, True

    try:
        # 1. Caricamento YOLO
        print(f"[YOLO] Cerco file in: {os.path.abspath(YOLO_MODEL_PATH)}")
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"File YOLO non trovato: {YOLO_MODEL_PATH}")
        
        from ultralytics import YOLO
        print(f"[YOLO] Caricamento modello in memoria...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("[YOLO] Caricato con successo!")

        # 2. Caricamento DINOv3 Backbone
        print(f"[DINO] Cerco repo in: {os.path.abspath(DINO_REPO_PATH)}")
        if not os.path.exists(DINO_REPO_PATH):
            raise FileNotFoundError(f"Cartella DINO non trovata: {DINO_REPO_PATH}")
        
        if DINO_REPO_PATH not in sys.path:
            print(f"[DINO] Aggiunta repo al PATH di sistema.")
            sys.path.append(DINO_REPO_PATH)
        
        # Caricamento dinamico Large/Base
        model_name = 'dinov3_vitl16' 
        print(f"[DINO] Caricamento architettura backbone ({model_name})...")
        try:
            backbone = torch.hub.load(DINO_REPO_PATH, model_name, source='local', pretrained=False)
        except RuntimeError:
            print(f"[DINO] {model_name} non trovato, provo 'dinov3_vitl14'...")
            model_name = 'dinov3_vitl14'
            backbone = torch.hub.load(DINO_REPO_PATH, model_name, source='local', pretrained=False)

        backbone.to(device).eval()
        print(f"[DINO] Backbone {model_name} inizializzato.")

        # Calcolo dimensione feature
        print("[DINO] Inferenza dummy per calcolo dimensioni feature...")
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            out = backbone(dummy)
            feats = out[0] if isinstance(out, (tuple, list)) else out
            feat_dim = feats.shape[-1] if hasattr(feats, 'ndim') and feats.ndim >= 2 else 1024
        print(f"[DINO] Feature Dimension rilevata: {feat_dim}")
        
        dino_model = ClassifierModel(backbone, feat_dim=feat_dim).to(device)

        # 3. Caricamento Pesi DINO
        print(f"[DINO] Caricamento pesi custom da: {DINO_CHECKPOINT_PATH}")
        if not os.path.exists(DINO_CHECKPOINT_PATH):
             raise FileNotFoundError(f"Checkpoint DINO non trovato: {DINO_CHECKPOINT_PATH}")

        ck = torch.load(DINO_CHECKPOINT_PATH, map_location='cpu')
        sd = ck.get('model_state_dict', ck)
        sd = {k[7:] if k.startswith('module.') else k:v for k,v in sd.items()}
        
        dino_model.load_state_dict(sd, strict=True)
        dino_model.eval()
        print("[DINO] Pesi caricati correttamente!")
        
        print("="*40)
        print(">>> TUTTI I MODELLI SONO PRONTI <<<")
        print("="*40 + "\n")
        return yolo_model, dino_model, device, False 

    except Exception as e:
        print("\n" + "!"*40)
        print("ERRORE CRITICO DURANTE IL CARICAMENTO")
        print(f"Messaggio: {e}")
        import traceback
        traceback.print_exc()
        print("!"*40)
        print("Fallback: Attivazione modalità DEMO (MOCK).\n")
        return None, None, device, True 

class ThyroidApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup Finestra
        self.title("Thyroid AI Diagnosis Pipeline")
        self.geometry("1200x850")
        ctk.set_appearance_mode("Dark")
        
        self.yolo, self.dino, self.device, self.is_mock = load_models_safely()
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # GUI Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="AI DIAGNOSIS", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        # Pulsante Carica
        self.btn_load = ctk.CTkButton(self.sidebar, text="Carica Ecografia", command=self.load_image)
        self.btn_load.pack(pady=10, padx=20)

        # Interruttore Filtri
        self.filter_switch = ctk.CTkSwitch(self.sidebar, text="Applica Filtri\n(CLAHE + Sharpness)")
        self.filter_switch.select() # Default: Attivo
        self.filter_switch.pack(pady=20, padx=20)

        status_text = "MODALITÀ DEMO" if self.is_mock else "SISTEMA ONLINE"
        status_color = "orange" if self.is_mock else "green"
        self.lbl_status = ctk.CTkLabel(self.sidebar, text=status_text, text_color=status_color, font=("Arial", 12, "bold"))
        self.lbl_status.pack(pady=10)

        self.textbox = ctk.CTkTextbox(self.sidebar, width=220, height=400)
        self.textbox.pack(pady=20, padx=10)
        self.textbox.insert("0.0", "Log Sistema inizializzato.\n")
        
        if self.is_mock:
            self.log("ATTENZIONE: Modelli non caricati.")
            self.log("Uso simulazione (Mock).")
        else:
            self.log("Sistema pronto. Modelli caricati su " + self.device)

        self.image_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_image_label = ctk.CTkLabel(self.image_frame, text="Carica un'immagine")
        self.main_image_label.pack(expand=True, fill="both")

    def log(self, message):
        print(f"[GUI] {message}")
        self.textbox.insert("end", f"> {message}\n")
        self.textbox.see("end")

    def preprocess_pipeline(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None, None

        # Controllo stato dello switch
        if self.filter_switch.get() == 1:
            self.log(f"Preprocessing: ON (CLAHE + Sharpening)")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
            enhanced_gray = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            img_pil = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_sharp = enhancer.enhance(SHARPNESS_FACTOR)
            
            return img_sharp, cv2.cvtColor(np.array(img_sharp), cv2.COLOR_RGB2BGR)
        else:
            self.log(f"Preprocessing: OFF (Immagine Originale)")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            return img_pil, img # img è già BGR, perfetto per OpenCV

    def get_crop_with_margin(self, img_array, box, margin=0.15):
        h, w = img_array.shape[:2]
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        pad_x = width * margin
        pad_y = height * margin
        new_x1 = max(0, int(x1 - pad_x))
        new_y1 = max(0, int(y1 - pad_y))
        new_x2 = min(w, int(x2 + pad_x))
        new_y2 = min(h, int(y2 + pad_y))
        return img_array[new_y1:new_y2, new_x1:new_x2], (new_x1, new_y1, new_x2, new_y2)

    def run_inference(self, img_path):
        pil_img, cv_img = self.preprocess_pipeline(img_path)
        if pil_img is None: 
            self.log("Errore lettura immagine.")
            return
        
        display_img = cv_img.copy()
        boxes = []
        
        # 1. DETECTION
        if self.is_mock:
            self.log("Simulazione Detection...")
            h, w, _ = cv_img.shape
            boxes = [[int(w*0.3), int(h*0.3), int(w*0.6), int(h*0.6)]]
        else:
            self.log("Esecuzione YOLOv12...")
            try:
                results = self.yolo(cv_img, verbose=False)
                for r in results:
                    for box in r.boxes.xyxy:
                        boxes.append(box.cpu().numpy().astype(int))
            except Exception as e:
                self.log(f"Errore inferenza YOLO: {e}")

        if not boxes:
            self.log("Nessun nodulo trovato.")
            
        # 2. CLASSIFICAZIONE
        for i, box in enumerate(boxes):
            crop_img, _ = self.get_crop_with_margin(cv_img, box, margin=CROP_MARGIN)
            
            if crop_img.size == 0: 
                self.log(f"Nodulo {i}: Crop non valido")
                continue

            diagnosis = "Unknown"
            prob = 0.0

            if self.is_mock:
                prob = 0.95
                diagnosis = "MALIGNO (SIM)"
            else:
                try:
                    crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits = self.dino(input_tensor)
                        prob = torch.sigmoid(logits).item()
                        diagnosis = "MALIGNO" if prob >= DINO_THRESHOLD else "BENIGNO"
                except Exception as e:
                    self.log(f"Errore DINO: {e}")
                    diagnosis = "ERROR"

            color = (0, 0, 255) if "MALIGNO" in diagnosis else (0, 255, 0)
            cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{diagnosis} {prob:.1%}"
            cv2.putText(display_img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self.log(f"Nodulo {i+1}: {label}")

        final_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        self.display_image(final_pil)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.dicom")])
        if file_path:
            self.log(f"--- File: {os.path.basename(file_path)} ---")
            self.after(100, lambda: self.run_inference(file_path))

    def display_image(self, pil_image):
        w_c = self.image_frame.winfo_width()
        h_c = self.image_frame.winfo_height()
        if w_c < 10: w_c, h_c = 800, 600
        
        ratio = min(w_c/pil_image.width, h_c/pil_image.height)
        new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
        
        # Usa CTkImage per supporto HighDPI e zero warning
        ctk_img = ctk.CTkImage(light_image=pil_image, 
                               dark_image=pil_image, 
                               size=new_size)
        
        self.main_image_label.configure(image=ctk_img, text="")
        self.main_image_label.image = ctk_img

if __name__ == "__main__":
    app = ThyroidApp()
    app.mainloop()