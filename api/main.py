# api/main.py

import os
import sys
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms

# --- Make project root importable ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.vit_backbone import ViTForBreastCancer
from models.risk_head import risk_from_prob

app = FastAPI(title="Breast-Path-ViT API")

# Allow Streamlit frontend (running on another port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model once at startup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForBreastCancer()
# If you have a trained checkpoint, load it here:
# checkpoint_path = os.path.join(PROJECT_ROOT, "saved_models", "vit_breakhis_best.pth")
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def read_root():
    return {"message": "Breast-Path-ViT API up!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a PNG/JPG image -> returns cancer probability + risk level.
    """
    # 1. Read uploaded bytes and open as PIL image
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 2. Preprocess
    x = transform(image).unsqueeze(0).to(device)

    # 3. Run model
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    risk = risk_from_prob(prob)

    return {
        "cancer_probability": float(prob),
        "risk_level": risk,
        "device": str(device),
    }
