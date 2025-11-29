# scripts/test_vit_model.py

import os
import sys

# ------------------------------------------------
# 1. Add the project root (Breast-Path-ViT) to sys.path
# ------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------
# 2. Import libraries and model files
# ------------------------------------------------
import torch
from torchvision import transforms
from PIL import Image

from models.vit_backbone import ViTForBreastCancer
from models.risk_head import risk_from_prob

# ------------------------------------------------
# 3. Load the image
# ------------------------------------------------
image_path = os.path.join(PROJECT_ROOT, "sample.png")   # USING sample.png
image = Image.open(image_path).convert("RGB")

# ------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

input_tensor = transform(image).unsqueeze(0)

# ------------------------------------------------
# 5. Run the model
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ViTForBreastCancer().to(device)
model.eval()

with torch.no_grad():
    outputs = model(input_tensor.to(device))

# Softmax over two classes (benign=0, malignant=1)
prob = torch.softmax(outputs, dim=1)[0][1].item()
print("\nCancer Probability:", prob)

# ------------------------------------------------
# 6. Risk Stratification
# ------------------------------------------------
risk = risk_from_prob(prob)
print("Risk Level:", risk)
print("\nTest completed successfully.")
