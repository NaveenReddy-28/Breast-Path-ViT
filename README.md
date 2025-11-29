# Breast-Path-ViT

Explainable ViT-based demo for breast cancer region detection & risk stratification on histopathology patches.

## Tech Stack

- Python 3.11
- PyTorch, timm (ViT backbone)
- FastAPI (REST API for inference)
- Streamlit (web UI)
- OpenSlide (for WSI support – planned)
- CUDA acceleration (NVIDIA RTX 4050)

## How to Run Locally

```bash
conda create -n breast_path_vit python=3.11
conda activate breast_path_vit
pip install -r requirements.txt

# start backend API
uvicorn api.main:app --reload --port 8000

# in another terminal, start UI
streamlit run app/streamlit_app.py
