import streamlit as st
import requests
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.title("Breast-Path-ViT — Full App (UI → API)")

uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    st.write("Sending to API...")

    response = requests.post(
        API_URL,
        files={"file": ("image.png", img_bytes, "image/png")}
    )

    if response.status_code == 200:
        result = response.json()
        st.success("Prediction received!")

        st.write(f"### Cancer Probability: **{result['cancer_probability']:.3f}**")
        st.write(f"### Risk Level: **{result['risk_level']}**")

        if result["device"] == "cuda":
            st.success("Running on GPU (CUDA) ✔")
        else:
            st.warning("Running on CPU")
    else:
        st.error(f"API Error: {response.status_code}")
