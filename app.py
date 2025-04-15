import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
import os
import gdown

st.title("🧠 Multi-Model Deep Learning Inference")

# --- Google Drive model file IDs ---
RESNET_ID = "1-51AUdrOgRadWMeHYK7xLXhIpwDMX5Wd"
VIT_ID = "1zThePeTMXd16fnVLsS0doz9D6RQ4EHfx"

# --- Helper function to download model from Google Drive ---
def download_model(file_id, output_name):
    if not os.path.exists(output_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_name, quiet=False)

# --- Download models if not present ---
download_model(RESNET_ID, "resnet50v2_bam_best.h5")
download_model(VIT_ID, "vit_model_best.pth")

# --- UI components ---
model_choice = st.selectbox("Select a model", ["ResNet50V2 + BAM (Keras)", "ViT (PyTorch)"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write("🧪 Running prediction...")

        if model_choice == "ResNet50V2 + BAM (Keras)":
            model = tf.keras.models.load_model("resnet50v2_bam_best.h5")
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            st.success(f"Prediction: {np.argmax(prediction)}")

        elif model_choice == "ViT (PyTorch)":
            model = torch.load("vit_model_best.pth", map_location=torch.device("cpu"))
            model.eval()
            img = image.resize((224, 224))
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255
            with torch.no_grad():
                output = model(img_tensor)
            st.success(f"Prediction: {output.argmax(1).item()}")
