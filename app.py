import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
import numpy as np

st.title(" Multi-Model Deep Learning Inference")

model_choice = st.selectbox("Select a model", ["ResNet50V2 + BAM (Keras)", "ViT (PyTorch)"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write(" Running prediction...")

        if model_choice == "ResNet50V2 + BAM (Keras)":
            model = tf.keras.models.load_model("resnet50v2_bam_best.h5")
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            st.success(f"Prediction: {np.argmax(prediction)}")

        elif model_choice == "ViT (PyTorch)":
            model = torch.load("vit_model_best.pth", map_location=torch.device("cpu"))
            model.eval()
            transform = tf.keras.applications.resnet50.preprocess_input
            img = image.resize((224, 224))
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255
            with torch.no_grad():
                output = model(img_tensor)
            st.success(f"Prediction: {output.argmax(1).item()}")
