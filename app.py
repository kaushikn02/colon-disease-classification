import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
import os
import gdown

st.title("🧠 Multi-Model Deep Learning Inference")

# --- Google Drive model file IDs ---
RESNET_KERAS_ID = "1G3xRNdW7LK7lAtln80HwJjaUk_Cx-CII"  # New Keras model ID
VIT_ID = "1zThePeTMXd16fnVLsS0doz9D6RQ4EHfx"

# --- BAM block definition ---
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Multiply, Reshape,
    Conv2D, BatchNormalization, Activation, Add
)

def bam_block(input_feature, ratio=8, dilation_rate=4):
    channel = input_feature.shape[-1]

    shared_dense = Dense(channel // ratio, activation='relu', use_bias=False)
    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)

    avg_pool = shared_dense(avg_pool)
    max_pool = shared_dense(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Dense(channel, activation='sigmoid', use_bias=False)(channel_attention)
    channel_attention = Multiply()([input_feature, Reshape((1, 1, channel))(channel_attention)])

    x = Conv2D(channel // ratio, kernel_size=1, padding="same")(channel_attention)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(1, kernel_size=3, padding="same", dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    spatial_attention = Multiply()([channel_attention, x])
    return spatial_attention

# --- Helper function to download from Google Drive ---
def download_model(file_id, output_name):
    if not os.path.exists(output_name):
        gdown.download(id=file_id, output=output_name, quiet=False)

# --- Download models if not already present ---
download_model(RESNET_KERAS_ID, "resnet50v2_bam_best.keras")
download_model(VIT_ID, "vit_model_best.pth")

# --- UI Components ---
model_choice = st.selectbox("Select a model", ["ResNet50V2 + BAM (Keras)", "ViT (PyTorch)"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

class_names = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write("🧪 Running prediction...")

        if model_choice == "ResNet50V2 + BAM (Keras)":
            model = tf.keras.models.load_model(
                "resnet50v2_bam_best.keras",
                custom_objects={"bam_block": bam_block}
            )
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            st.success(f"📌 Prediction: **{predicted_class}**")

        elif model_choice == "ViT (PyTorch)":
            model = torch.load("vit_model_best.pth", map_location=torch.device("cpu"))
            model.eval()
            img = image.resize((224, 224))
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255
            with torch.no_grad():
                output = model(img_tensor)
            predicted_class = class_names[output.argmax(1).item()]
            st.success(f"📌 Prediction: **{predicted_class}**")
