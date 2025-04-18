import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
import os
import gdown

st.title(" GastroIntestinal Disease Classification")

# --- Google Drive model file IDs ---
RESNET_KERAS_ID = "1G3xRNdW7LK7lAtln80HwJjaUk_Cx-CII"
VIT_ID = "1BXiIpLdShnGuOmJGAm4TdH_Xt4UWBtui"  # Updated ID for ViT model (state_dict)

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

# --- Download models ---
download_model(RESNET_KERAS_ID, "resnet50v2_bam_best.keras")
download_model(VIT_ID, "vit_model_best2.pth")

# --- UI ---
model_choice = st.selectbox("Select a model", ["ResNet50V2 + BAM", "VisionTransformer"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

class_names = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write(" Running prediction...")

        if model_choice == "ResNet50V2 + BAM":
            model = tf.keras.models.load_model(
                "resnet50v2_bam_best.keras",
                custom_objects={"bam_block": bam_block}
            )
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            st.success(f" Prediction: **{predicted_class}**")

        elif model_choice == "VisionTransformer":
            from transformers import ViTForImageClassification, AutoFeatureExtractor

            # Define model architecture
            class MyViT(torch.nn.Module):
                def __init__(self, num_classes=4):
                    super(MyViT, self).__init__()
                    self.vit = ViTForImageClassification.from_pretrained(
                        'google/vit-base-patch16-224-in21k',
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )

                def forward(self, x):
                    return self.vit(x).logits

            # Load model
            model = MyViT(num_classes=4)
            model.load_state_dict(torch.load("vit_model_best2.pth", map_location=torch.device("cpu")))
            model.eval()

            # Preprocess image
            feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            img = image.resize((224, 224))
            img_np = np.array(img).astype(np.float32) / 255.0
            img_np = (img_np - feature_extractor.image_mean) / feature_extractor.image_std
            img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)

            # Predict
            with torch.no_grad():
                output = model(img_tensor)
            predicted_class = class_names[output.argmax(1).item()]
            st.success(f"📌 Prediction: **{predicted_class}**")
