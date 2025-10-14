import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from src.utils import generate_gradcam
import tempfile
import matplotlib.pyplot as plt
from PIL import Image

# 🩻 Load Model

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("runs/pneumonia_densenet121.h5")
    return model

model = load_model()

# 🖼️ Streamlit UI Setup
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩻", layout="centered")

st.title(" Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image and let the CNN model predict if pneumonia is present.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    img_path = temp_file.name

    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess
    img_array = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Analyzing X-ray..."):
        prob = model.predict(img_array)[0][0]
        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    # Display Results
    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: `{prob:.4f}`")

    if label == "PNEUMONIA":
        st.error("⚠️ Pneumonia detected! Please consult a radiologist.")
    else:
        st.success("✅ No pneumonia detected.")

    
