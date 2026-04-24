import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests

# =========================
# LOAD MODEL FROM HUGGING FACE
# =========================
@st.cache_resource
def load_model():
    MODEL_URL = "https://huggingface.co/2005-wajahat/gaarbage-classifier-v2/resolve/main/model.keras"
    # get_file automatically downloads and caches the file so it only happens once
    model_file = tf.keras.utils.get_file("garbage_model.keras", MODEL_URL)
    return tf.keras.models.load_model(model_file)

model = load_model()

# =========================
# CLASS NAMES
# =========================
class_names = [
    'cardboard', 'e-waste', 'glass', 'metal',
    'organic', 'paper', 'plastic', 'textile', 'trash'
]

IMG_SIZE = (380, 380)

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# UI
# =========================
st.title("♻️ Garbage Classifier")
st.write("Upload an image OR use camera to classify waste.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Or take a picture")

image = None

if camera_image is not None:
    image = Image.open(camera_image)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

# =========================
# PREDICTION & CONFIDENCE LOGIC
# =========================
# =========================
# PREDICTION (IMPROVED)
# =========================
if image is not None:
    st.image(image, caption="Input Image")

    img = preprocess_image(image)

    with st.spinner("Analyzing image..."):
        pred = model.predict(img)[0]

    # Top predictions
    top_indices = pred.argsort()[-3:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_scores = [pred[i] for i in top_indices]

    best_class = top_classes[0]
    confidence = top_scores[0]

    # Threshold system
    CONFIDENCE_THRESHOLD = 0.65
    GAP_THRESHOLD = 0.15  # difference between top1 and top2

    if confidence < CONFIDENCE_THRESHOLD or (top_scores[0] - top_scores[1] < GAP_THRESHOLD):
        st.error("❌ Unknown / Not Garbage")
        st.write("This item does not clearly belong to known garbage categories.")
    else:
        st.success(f"♻️ Prediction: **{best_class.upper()}**")
        st.info(f"Confidence: **{confidence*100:.1f}%**")

    # Show top 3 predictions
    st.subheader("Top Predictions")
    for cls, score in zip(top_classes, top_scores):
        st.write(f"{cls}: {score*100:.2f}%")

    # Chart
    chart_data = pd.DataFrame(
        pred,
        index=class_names,
        columns=["Probability"]
    )
    st.bar_chart(chart_data)

    # Better Bar Chart (Shows Names instead of 0, 1, 2...)
    chart_data = pd.DataFrame(
        pred[0],
        index=class_names,
        columns=["Probability"]
    )
    st.bar_chart(chart_data)
