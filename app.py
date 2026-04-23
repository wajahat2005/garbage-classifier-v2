# =========================
# IMPORTS
# =========================
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from huggingface_hub import hf_hub_download

# =========================
# LOAD MODEL FROM HUGGING FACE
# =========================

MODEL_REPO = "2005-wajahat/gaarbage-classifier-v2"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="trash_classifier.keras"   # change if name is different
)

model = tf.keras.models.load_model(model_path)

# =========================
# CLASS NAMES (must match training order)
# =========================
class_names = [
    'cardboard', 'e-waste', 'glass', 'metal',
    'organic', 'paper', 'plastic', 'textile', 'trash'
]

# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(image):
    image = image.resize((300, 300))
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# UI
# =========================
st.title("♻️ Garbage Classifier AI")
st.write("Upload image or use camera to classify waste type")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Or take a picture")

image = None

if camera_image is not None:
    image = Image.open(camera_image)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

# =========================
# PREDICTION
# =========================
if image is not None:
    st.image(image, caption="Input Image")

    img = preprocess_image(image)

    with st.spinner("Predicting..."):
        pred = model.predict(img)

    class_index = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Prediction: {class_names[class_index]}")
    st.info(f"Confidence: {confidence:.2f}")

    st.bar_chart(pred[0])
