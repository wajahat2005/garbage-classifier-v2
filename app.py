import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# =========================
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
# PREDICTION
# =========================
if image is not None:
    st.image(image, caption="Input Image")

    img = preprocess_image(image)

    with st.spinner("Analyzing image..."):
        pred = model.predict(img)

    class_index = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Prediction: {class_names[class_index]}")
    st.info(f"Confidence: {confidence:.2f}")

    st.bar_chart(pred[0])
