import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# ===== FIX FOR quantization_config ERROR =====

original_from_config = Dense.from_config

@classmethod
def fixed_from_config(cls, config):
    config.pop("quantization_config", None)
    return original_from_config(config)

Dense.from_config = fixed_from_config

# ===== LOAD MODEL WITH CACHE =====

@st.cache_resource
def load_my_model():
    return load_model(
        "mobilenetv2_model.keras",
        compile=False,
        safe_mode=False
    )

model = load_my_model()

# ===== CLASS NAMES =====

class_names = [
    "ModerateDemented",
    "MildDemented",
    "NonDemented"
]

# ===== TITLE =====

st.title("Alzheimer MRI Detection")

# ===== FILE UPLOAD =====

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ===== PREDICTION =====

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    img = img.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")

    st.success(f"Prediction: {predicted_class}")

    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
# py -3.10 -m streamlit run app.py
#cd "E:\MAJOR PROJECT\Alzheimer_Web_App"

#
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

#
# .\.venv\Scripts\Activate.ps1

#
# py -3.10 -m streamlit run app.py
