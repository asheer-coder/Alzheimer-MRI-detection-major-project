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

# ===== LOAD MODEL =====

model = load_model(
    "mobilenetv2_model.keras",
    compile=False,
    safe_mode=False
)

# ===== CLASS NAMES =====
# SAME ORDER AS TRAINING

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

    # open image
    img = Image.open(uploaded_file).convert("RGB")

    # show image
    st.image(
        img,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # resize
    img = img.resize((224, 224))

    # convert to array
    img_array = np.array(img)

    # normalize
    img_array = img_array / 255.0

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)

    # highest probability
    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # ===== RESULT =====

    st.subheader("Prediction Result")

    st.success(f"Prediction: {predicted_class}")

    st.write(f"Confidence: {confidence:.2f}%")

    # ===== ALL PROBABILITIES =====

    st.subheader("Class Probabilities")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]} : {prob*100:.2f}%")
        
# py -3.10 -m streamlit run app.py