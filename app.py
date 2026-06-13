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
        "final_alzheimer_classifier.keras",
        compile=False,
        safe_mode=False
    )

model = load_my_model()

# ===== 4 CLASS NAMES =====

class_names = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
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

    # Resize image
    img = img.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Debug Information
    st.write("Prediction Shape:", prediction.shape)
    st.write("Prediction Values:", prediction)

    predicted_index = np.argmax(prediction)

    # Safety Check
    if predicted_index >= len(class_names):
        st.error(
            f"Model returned class index {predicted_index}, "
            f"but class_names contains only {len(class_names)} classes."
        )
    else:

        predicted_class = class_names[predicted_index]

        confidence = np.max(prediction) * 100

        st.subheader("Prediction Result")

        st.success(f"Prediction: {predicted_class}")

        st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")

        for i, prob in enumerate(prediction[0]):
            if i < len(class_names):
                st.write(
                    f"{class_names[i]} : {prob * 100:.2f}%"
                )
