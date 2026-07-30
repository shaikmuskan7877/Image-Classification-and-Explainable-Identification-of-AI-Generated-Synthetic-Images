import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Class names
class_names = ["Real Image", "AI-Generated Image"]

# Page title
st.set_page_config(page_title="AI Image Classifier")
st.title("Image Classification and Explainable Identification of AI-Generated Synthetic Images")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    if prediction[0][0] >= 0.5:
        label = class_names[1]
        confidence = prediction[0][0] * 100
    else:
        label = class_names[0]
        confidence = (1 - prediction[0][0]) * 100

    st.subheader("Prediction")
    st.success(label)

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}%")
