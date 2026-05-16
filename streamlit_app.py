# streamlit_app.py

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('runs/detect/train/weights/best.pt')

model = load_model()

st.title("PAN / Aadhar Card Detector")
st.write("Upload an image to detect whether it is a PAN or Aadhar card.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(np.array(image), caption='Uploaded Image', width=400)
    st.write("Classifying...")

    # Prediction
    results = model.predict(source=np.array(image), save=False)
    result = results[0]  # YOLOv8 returns a list, we take the first image

    if len(result.boxes) > 0:
        # Get class names and confidence
        pred_classes = [result.names[int(cls)] for cls in result.boxes.cls]
        confidences = [float(conf) for conf in result.boxes.conf]

        for cls, conf in zip(pred_classes, confidences):
            st.write(f"Prediction: **{cls}** with confidence {conf:.2f}")
        
        # Show image with bounding boxes
        results_img = result.plot()  # Returns image with boxes
        st.image(results_img, caption='Prediction', width=400)
    else:
        st.write("No object detected")