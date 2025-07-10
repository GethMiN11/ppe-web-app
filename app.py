
import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="PPE Detection App", layout="centered")
st.title("🦺 PPE Detection System")
st.markdown("Upload an image or use your webcam to detect PPE (Helmet, Vest, Mask, etc.)")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='ppe-model.pt', force_reload=True)

def detect_ppe(image):
    results = model(image)
    labels = results.xyxyn[0][:, -1].numpy()
    names = results.names
    label_names = [names[int(i)] for i in labels]
    return "Allowed to Enter" if any(label in label_names for label in ['helmet', 'vest', 'gloves', 'mask']) else "Not Allowed to Enter", results

option = st.radio("Choose input method:", ['Upload Image', 'Use Webcam'])

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        status, results = detect_ppe(image)
        st.success(f"Result: {status}")
        results.render()
elif option == 'Use Webcam':
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        status, results = detect_ppe(img_rgb)
        results.render()
        st.write(f"Result: {status}")
        FRAME_WINDOW.image(img_rgb)
    cap.release()
