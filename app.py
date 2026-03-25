import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2

st.title("🍅 Tomato Leaf Disease Detection")

# Load model
model = YOLO("fbest.pt")

# 📸 Camera / Upload
img_file = st.camera_input("Take a picture")

if img_file is not None:
    bytes_data = img_file.getvalue()

    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    results = model(frame)
    annotated = results[0].plot()

    st.image(annotated, channels="BGR")

    # Show prediction
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"Detected: {model.names[cls]} ({conf:.2f})")