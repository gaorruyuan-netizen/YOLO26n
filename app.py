import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from PIL import Image
import tempfile

st.set_page_config(page_title="Crack Detection System", layout="wide")

st.title("Crack Detection and Damage Assessment System")
st.write("Upload an image to detect cracks and evaluate damage level.")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# MP damage evaluation
def evaluate_damage(area_rate, total_length, num_cracks):
    if area_rate < 0.5:
        return "I"
    elif area_rate < 1.5:
        return "II"
    elif area_rate < 2.5:
        return "III"
    elif area_rate < 5:
        return "IV"
    else:
        return "V"

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    results = model(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    num_cracks = 0

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        num_cracks = len(masks)
        for m in masks:
            mask = np.maximum(mask, (m > 0.5).astype(np.uint8))

    skeleton = skeletonize(mask > 0)
    total_length = np.sum(skeleton)
    area_rate = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100

    level = evaluate_damage(area_rate, total_length, num_cracks)

    col1, col2 = st.columns(2)

    with col1:
        st.image(mask, caption="Crack Mask")

    with col2:
        st.write("### Damage Assessment Results")
        st.write(f"Crack Length: {total_length}")
        st.write(f"Number of Cracks: {num_cracks}")
        st.write(f"Crack Area Ratio: {area_rate:.2f}%")
        st.write(f"Damage Level: {level}")
