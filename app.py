import streamlit as st
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from PIL import Image

st.set_page_config(
    page_title="裂缝检测与损伤评估系统",
    layout="wide"
)

st.title("裂缝检测与损伤评估系统")
st.write("上传图像后，系统将自动识别裂缝并评估损伤等级。")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def evaluate_damage(area_rate, total_length, num_cracks):
    if area_rate < 0.5:
        return "Ⅰ级（轻微损伤）"
    elif area_rate < 1.5:
        return "Ⅱ级（中等损伤）"
    elif area_rate < 2.5:
        return "Ⅲ级（较重损伤）"
    elif area_rate < 5:
        return "Ⅳ级（严重损伤）"
    else:
        return "Ⅴ级（极严重损伤）"

uploaded_file = st.file_uploader("上传裂缝图像", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    st.subheader("原始图像")
    st.image(image, width="stretch")

    with st.spinner("正在进行裂缝检测，请稍候..."):
        results = model(image, verbose=False)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    num_cracks = 0

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        num_cracks = len(masks)

        for m in masks:
            if m.shape != mask.shape:
                import cv2
                m = cv2.resize(m, (mask.shape[1], mask.shape[0]))
            mask = np.maximum(mask, (m > 0.5).astype(np.uint8))

    skeleton = skeletonize(mask > 0)
    total_length = int(np.sum(skeleton))
    area_rate = float(np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100) if mask.size else 0.0
    level = evaluate_damage(area_rate, total_length, num_cracks)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("裂缝识别结果")
        st.image(mask * 255, width="stretch", clamp=True)

    with col2:
        st.subheader("损伤评估结果")
        st.write(f"**裂缝总长度：** {total_length}")
        st.write(f"**裂缝数量：** {num_cracks}")
        st.write(f"**裂缝面积率：** {area_rate:.2f}%")
        st.write(f"**损伤等级：** {level}")
