import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="PABC裂缝检测系统", layout="wide")

st.title("PABC裂缝识别与损伤评估系统")
st.write("上传混凝土图像，系统将自动识别裂缝并评估损伤程度")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图像", use_column_width=True)

    if st.button("开始识别"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        results = model(temp_path)

        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted)

        st.image(res_image, caption="识别结果", use_column_width=True)

        # 裂缝数量统计
        crack_num = len(results[0].boxes)
        st.write(f"检测到裂缝数量：{crack_num}")

        # 简单损伤等级评估
        if crack_num == 0:
            level = "无损伤"
        elif crack_num <= 2:
            level = "轻微损伤"
        elif crack_num <= 5:
            level = "中等损伤"
        else:
            level = "严重损伤"

        st.write(f"结构损伤等级：{level}")

        os.remove(temp_path)
