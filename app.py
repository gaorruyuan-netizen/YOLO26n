import os
import tempfile
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="裂缝检测系统", page_icon="🧱", layout="wide")

st.title("裂缝识别与损伤评估系统")
st.write("上传图像后，系统将自动进行裂缝识别并给出损伤等级。")

@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("未找到模型文件 best.pt，请确认它已上传到仓库根目录。")
        st.stop()
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption="原始图像", use_container_width=True)

    if st.button("开始识别"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        try:
            results = model(temp_path, verbose=False)
            plotted = results[0].plot()
            result_img = Image.fromarray(plotted[..., ::-1] if plotted.shape[-1] == 3 else plotted)

            st.image(result_img, caption="识别结果", use_container_width=True)

            crack_num = 0 if results[0].boxes is None else len(results[0].boxes)

            if crack_num == 0:
                level = "无损伤"
            elif crack_num <= 2:
                level = "轻微损伤"
            elif crack_num <= 5:
                level = "中等损伤"
            else:
                level = "严重损伤"

            st.success(f"检测到裂缝数量：{crack_num}")
            st.info(f"结构损伤等级：{level}")

        except Exception as e:
            st.error(f"识别失败：{e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
