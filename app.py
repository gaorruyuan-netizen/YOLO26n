import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="裂缝检测系统", layout="wide")

st.title("混凝土裂缝识别与损伤评估系统")
st.write("上传结构表面图像，系统将自动识别裂缝并评估损伤等级。")

@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("未找到 best.pt 模型文件，请确认已上传到仓库。")
        st.stop()
    model = YOLO(model_path)
    return model

model = load_model()

uploaded_file = st.file_uploader("上传图像", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption="原始图像", use_container_width=True)

    if st.button("开始检测"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        try:
            results = model(temp_path)
            plotted = results[0].plot()
            result_img = Image.fromarray(plotted[:, :, ::-1])

            st.image(result_img, caption="检测结果", use_container_width=True)

            crack_num = 0 if results[0].boxes is None else len(results[0].boxes)

            if crack_num == 0:
                level = "无损伤"
            elif crack_num <= 2:
                level = "轻微损伤"
            elif crack_num <= 5:
                level = "中等损伤"
            else:
                level = "严重损伤"

            st.success(f"裂缝数量：{crack_num}")
            st.warning(f"损伤等级：{level}")

        except Exception as e:
            st.error(f"检测失败：{e}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
