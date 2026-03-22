import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageOps
from ultralytics import YOLO
from skimage.morphology import skeletonize

# =====================================================
# 页面设置
# =====================================================
st.set_page_config(
    page_title="裂缝检测与损伤评估系统",
    page_icon="🧱",
    layout="wide"
)

# =====================================================
# 样式
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #0b2e59;
}
.result-box {
    background: #eef9f0;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #b7dfc0;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 损伤评估算法
# =====================================================
class MPAlgorithm:
    def __init__(self):
        self.thresholds = {
            'I': {'area_rate': (0, 0.5), 'total_length': (0, 500), 'num_cracks': (0, 10)},
            'II': {'area_rate': (0.5, 1.5), 'total_length': (500, 2000), 'num_cracks': (10, 30)},
            'III': {'area_rate': (1.5, 2.5), 'total_length': (2000, 7000), 'num_cracks': (30, 70)},
            'IV': {'area_rate': (2.5, 5.0), 'total_length': (7000, 10000), 'num_cracks': (70, 110)},
            'V': {'area_rate': (5.0, float('inf')), 'total_length': (10000, float('inf')), 'num_cracks': (110, float('inf'))}
        }

    def evaluate(self, area_rate, total_length, num_cracks):
        def get_level(val, key):
            for lvl in ['I', 'II', 'III', 'IV', 'V']:
                low, high = self.thresholds[lvl][key]
                if low <= val < high:
                    return lvl
            return 'V'

        l1 = get_level(area_rate, 'area_rate')
        l2 = get_level(total_length, 'total_length')
        l3 = get_level(num_cracks, 'num_cracks')

        priority = {'I':1,'II':2,'III':3,'IV':4,'V':5}
        return max([l1,l2,l3], key=lambda x:priority[x])

mp_algo = MPAlgorithm()

# =====================================================
# 加载模型
# =====================================================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# =====================================================
# 图像处理
# =====================================================
def process_image(image_pil):
    image_rgb = np.array(image_pil)
    h, w = image_rgb.shape[:2]

    results = model(image_rgb, verbose=False)

    union_mask = np.zeros((h, w), dtype=np.uint8)
    num_cracks = 0

    if results[0].masks is not None:
        masks_data = results[0].masks.data.cpu().numpy()
        num_cracks = len(masks_data)

        for m in masks_data:
            m_img = Image.fromarray((m * 255).astype(np.uint8))
            m_img = m_img.resize((w, h), Image.NEAREST)
            m_arr = (np.array(m_img) > 127).astype(np.uint8)
            union_mask = np.maximum(union_mask, m_arr)

    skeleton = skeletonize(union_mask > 0).astype(np.uint8)
    total_length = int(np.sum(skeleton))
    area_ratio = float(np.sum(union_mask > 0) / (h * w) * 100)

    level = mp_algo.evaluate(area_ratio, total_length, num_cracks)

    overlay = image_rgb.copy()
    overlay[union_mask > 0] = [0,255,0]

    return overlay, total_length, num_cracks, area_ratio, level

# =====================================================
# 页面
# =====================================================
st.markdown('<div class="main-title">裂缝检测与损伤评估系统</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("上传图像", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    if st.button("开始检测"):
        with st.spinner("检测中..."):
            overlay, length, num, area, level = process_image(image)

        col1, col2 = st.columns(2)
        col1.image(image, caption="原始图像")
        col2.image(overlay, caption="检测结果")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.write(f"**裂缝总长度：** {length}")
        st.write(f"**裂缝数量：** {num}")
        st.write(f"**裂缝面积率：** {area:.3f}%")
        st.write(f"**损伤等级：** {level}")
        st.markdown('</div>', unsafe_allow_html=True)
