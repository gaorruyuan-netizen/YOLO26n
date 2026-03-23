import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize
from PIL import Image

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="裂缝检测与损伤评估系统",
    layout="wide"
)

# =========================
# 自定义样式
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}
h1, h2, h3 {
    color: #111111;
}
.result-card {
    background-color: #f8f9fb;
    border: 1px solid #e6e9ef;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 8px;
}
.metric-title {
    font-size: 15px;
    color: #666;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #111;
}
.level-box {
    padding: 12px 14px;
    border-radius: 10px;
    font-weight: 700;
    text-align: center;
}
.level-low {
    background-color: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #c8e6c9;
}
.level-mid {
    background-color: #fff8e1;
    color: #ef6c00;
    border: 1px solid #ffe082;
}
.level-high {
    background-color: #ffebee;
    color: #c62828;
    border: 1px solid #ffcdd2;
}
.small-note {
    color: #666;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 标题区
# =========================
st.title("裂缝检测与损伤评估系统")
st.markdown("### 上传混凝土裂缝图像，系统将自动识别裂缝并评估损伤等级")
st.info("本系统基于 YOLO 裂缝分割模型与骨架提取方法，可自动识别裂缝区域，并输出裂缝总长度、裂缝数量、裂缝面积率及损伤等级。")

# =========================
# 参数
# =========================
MAX_SIDE = 960
MASK_THRESHOLD = 0.5

# =========================
# 模型加载
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# 等级评估
# =========================
def evaluate_damage(area_rate, total_length, num_cracks):
    if area_rate < 0.5:
        return "Ⅰ级（轻微损伤）"
    elif area_rate < 1.5:
        return "Ⅱ级（中等损伤）"
    elif area_rate < 2.5:
        return "Ⅲ级（较重损伤）"
    elif area_rate < 5.0:
        return "Ⅳ级（严重损伤）"
    else:
        return "Ⅴ级（极严重损伤）"

def get_level_class(level_text):
    if "Ⅰ级" in level_text or "Ⅱ级" in level_text:
        return "level-low"
    elif "Ⅲ级" in level_text:
        return "level-mid"
    return "level-high"

# =========================
# 工具函数
# =========================
def resize_for_inference(image, max_side=960):
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    if scale < 1.0:
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        image_resized = image.copy()
    return image_resized, scale

def build_overlay(image_rgb, binary_mask):
    overlay = image_rgb.copy()
    color_mask = np.zeros_like(image_rgb)
    color_mask[:, :, 0] = 255
    alpha = 0.35
    m = binary_mask > 0
    overlay[m] = ((1 - alpha) * overlay[m] + alpha * color_mask[m]).astype(np.uint8)
    return overlay

def detect_cracks(image_rgb):
    # 只做输入缩放，不限制 imgsz
    image_infer, scale = resize_for_inference(image_rgb, max_side=MAX_SIDE)

    # 不限制 imgsz，保持默认推理尺寸
    results = model(image_infer, verbose=False)

    h, w = image_infer.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    num_cracks = 0

    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        num_cracks = len(masks)

        for m in masks:
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (m_resized > MASK_THRESHOLD).astype(np.uint8))

    # 保留骨架细化
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    total_length = int(np.sum(skeleton))

    area_rate = float(np.sum(mask > 0) / (h * w) * 100.0)
    level = evaluate_damage(area_rate, total_length, num_cracks)
    overlay = build_overlay(image_infer, mask)

    return {
        "image": image_infer,
        "mask": mask,
        "overlay": overlay,
        "num_cracks": num_cracks,
        "total_length": total_length,
        "area_rate": area_rate,
        "level": level
    }

# =========================
# 上传区
# =========================
st.subheader("上传裂缝图像")
st.caption("支持 JPG、PNG、JPEG 格式图像。上传后系统将自动执行裂缝检测与损伤评估。")

uploaded_file = st.file_uploader(
    label="请选择裂缝图像文件",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

# =========================
# 主逻辑
# =========================
if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image = np.array(image_pil)

        with st.spinner("正在进行裂缝检测，请稍候..."):
            result = detect_cracks(image)

        st.markdown("---")
        st.subheader("检测结果展示")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**原始图像**")
            st.image(result["image"], width=460)

        with col2:
            st.markdown("**裂缝识别结果**")
            st.image(result["overlay"], width=460)
            st.caption("红色区域表示识别出的裂缝区域。")

        st.markdown("---")
        st.subheader("损伤评估结果")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-title">裂缝总长度</div>
                <div class="metric-value">{result["total_length"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-title">裂缝数量</div>
                <div class="metric-value">{result["num_cracks"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-title">裂缝面积率 (%)</div>
                <div class="metric-value">{result["area_rate"]:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            level_class = get_level_class(result["level"])
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-title">损伤等级</div>
                <div class="level-box {level_class}">{result["level"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("查看二值裂缝图"):
            st.image((result["mask"] * 255).astype(np.uint8), width=520)
            st.caption("白色区域为提取出的裂缝二值掩膜。")

        st.markdown(
            '<div class="small-note">注：裂缝总长度基于骨架提取结果统计，裂缝面积率为裂缝像素面积占整幅图像面积的百分比。</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"检测失败：{e}")
        st.warning("请检查上传图像是否有效，或稍后重试。")
else:
    st.markdown("---")
    st.markdown("#### 使用说明")
    st.markdown("""
1. 点击上方上传区域，选择一张混凝土裂缝图像；  
2. 系统将自动完成裂缝识别与损伤评估；  
3. 页面将展示原始图像、裂缝识别结果以及裂缝形态指标与损伤等级。  
""")
