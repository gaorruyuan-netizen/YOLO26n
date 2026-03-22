import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
from skimage.morphology import skeletonize

# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="裂缝检测与损伤评估系统",
    page_icon="🧱",
    layout="wide"
)

# =========================================================
# 页面样式
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        color: #1f2937;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0b2e59;
        margin-bottom: 0.35rem;
    }

    .sub-text {
        color: #4b5563;
        font-size: 1.05rem;
        line-height: 1.7;
        margin-bottom: 1.2rem;
    }

    .white-card {
        background: #ffffff;
        border: 1px solid #dbe3ee;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }

    .result-card {
        background: linear-gradient(135deg, #eef9f0 0%, #f7fcf8 100%);
        border: 1px solid #b7dfc0;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .metric-title {
        color: #166534 !important;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #14532d !important;
        font-size: 1.8rem;
        font-weight: 800;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0b2e59 0%, #174a84 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        height: 3em;
        font-size: 16px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #174a84 0%, #215c9d 100%) !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    .stDataFrame {
        border: 1px solid #d9e2ec;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 损伤评估算法
# =========================================================
class MPAlgorithm:
    def __init__(self):
        self.thresholds = {
            'I': {'area_rate': (0, 0.5), 'total_length': (0, 500), 'num_cracks': (0, 10)},
            'II': {'area_rate': (0.5, 1.5), 'total_length': (500, 2000), 'num_cracks': (10, 30)},
            'III': {'area_rate': (1.5, 2.5), 'total_length': (2000, 7000), 'num_cracks': (30, 70)},
            'IV': {'area_rate': (2.5, 5.0), 'total_length': (7000, 10000), 'num_cracks': (70, 110)},
            'V': {'area_rate': (5.0, float('inf')), 'total_length': (10000, float('inf')),
                  'num_cracks': (110, float('inf'))}
        }
        self.descriptions = {
            'I': "Ⅰ级（无损伤）：结构完整性良好。",
            'II': "Ⅱ级（轻微损伤）：存在少量细微裂缝，建议持续监测。",
            'III': "Ⅲ级（中等损伤）：裂缝发展明显，可能影响耐久性。",
            'IV': "Ⅳ级（严重损伤）：裂缝较多且扩展显著，存在较高风险。",
            'V': "Ⅴ级（极严重损伤）：结构状态较差，建议立即处理。"
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

        priority = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
        final_lvl = max([l1, l2, l3], key=lambda x: priority[x])

        return final_lvl, self.descriptions[final_lvl]


# =========================================================
# 模型加载
# =========================================================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    return YOLO(model_path)


def get_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"模型加载失败：{e}")
        st.info("请检查 best.pt 是否已上传到仓库根目录，以及 requirements.txt 中依赖版本是否正确。")
        st.stop()


# =========================================================
# 工具函数
# =========================================================
def resize_mask(mask_2d: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8))
    mask_img = mask_img.resize((target_w, target_h), Image.Resampling.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)


def create_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[binary_mask > 0] = [0, 255, 0]
    return overlay


def normalize_uploaded_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def process_image(image_pil: Image.Image, model, mp_algo: MPAlgorithm):
    start_time = time.time()

    image_rgb = np.array(image_pil)
    h, w = image_rgb.shape[:2]

    results = model(image_rgb, verbose=False)

    union_mask = np.zeros((h, w), dtype=np.uint8)
    num_cracks = 0

    if results and len(results) > 0 and results[0].masks is not None:
        masks_data = results[0].masks.data.cpu().numpy()
        num_cracks = len(masks_data)

        for m in masks_data:
            m_resized = resize_mask(m, w, h)
            union_mask = np.maximum(union_mask, m_resized)

    skeleton = skeletonize(union_mask > 0).astype(np.uint8)
    total_length = int(np.sum(skeleton))
    area_ratio = float(np.sum(union_mask > 0) / (h * w) * 100)

    level, description = mp_algo.evaluate(area_ratio, total_length, num_cracks)
    overlay = create_overlay(image_rgb, union_mask)
    duration_ms = (time.time() - start_time) * 1000

    return {
        "image_rgb": image_rgb,
        "overlay_rgb": overlay,
        "mask": union_mask,
        "metrics": {
            "total_length": total_length,
            "num_cracks": num_cracks,
            "area_ratio": area_ratio,
            "level": level,
            "description": description,
            "time_ms": duration_ms
        }
    }


# =========================================================
# 页面标题
# =========================================================
st.markdown('<div class="main-title">裂缝检测与损伤评估系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">上传一张或多张图像，系统将自动完成裂缝识别、参数统计与损伤等级评估。</div>',
    unsafe_allow_html=True
)

# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.header("模型信息")
    st.write("**检测模型：** YOLO")
    st.write("**评估方法：** MP 损伤评估算法")
    st.write("**输入类型：** 图像文件")

    st.header("输出指标")
    st.write("**裂缝总长度**")
    st.write("**裂缝数量**")
    st.write("**裂缝面积率**")
    st.write("**损伤等级**")
    st.write("**推理耗时**")

# =========================================================
# 加载模型
# =========================================================
model = get_model()
mp_algo = MPAlgorithm()

# =========================================================
# 上传区域
# =========================================================
st.markdown('<div class="white-card">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "上传图像（支持多选）",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True
)
run_button = st.button("开始检测")
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 运行检测
# =========================================================
if run_button:
    if not uploaded_files:
        st.warning("请先上传至少一张图像。")
    else:
        results_list = []
        failed_files = []

        progress_bar = st.progress(0)

        for idx, file in enumerate(uploaded_files):
            try:
                image_pil = normalize_uploaded_image(file)
                result = process_image(image_pil, model, mp_algo)
                result["name"] = file.name
                results_list.append(result)
            except Exception as e:
                failed_files.append((file.name, str(e)))

            progress_bar.progress((idx + 1) / len(uploaded_files))

        if failed_files:
            for file_name, err in failed_files:
                st.warning(f"{file_name} 处理失败：{err}")

        if not results_list:
            st.error("所有图像均处理失败，请检查模型文件或输入图像格式。")
            st.stop()

        st.success(f"检测完成，共成功处理 {len(results_list)} 张图像。")

        # 汇总表
        summary_rows = []
        for res in results_list:
            m = res["metrics"]
            summary_rows.append({
                "图像名称": res["name"],
                "裂缝总长度（像素）": m["total_length"],
                "裂缝数量": m["num_cracks"],
                "面积率（%）": round(m["area_ratio"], 3),
                "损伤等级": m["level"],
                "推理耗时（ms）": round(m["time_ms"], 2)
            })

        st.subheader("批量检测结果汇总")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        # 单图详情
        st.subheader("单张图像检测详情")
        selected_name = st.selectbox(
            "选择图像查看详细结果",
            [res["name"] for res in results_list]
        )

        selected_result = next(res for res in results_list if res["name"] == selected_name)
        m = selected_result["metrics"]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="metric-title">损伤等级</div>
                <div class="metric-value">{m["level"]}级</div>
                <div style="margin-top:8px;color:#166534;font-weight:600;">{m["description"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("裂缝总长度", f'{m["total_length"]} 像素')
        c2.metric("裂缝数量", f'{m["num_cracks"]}')
        c3.metric("面积率", f'{m["area_ratio"]:.3f} %')
        c4.metric("推理耗时", f'{m["time_ms"]:.1f} ms')

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(
                selected_result["image_rgb"],
                caption=f"原始图像：{selected_result['name']}",
                use_container_width=True
            )

        with img_col2:
            st.image(
                selected_result["overlay_rgb"],
                caption="检测结果（裂缝叠加图）",
                use_container_width=True
            )
