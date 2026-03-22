import os
import traceback
import streamlit as st

st.set_page_config(page_title="环境诊断", page_icon="🛠️", layout="wide")

st.title("Streamlit 部署诊断页面")
st.write("这个页面用于定位应用启动失败的具体原因。")

st.subheader("1. 基础文件检查")
files_to_check = ["app.py", "requirements.txt", "best.pt"]
for f in files_to_check:
    st.write(f"{f}: {'存在' if os.path.exists(f) else '不存在'}")

st.subheader("2. Python 环境信息")
import sys
st.code(sys.version)

st.subheader("3. 依赖导入测试")

def try_import(name, import_func):
    try:
        module = import_func()
        st.success(f"{name} 导入成功")
        return module
    except Exception as e:
        st.error(f"{name} 导入失败：{e}")
        st.code(traceback.format_exc())
        return None

np = try_import("numpy", lambda: __import__("numpy"))
pd = try_import("pandas", lambda: __import__("pandas"))
pil = try_import("PIL", lambda: __import__("PIL"))
skimage = try_import("skimage.morphology", lambda: __import__("skimage.morphology", fromlist=["skeletonize"]))
torch = try_import("torch", lambda: __import__("torch"))
ultralytics = try_import("ultralytics", lambda: __import__("ultralytics", fromlist=["YOLO"]))

st.subheader("4. 模型加载测试")
if ultralytics is not None:
    try:
        YOLO = getattr(ultralytics, "YOLO")
        model = YOLO("best.pt")
        st.success("best.pt 加载成功")
        st.write("模型对象类型：", type(model))
    except Exception as e:
        st.error(f"best.pt 加载失败：{e}")
        st.code(traceback.format_exc())
else:
    st.warning("由于 ultralytics 未成功导入，跳过模型加载测试。")

st.subheader("5. 结论")
st.info("把本页面显示的第一个报错发给我，我就能精确告诉你该改哪一项。")
