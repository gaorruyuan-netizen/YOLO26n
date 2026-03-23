import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PABC Strength Prediction System",
    page_icon="📊",
    layout="wide"
)

# =========================
# 标题
# =========================
st.title("PABC Compressive Strength Prediction System")
st.markdown(
    "This web application is developed for predicting the compressive strength of PABC materials using machine learning models."
)

# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("lgbm_model.pkl")  # 你的模型文件
    return model

model = load_model()

# =========================
# 侧边栏
# =========================
st.sidebar.header("Input Options")

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Input", "Upload Excel File"]
)

# =========================
# 手动输入
# =========================
if input_method == "Manual Input":
    st.subheader("Manual Input Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        Cement = st.number_input("Cement")
        Sand = st.number_input("Sand")
        Water = st.number_input("Water")

    with col2:
        SA = st.number_input("SA")
        FA = st.number_input("FA")
        CA = st.number_input("CA")

    with col3:
        Superplasticizer = st.number_input("Superplasticizer")
        Fiber = st.number_input("Fiber")
        Temperature = st.number_input("Temperature")

    if st.button("Predict"):
        input_data = np.array([[Cement, Sand, Water, SA, FA, CA, Superplasticizer, Fiber, Temperature]])
        prediction = model.predict(input_data)
        st.success(f"Predicted Compressive Strength: {prediction[0]:.2f} MPa")

# =========================
# 上传 Excel
# =========================
else:
    st.subheader("Upload Excel File for Batch Prediction")

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("Input Data:")
        st.dataframe(data)

        if st.button("Run Prediction"):
            predictions = model.predict(data)
            data["Predicted Strength"] = predictions

            st.write("Prediction Results:")
            st.dataframe(data)

            # 下载按钮
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv",
            )

            # 可视化
            st.subheader("Prediction Visualization")
            fig, ax = plt.subplots()
            ax.plot(predictions)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Predicted Strength (MPa)")
            st.pyplot(fig)

# =========================
# 页脚
# =========================
st.markdown("---")
st.markdown(
    "Developed for academic research purposes. "
    "This system can be used for predicting compressive strength of PABC materials."
)
