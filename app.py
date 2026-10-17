import streamlit as st
import pickle
import numpy as np
import pandas as pd
st.set_page_config(
    page_title="Dự đoán chỉ số AQI",
    page_icon="🎋",
    layout="centered"
)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
    if hasattr(model, "feature_names_in_"):
        features_list = list(model.feature_names_in_)
    else:
        features_list = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    st.title("Dự đoán chỉ số ô nhiễm môi trường đô thị(AQI)")
    st.markdown("Nhập các thông số nồng độ chất ô nhiễm để bắt đầu dự đoán.")
    st.write("---")
    st.subheader("Nhập thông số môi trường")
    col1, col2 = st.columns(2)
    user_inputs = {}

    for i, feature in enumerate(features_list):
        with col1 if i % 2 == 0 else col2:
            user_inputs[feature] = st.number_input(
                label=f"Nồng độ {feature}", 
                min_value=0.0, 
                value=0.0, 
                step=0.1,
                format="%.1f"
            )

    st.write("---")
    if st.button("Bắt đầu dự đoán", type="primary", use_container_width=True):
        input_data = np.array([[user_inputs[f] for f in features_list]])
        prediction = model.predict(input_data)[0]
        st.subheader("📌 Kết quả dự đoán:")
        
        if prediction <= 50:
            st.success(f"Chỉ số AQI dự kiến: {prediction:.2f} (Tốt - Không khí trong lành)")
        elif prediction <= 100:
            st.info(f"Chỉ số AQI dự kiến: {prediction:.2f} (Vừa phải - Chấp nhận được)")
        elif prediction <= 150:
            st.warning(f"Chỉ số AQI dự kiến: {prediction:.2f} (Kém - Nhạy cảm cần lưu ý)")
        else:
            st.error(f"Chỉ số AQI dự kiến: {prediction:.2f} (Nguy hiểm - Ô nhiễm nghiêm trọng)")

except FileNotFoundError:
    st.error("Không tìm thấy file `model.pkl`.")
except Exception as e:
    st.error(f"Error: {e}")
