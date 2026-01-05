import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Weather Temperature Prediction",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather Temperature Prediction")
st.write("Predict Daily Temperature using ML")

# ======================================
# LOAD MODEL & SCALER
# ======================================
@st.cache_resource
def load_model():
    with open("weather_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("weather-scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# ======================================
# INPUT MODE
# ======================================
st.sidebar.header("⚙️ Input Method")
input_mode = st.sidebar.radio(
    "Choose Input Type",
    ["Manual Input"]
)

# ======================================
# MANUAL INPUT
# ======================================
if input_mode == "Manual Input":
    st.subheader("📥 Enter Weather Details")

    col1, col2 = st.columns(2)

    with col1:
        hours_sunlight = st.number_input(
            "☀️ Hours of Sunlight",
            0.0, 24.0, step=0.1
        )

    with col2:
        humidity_level = st.number_input(
            "💧 Humidity Level (%)",
            0, 100, step=1
        )

    input_df = pd.DataFrame({
        "hours_sunlight": [hours_sunlight],
    
    })

# ======================================
# CSV UPLOAD
# ======================================
else:
       st.stop()

# ======================================
# PREVIEW
# ======================================
st.write("### 🔍 Input Preview")
st.dataframe(input_df)

# ======================================
# PREDICTION
# ======================================
if st.button("🌡️ Predict Temperature"):
    try:
        scaled_data = scaler.transform(input_df)
        predictions = model.predict(scaled_data)

        input_df["Predicted_Temperature (°C)"] = np.round(predictions, 2)

        st.success("✅ Prediction Successful")
        st.dataframe(input_df)

        avg_temp = predictions.mean()

        if avg_temp >= 30:
            st.error("🔥 Very Hot Weather")
        elif avg_temp >= 22:
            st.warning("🌤️ Warm Weather")
        else:
            st.info("❄️ Cool Weather")

    except Exception as e:
        st.error("❌ Prediction Error")
        st.exception(e)

# ======================================
# FOOTER
# ======================================
st.markdown("---")
st.caption("ML Weather Prediction App | Streamlit")
