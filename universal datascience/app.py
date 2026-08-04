"""
Streamlit Web Application: Deep Learning-Based Flood Prediction
Deep Learning-Based Flood Prediction Using Rainfall Data

An academic demonstration interface for evaluating real-time flood prediction
using trained Machine Learning baseline models (XGBoost / Logistic Regression).
"""

import sys
import os
import streamlit as st
import pandas as pd

# Add project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_flood, VALID_LAND_COVERS, VALID_SOIL_TYPES

# Page configuration
st.set_page_config(
    page_title="Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.3rem; color: #1E3A8A; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #4B5563; text-align: center; margin-bottom: 25px; }
    .disclaimer-box { background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 12px; border-radius: 4px; margin-bottom: 20px; font-size: 0.95rem; }
    .flood-banner { background-color: #FEE2E2; border: 2px solid #EF4444; color: #991B1B; padding: 20px; border-radius: 8px; text-align: center; font-size: 1.8rem; font-weight: bold; margin-top: 15px; }
    .noflood-banner { background-color: #D1FAE5; border: 2px solid #10B981; color: #065F46; padding: 20px; border-radius: 8px; text-align: center; font-size: 1.8rem; font-weight: bold; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌊 Deep Learning-Based Flood Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hydro-Meteorological & Terrain Feature Inference Model</div>', unsafe_allow_html=True)

# Safety Disclaimer
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Academic Safety Notice</strong>: This is an academic machine learning prediction system developed for research demonstration and should not be used as an official emergency warning system.
</div>
""", unsafe_allow_html=True)

# Sidebar Options
st.sidebar.header("⚙️ Model Settings")
selected_model = st.sidebar.selectbox(
    "Select Model Classifier",
    options=["XGBoost", "Logistic Regression", "Random Forest", "Decision Tree"],
    index=0,
    help="XGBoost is recommended for highest overall accuracy (51.70%). Logistic Regression is recommended for highest flood recall (56.68%)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("About the Project")
st.sidebar.info("""
- **Dataset**: 10,000 spatial observations across India
- **Features**: 30 engineered hydro-meteorological indicators
- **Primary Model**: XGBoost / Logistic Regression Baseline
""")

# Input Form
st.subheader("📊 Hydro-Meteorological & Spatial Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🌧️ Meteorological Metrics")
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

with col2:
    st.markdown("##### 🌊 Hydrological Metrics")
    river_discharge = st.number_input("River Discharge (m³/s)", min_value=0.0, max_value=20000.0, value=3500.0, step=50.0)
    water_level = st.number_input("Water Level (m)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
    elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=9000.0, value=120.0, step=5.0)

with col3:
    st.markdown("##### 🏔️ Physical & Demographic Metrics")
    land_cover = st.selectbox("Land Cover Type", options=VALID_LAND_COVERS, index=1)
    soil_type = st.selectbox("Soil Type", options=VALID_SOIL_TYPES, index=0)
    population_density = st.number_input("Population Density (people/km²)", min_value=0.0, max_value=50000.0, value=3000.0, step=100.0)

st.markdown("---")
col_sub1, col_sub2, col_sub3, col_sub4 = st.columns(4)

with col_sub1:
    infrastructure = st.radio("Infrastructure Presence", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
with col_sub2:
    historical_floods = st.radio("Historical Floods", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
with col_sub3:
    latitude = st.number_input("Latitude (°N)", min_value=6.0, max_value=38.0, value=20.5, step=0.1)
with col_sub4:
    longitude = st.number_input("Longitude (°E)", min_value=68.0, max_value=98.0, value=78.9, step=0.1)

st.markdown("---")

# Predict Button
if st.button("🚀 Predict Flood Risk", use_container_width=True, type="primary"):
    raw_input = {
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Humidity_pct": humidity,
        "River_Discharge_m3s": river_discharge,
        "Water_Level_m": water_level,
        "Elevation_m": elevation,
        "Land_Cover": land_cover,
        "Soil_Type": soil_type,
        "Population_Density": population_density,
        "Infrastructure": infrastructure,
        "Historical_Floods": historical_floods,
        "Latitude": latitude,
        "Longitude": longitude
    }

    with st.spinner("Processing hydro-meteorological features and computing risk..."):
        try:
            res = predict_flood(raw_input, model_name=selected_model.lower().replace(" ", "_"))

            st.markdown("### 📈 Prediction Results")

            if res["prediction"] == "FLOOD":
                st.markdown(f'<div class="flood-banner">🚨 PREDICTION: FLOOD RISK DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="noflood-banner">✅ PREDICTION: NO FLOOD RISK</div>', unsafe_allow_html=True)

            if res["flood_probability"] is not None:
                prob = res["flood_probability"]
                st.markdown(f"#### Calculated Flood Probability: **{prob:.2f}%**")
                st.progress(float(prob / 100.0))

            # Input Summary Table
            st.markdown("#### 📋 Input Feature Summary")
            summary_df = pd.DataFrame(list(raw_input.items()), columns=["Feature Parameter", "Input Value"])
            st.dataframe(summary_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
