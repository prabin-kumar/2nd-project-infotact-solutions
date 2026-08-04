"""
Streamlit Web Application: Geospatial Real Estate Valuation
------------------------------------------------------------
Interactive web UI for predicting residential property prices using physical property attributes
and automatically generated spatial features and continuous spatial embeddings.

Run Command: streamlit run app.py
"""

import os
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

# Set Streamlit Page Config
st.set_page_config(
    page_title="Geospatial Real Estate Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Great Circle distance in kilometers."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

@st.cache_resource
def load_artifacts():
    """Load model, spatial scaler, center coordinates, and model metadata."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "spatial_scaler.pkl")
    center_path = os.path.join(models_dir, "spatial_center.pkl")
    name_path = os.path.join(models_dir, "best_model_name.txt")
    eval_path = os.path.join(reports_dir, "best_model_evaluation.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: '{model_path}'. Train the model first.")
    if not os.path.exists(scaler_path) or not os.path.exists(center_path):
        raise FileNotFoundError("Preprocessing artifacts missing in 'models/'. Run feature engineering first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    center = joblib.load(center_path)

    model_name = "Best Model"
    if os.path.exists(name_path):
        with open(name_path, "r") as f:
            model_name = f.read().strip()

    eval_text = "No evaluation metrics found."
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_text = f.read().strip()

    return model, scaler, center, model_name, eval_text

def predict_house_value(inputs, model, scaler, center):
    """Generate spatial features, embeddings, and predict property value."""
    lat = inputs["latitude"]
    lon = inputs["longitude"]
    mean_lat = center["mean_lat"]
    mean_lon = center["mean_lon"]

    # 1. Spatial Interactions
    lat_lon = lat * lon
    lat_sq = lat ** 2
    lon_sq = lon ** 2

    # 2. Spatial Grid
    grid_size = 0.1
    lat_grid = np.floor(lat / grid_size) * grid_size
    lon_grid = np.floor(lon / grid_size) * grid_size

    # 3. Distance from Center
    dist_km = float(haversine_distance(lat, lon, mean_lat, mean_lon))

    # 4. Standardized Spatial Embeddings
    spatial_cols = ['latitude', 'longitude', 'latitude_squared', 'longitude_squared', 'latitude_longitude', 'distance_from_center']
    spatial_df = pd.DataFrame([{
        'latitude': lat,
        'longitude': lon,
        'latitude_squared': lat_sq,
        'longitude_squared': lon_sq,
        'latitude_longitude': lat_lon,
        'distance_from_center': dist_km
    }])[spatial_cols]

    embedded = scaler.transform(spatial_df)[0]

    # 5. Assemble exact training feature vector
    feature_dict = {
        "median_income": inputs["median_income"],
        "house_age": inputs["house_age"],
        "average_rooms": inputs["average_rooms"],
        "average_bedrooms": inputs["average_bedrooms"],
        "population": inputs["population"],
        "average_occupancy": inputs["average_occupancy"],
        "latitude": lat,
        "longitude": lon,
        "latitude_longitude": lat_lon,
        "latitude_squared": lat_sq,
        "longitude_squared": lon_sq,
        "latitude_grid": lat_grid,
        "longitude_grid": lon_grid,
        "distance_from_center": dist_km,
        "spatial_emb_1": embedded[0],
        "spatial_emb_2": embedded[1],
        "spatial_emb_3": embedded[2],
        "spatial_emb_4": embedded[3],
        "spatial_emb_5": embedded[4],
        "spatial_emb_6": embedded[5]
    }

    input_df = pd.DataFrame([feature_dict])
    raw_val = float(model.predict(input_df)[0])
    usd_val = raw_val * 100000.0

    return raw_val, usd_val, dist_km

def main():
    # Header Banner
    st.title("🏠 Geospatial Real Estate Valuation")
    st.caption("### AI-Based Property Price Prediction Using Spatial Features")
    st.markdown("---")

    # Load artifacts gracefully
    try:
        model, scaler, center, model_name, eval_text = load_artifacts()
    except Exception as e:
        st.error(f"❌ Error loading system artifacts: {e}")
        st.stop()

    # Sidebar: About Project & Model Info
    st.sidebar.title("ℹ️ About the Project")
    st.sidebar.info(
        "This project uses machine learning and geospatial features to estimate "
        "residential property values. Latitude and longitude are transformed into spatial features "
        "and embeddings so that the model can learn relationships between location and property value."
    )

    st.sidebar.markdown("### 📊 Project Metadata")
    st.sidebar.markdown("- **Dataset**: California Housing Dataset")
    st.sidebar.markdown("- **Task**: Regression")
    st.sidebar.markdown("- **Target**: House Value ($100k units)")
    st.sidebar.markdown("- **ML Approach**: Spatial Feature Engineering + Machine Learning")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Information")
    st.sidebar.success(f"**Selected Model**: {model_name}")

    with st.sidebar.expander("📋 View Model Evaluation Metrics"):
        st.code(eval_text)

    # Main Body: Input Controls
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("📋 Property Characteristics")

        median_income = st.slider(
            "Median Household Income ($10,000s)",
            min_value=0.5, max_value=15.0, value=3.87, step=0.1,
            help="Median income for households in the block group."
        )

        house_age = st.slider(
            "House Age (Years)",
            min_value=1.0, max_value=52.0, value=28.0, step=1.0,
            help="Median age of houses in the block group."
        )

        average_rooms = st.number_input(
            "Average Rooms per Household",
            min_value=1.0, max_value=20.0, value=5.4, step=0.1
        )

        average_bedrooms = st.number_input(
            "Average Bedrooms per Household",
            min_value=0.5, max_value=10.0, value=1.1, step=0.1
        )

        population = st.number_input(
            "Block Group Population",
            min_value=3.0, max_value=10000.0, value=1425.0, step=25.0
        )

        average_occupancy = st.number_input(
            "Average Household Occupancy",
            min_value=0.5, max_value=20.0, value=3.0, step=0.1
        )

    with col2:
        st.subheader("📍 Location & Geospatial Coordinates")

        latitude = st.number_input(
            "Latitude (°N)",
            min_value=32.5, max_value=42.0, value=37.88, step=0.01,
            format="%.4f"
        )

        longitude = st.number_input(
            "Longitude (°W)",
            min_value=-124.5, max_value=-114.0, value=-122.23, step=0.01,
            format="%.4f"
        )

        # Interactive Location Map
        st.markdown("**Property Map Location**")
        map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_data, zoom=10)

    st.markdown("---")

    # Prediction Action
    if st.button("🔮 Predict Property Value", use_container_width=True, type="primary"):
        input_data = {
            "median_income": median_income,
            "house_age": house_age,
            "average_rooms": average_rooms,
            "average_bedrooms": average_bedrooms,
            "population": population,
            "average_occupancy": average_occupancy,
            "latitude": latitude,
            "longitude": longitude
        }

        try:
            raw_pred, usd_pred, dist_km = predict_house_value(input_data, model, scaler, center)

            st.markdown("### 💰 Prediction Results")
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric(
                    label="Estimated Property Value",
                    value=f"${usd_pred:,.2f}"
                )

            with res_col2:
                st.metric(
                    label="Raw Model Target Value",
                    value=f"{raw_pred:.4f} ($100k)"
                )

            with res_col3:
                st.metric(
                    label="Distance to Spatial Center",
                    value=f"{dist_km:.2f} km"
                )

            st.caption(
                "ℹ️ **Note**: Prediction is an ML-based estimate and should not be considered "
                "a professional property appraisal."
            )

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")

if __name__ == "__main__":
    main()
