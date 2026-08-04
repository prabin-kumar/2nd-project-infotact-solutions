"""
Inference & Property Price Prediction Pipeline
----------------------------------------------
This module provides automated property price predictions for new property inputs.
It automatically calculates interaction features, grid coordinates, Haversine distance
from dataset center, and applies the saved StandardScaler to construct spatial embeddings
before feeding features into the trained best model artifact.

Target Unit: $100,000s USD
"""

import os
import joblib
import pandas as pd
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Great Circle distance in kilometers."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def predict_property_price(
    median_income: float,
    house_age: float,
    average_rooms: float,
    average_bedrooms: float,
    population: float,
    average_occupancy: float,
    latitude: float,
    longitude: float,
    models_dir: str = None
) -> dict:
    """
    Accepts 8 raw property & location inputs, automatically generates spatial features
    & spatial embeddings, and predicts estimated market property valuation.
    """
    if models_dir is None:
        base_dir = r"C:\Users\hkris\OneDrive\Desktop\Construction & Real Estate – Geospatial Valuation via Spatial Embeddings\geospatial-real-estate-valuation"
        models_dir = os.path.join(base_dir, "models")

    best_model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "spatial_scaler.pkl")
    center_path = os.path.join(models_dir, "spatial_center.pkl")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Train the model first.")
    if not os.path.exists(scaler_path) or not os.path.exists(center_path):
        raise FileNotFoundError("Spatial preprocessing artifacts missing. Run feature_engineering.py first.")

    model = joblib.load(best_model_path)
    scaler = joblib.load(scaler_path)
    center_coords = joblib.load(center_path)

    mean_lat = center_coords["mean_lat"]
    mean_lon = center_coords["mean_lon"]

    # 1. Compute Spatial Interaction Features
    latitude_longitude = latitude * longitude
    latitude_squared = latitude ** 2
    longitude_squared = longitude ** 2

    # 2. Compute Spatial Grid Features
    grid_size = 0.1
    latitude_grid = np.floor(latitude / grid_size) * grid_size
    longitude_grid = np.floor(longitude / grid_size) * grid_size

    # 3. Compute Distance from Center
    distance_from_center = float(haversine_distance(latitude, longitude, mean_lat, mean_lon))

    # 4. Generate Spatial Embeddings using pre-fitted StandardScaler
    spatial_cols = [
        'latitude',
        'longitude',
        'latitude_squared',
        'longitude_squared',
        'latitude_longitude',
        'distance_from_center'
    ]
    
    spatial_df = pd.DataFrame([{
        'latitude': latitude,
        'longitude': longitude,
        'latitude_squared': latitude_squared,
        'longitude_squared': longitude_squared,
        'latitude_longitude': latitude_longitude,
        'distance_from_center': distance_from_center
    }])[spatial_cols]

    embedded_matrix = scaler.transform(spatial_df)[0]

    # 5. Assemble Feature Dictionary matching exact training column order
    feature_dict = {
        "median_income": median_income,
        "house_age": house_age,
        "average_rooms": average_rooms,
        "average_bedrooms": average_bedrooms,
        "population": population,
        "average_occupancy": average_occupancy,
        "latitude": latitude,
        "longitude": longitude,
        "latitude_longitude": latitude_longitude,
        "latitude_squared": latitude_squared,
        "longitude_squared": longitude_squared,
        "latitude_grid": latitude_grid,
        "longitude_grid": longitude_grid,
        "distance_from_center": distance_from_center,
        "spatial_emb_1": embedded_matrix[0],
        "spatial_emb_2": embedded_matrix[1],
        "spatial_emb_3": embedded_matrix[2],
        "spatial_emb_4": embedded_matrix[3],
        "spatial_emb_5": embedded_matrix[4],
        "spatial_emb_6": embedded_matrix[5]
    }

    input_df = pd.DataFrame([feature_dict])

    # 6. Predict Property Valuation
    raw_prediction = float(model.predict(input_df)[0])
    estimated_usd = raw_prediction * 100000.0

    return {
        "predicted_house_value_raw": round(raw_prediction, 4),
        "estimated_property_price_usd": round(estimated_usd, 2),
        "distance_from_center_km": round(distance_from_center, 2)
    }

def main():
    print("================================================================")
    print("  Construction & Real Estate Valuation - Property Predictor    ")
    print("================================================================\n")

    sample_inputs = {
        "median_income": 8.3252,       # $83,252 median household income
        "house_age": 41.0,             # 41 years old
        "average_rooms": 6.98,         # 6.98 average rooms
        "average_bedrooms": 1.02,      # 1.02 average bedrooms
        "population": 322.0,           # 322 residents in block
        "average_occupancy": 2.55,     # 2.55 members per household
        "latitude": 37.88,             # San Francisco / Berkeley latitude
        "longitude": -122.23           # San Francisco / Berkeley longitude
    }

    print("--- SAMPLE INPUT PROPERTY CHARACTERISTICS ---")
    for key, val in sample_inputs.items():
        print(f"  {key:<20}: {val}")

    result = predict_property_price(**sample_inputs)

    print("\n--- PREDICTION OUTPUT ---")
    print(f"  Raw Target Value ($100k units) : {result['predicted_house_value_raw']}")
    print(f"  Estimated Market Value (USD)   : ${result['estimated_property_price_usd']:,.2f}")
    print(f"  Distance from Spatial Center   : {result['distance_from_center_km']} km")
    print("================================================================\n")

if __name__ == "__main__":
    main()
