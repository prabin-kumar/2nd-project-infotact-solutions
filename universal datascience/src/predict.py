"""
Flood Prediction Inference Module
Deep Learning-Based Flood Prediction Using Rainfall Data

This module provides reusable functions for loading trained models and scalers,
validating user inputs, engineering runtime features, and generating flood risk
predictions with probabilities.
"""

import os
import sys
import json
import joblib
import logging
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import create_all_engineered_features, normalize_column_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VALID_LAND_COVERS = ["Water Body", "Urban", "Agriculture", "Grassland", "Forest", "Agricultural", "Desert"]
VALID_SOIL_TYPES = ["Clay", "Peat", "Silty", "Silt", "Loam", "Alluvial", "Sandy"]


def load_model(model_name: str = "xgboost", models_dir: str = "models"):
    """
    Loads trained classifier artifact from models directory.
    """
    filename_map = {
        "xgboost": "xgboost.pkl",
        "logistic_regression": "logistic_regression.pkl",
        "random_forest": "random_forest.pkl",
        "decision_tree": "decision_tree.pkl"
    }

    key = model_name.lower().replace(" ", "_")
    if key not in filename_map:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(filename_map.keys())}")

    filepath = os.path.join(models_dir, filename_map[key])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found. Please train models in Step 7.")

    model = joblib.load(filepath)
    logging.info(f"Successfully loaded trained model '{model_name}' from '{filepath}'")
    return model


def load_preprocessor(scaler_path: str = "models/scaler.pkl"):
    """Loads fitted StandardScaler artifact."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found.")
    scaler = joblib.load(scaler_path)
    logging.info(f"Successfully loaded scaler from '{scaler_path}'")
    return scaler


def validate_input(input_dict: dict) -> dict:
    """
    Validates user input dictionary for numerical types, non-negative ranges, and valid categories.
    """
    if not isinstance(input_dict, dict):
        raise ValueError("Input must be a valid key-value dictionary.")

    key_aliases = {
        'Rainfall (mm)': 'Rainfall_mm', 'Rainfall': 'Rainfall_mm',
        'Temperature (°C)': 'Temperature_C', 'Temperature (C)': 'Temperature_C', 'Temperature': 'Temperature_C',
        'Humidity (%)': 'Humidity_pct', 'Humidity': 'Humidity_pct',
        'River Discharge (m³/s)': 'River_Discharge_m3s', 'River Discharge (m3/s)': 'River_Discharge_m3s', 'River Discharge': 'River_Discharge_m3s',
        'Water Level (m)': 'Water_Level_m', 'Water Level': 'Water_Level_m',
        'Elevation (m)': 'Elevation_m', 'Elevation': 'Elevation_m',
        'Land Cover': 'Land_Cover',
        'Soil Type': 'Soil_Type',
        'Population Density': 'Population_Density',
        'Historical Floods': 'Historical_Floods'
    }

    clean_input = {}
    for k, v in input_dict.items():
        standard_k = key_aliases.get(k.strip(), k.strip())
        clean_input[standard_k] = v

    clean_input.setdefault('Latitude', 20.5937)
    clean_input.setdefault('Longitude', 78.9629)
    clean_input.setdefault('Population_Density', 1000.0)
    clean_input.setdefault('Infrastructure', 1)
    clean_input.setdefault('Historical_Floods', 0)

    numeric_fields = [
        'Rainfall_mm', 'Temperature_C', 'Humidity_pct', 'River_Discharge_m3s',
        'Water_Level_m', 'Elevation_m', 'Population_Density', 'Latitude', 'Longitude'
    ]

    for field in numeric_fields:
        if field in clean_input:
            try:
                val = float(clean_input[field])
            except (ValueError, TypeError):
                raise ValueError(f"Field '{field}' must be a valid numeric number. Got '{clean_input[field]}'")

            if field in ['Rainfall_mm', 'Humidity_pct', 'River_Discharge_m3s', 'Water_Level_m', 'Elevation_m', 'Population_Density']:
                if val < 0:
                    raise ValueError(f"Field '{field}' cannot be negative. Got {val}")
            clean_input[field] = val

    if 'Humidity_pct' in clean_input and not (0 <= clean_input['Humidity_pct'] <= 100):
        raise ValueError(f"Humidity (%) must be between 0 and 100. Got {clean_input['Humidity_pct']}")

    for bin_field in ['Infrastructure', 'Historical_Floods']:
        if bin_field in clean_input:
            try:
                val = int(clean_input[bin_field])
            except (ValueError, TypeError):
                raise ValueError(f"Field '{bin_field}' must be binary 0 or 1. Got '{clean_input[bin_field]}'")
            if val not in [0, 1]:
                raise ValueError(f"Field '{bin_field}' must be binary 0 or 1. Got {val}")
            clean_input[bin_field] = val

    return clean_input


def predict_flood(input_dict: dict, model_name: str = "xgboost", data_dir: str = "data/processed", models_dir: str = "models"):
    """
    Main prediction pipeline: validates input, engineers runtime features, scales inputs,
    and returns flood prediction label and probability.
    """
    validated_input = validate_input(input_dict)
    
    df_single = pd.DataFrame([validated_input])
    df_eng = create_all_engineered_features(df_single)

    meta_path = os.path.join(data_dir, "feature_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            expected_features = json.load(f)["feature_names"]
    else:
        expected_features = pd.read_csv(os.path.join(data_dir, "X_train.csv"), nrows=1).columns.tolist()

    df_aligned = df_eng.reindex(columns=expected_features, fill_value=0)

    scaler = load_preprocessor(os.path.join(models_dir, "scaler.pkl"))
    model = load_model(model_name, models_dir)

    X_scaled = scaler.transform(df_aligned)
    X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)

    pred_val = int(model.predict(X_scaled_df)[0])
    pred_label = "FLOOD" if pred_val == 1 else "NO FLOOD"

    flood_prob = None
    if hasattr(model, "predict_proba"):
        prob_vals = model.predict_proba(X_scaled_df)[0]
        flood_prob = round(float(prob_vals[1]) * 100.0, 2)

    result = {
        "model_used": model_name,
        "prediction": pred_label,
        "prediction_code": pred_val,
        "flood_probability": flood_prob,
        "input_summary": validated_input
    }

    return result


def format_cli_prediction(result: dict) -> str:
    """Formats prediction result dict into clean CLI text block."""
    inp = result["input_summary"]
    prob_str = f"{result['flood_probability']:.2f}%" if result["flood_probability"] is not None else "N/A"

    output = f"""
==================================================
        FLOOD PREDICTION SYSTEM OUTPUT
==================================================
Model Used          : {result['model_used'].upper()}

--- Input Features ---
Rainfall            : {inp.get('Rainfall_mm', 'N/A')} mm
Temperature         : {inp.get('Temperature_C', 'N/A')} °C
Humidity            : {inp.get('Humidity_pct', 'N/A')} %
River Discharge     : {inp.get('River_Discharge_m3s', 'N/A')} m³/s
Water Level         : {inp.get('Water_Level_m', 'N/A')} m
Elevation           : {inp.get('Elevation_m', 'N/A')} m
Land Cover          : {inp.get('Land_Cover', 'N/A')}
Soil Type           : {inp.get('Soil_Type', 'N/A')}

--------------------------------------------------
Prediction          : {result['prediction']}
Flood Probability   : {prob_str}
==================================================
"""
    return output


if __name__ == "__main__":
    sample_input = {
        "Rainfall_mm": 218.9,
        "Temperature_C": 34.1,
        "Humidity_pct": 84.5,
        "River_Discharge_m3s": 4236.0,
        "Water_Level_m": 8.5,
        "Elevation_m": 45.0,
        "Land_Cover": "Urban",
        "Soil_Type": "Clay",
        "Population_Density": 5000.0,
        "Infrastructure": 1,
        "Historical_Floods": 1
    }

    try:
        res = predict_flood(sample_input, model_name="xgboost")
        print(format_cli_prediction(res))
    except Exception as e:
        print(f"Error executing prediction: {e}")
