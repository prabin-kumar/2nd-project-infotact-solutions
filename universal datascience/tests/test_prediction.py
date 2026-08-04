"""
Unit Tests for Flood Prediction Inference System
Deep Learning-Based Flood Prediction Using Rainfall Data

Tests model loading, preprocessor loading, input validation, negative range handling,
feature engineering consistency, and prediction output format.
"""

import os
import sys
import pytest

# Ensure src path is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, load_preprocessor, validate_input, predict_flood


def test_load_model_valid():
    """Verify that trained model artifacts load without error."""
    xgb_model = load_model("xgboost", models_dir="models")
    assert xgb_model is not None, "XGBoost model failed to load."

    lr_model = load_model("logistic_regression", models_dir="models")
    assert lr_model is not None, "Logistic Regression model failed to load."


def test_load_model_invalid_name():
    """Verify that requesting an invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model name"):
        load_model("invalid_model_name", models_dir="models")


def test_load_preprocessor():
    """Verify that scaler artifact loads cleanly."""
    scaler = load_preprocessor("models/scaler.pkl")
    assert scaler is not None, "Scaler artifact failed to load."


def test_validate_input_valid():
    """Verify that valid input dictionary is processed cleanly."""
    valid_dict = {
        "Rainfall_mm": 120.0,
        "Temperature_C": 28.5,
        "Humidity_pct": 80.0,
        "River_Discharge_m3s": 3000.0,
        "Water_Level_m": 6.5,
        "Elevation_m": 150.0,
        "Land_Cover": "Urban",
        "Soil_Type": "Clay"
    }

    clean_dict = validate_input(valid_dict)
    assert clean_dict["Rainfall_mm"] == 120.0
    assert clean_dict["Humidity_pct"] == 80.0
    assert "Latitude" in clean_dict, "Default latitude should be added."


def test_validate_input_negative_rainfall():
    """Verify that negative rainfall value raises ValueError."""
    invalid_dict = {
        "Rainfall_mm": -50.0,
        "Temperature_C": 25.0,
        "Humidity_pct": 70.0,
        "River_Discharge_m3s": 1000.0,
        "Water_Level_m": 5.0,
        "Elevation_m": 100.0,
        "Land_Cover": "Forest",
        "Soil_Type": "Sandy"
    }

    with pytest.raises(ValueError, match="cannot be negative"):
        validate_input(invalid_dict)


def test_validate_input_invalid_humidity():
    """Verify that humidity > 100 raises ValueError."""
    invalid_dict = {
        "Rainfall_mm": 50.0,
        "Temperature_C": 25.0,
        "Humidity_pct": 150.0,  # Invalid >100%
        "River_Discharge_m3s": 1000.0,
        "Water_Level_m": 5.0,
        "Elevation_m": 100.0,
        "Land_Cover": "Forest",
        "Soil_Type": "Sandy"
    }

    with pytest.raises(ValueError, match="Humidity"):
        validate_input(invalid_dict)


def test_validate_input_non_numeric():
    """Verify that non-numeric strings for numeric fields raise ValueError."""
    invalid_dict = {
        "Rainfall_mm": "invalid_string_abc",
        "Temperature_C": 25.0,
        "Humidity_pct": 70.0,
        "River_Discharge_m3s": 1000.0,
        "Water_Level_m": 5.0,
        "Elevation_m": 100.0,
        "Land_Cover": "Forest",
        "Soil_Type": "Sandy"
    }

    with pytest.raises(ValueError, match="must be a valid numeric number"):
        validate_input(invalid_dict)


def test_predict_flood_output_format():
    """Verify end-to-end prediction pipeline and result keys."""
    sample_input = {
        "Rainfall_mm": 200.0,
        "Temperature_C": 30.0,
        "Humidity_pct": 85.0,
        "River_Discharge_m3s": 4000.0,
        "Water_Level_m": 8.0,
        "Elevation_m": 50.0,
        "Land_Cover": "Urban",
        "Soil_Type": "Clay",
        "Population_Density": 4000.0,
        "Infrastructure": 1,
        "Historical_Floods": 1
    }

    result = predict_flood(sample_input, model_name="xgboost")

    assert "prediction" in result
    assert result["prediction"] in ["FLOOD", "NO FLOOD"]
    assert "flood_probability" in result
    assert 0.0 <= result["flood_probability"] <= 100.0
    assert result["prediction_code"] in [0, 1]
    assert result["model_used"] == "xgboost"
