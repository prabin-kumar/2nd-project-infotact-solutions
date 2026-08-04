"""
LSTM Deep Learning & Time-Series Audit Module
Deep Learning-Based Flood Prediction Using Rainfall Data

This module provides data-integrity verification for time-series structure,
specifications for required hydrological gauge datasets, LSTM model architecture
templates (PyTorch and Keras), and evaluation audit logging.
"""

import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def verify_timeseries_structure(df: pd.DataFrame):
    """
    Audits DataFrame to check for genuine chronological time-series structure.
    
    Checks:
    - Presence of Date/Time/Timestamp columns
    - Observation sampling frequency
    - Unique entity/station continuous sequence history
    
    Returns:
        dict: Audit report dictionary containing verification status.
    """
    cols = df.columns.tolist()
    time_keywords = ['date', 'time', 'year', 'month', 'day', 'hour', 'timestamp']
    time_cols = [c for c in cols if any(k in c.lower() for k in time_keywords)]

    has_timestamp = len(time_cols) > 0
    total_records = len(df)
    
    audit_report = {
        "dataset_name": "flood_risk_dataset_india.csv",
        "total_records": total_records,
        "total_columns": len(cols),
        "time_columns_found": time_cols,
        "has_timestamp": has_timestamp,
        "spatial_coordinates_present": ('Latitude' in cols and 'Longitude' in cols),
        "is_time_series_valid": has_timestamp,
        "scientific_recommendation": (
            "VALID_TIME_SERIES" if has_timestamp else
            "SPATIAL_TABULAR_DATA_LSTM_INAPPROPRIATE"
        )
    }
    
    if not has_timestamp:
        logging.warning("AUDIT ALERT: Dataset contains NO timestamp/date columns! It consists of spatial observations across coordinates.")
        logging.warning("LSTM models require continuous chronological sequences. Fabricating sequences over spatial data introduces false temporal autocorrelation.")

    return audit_report


def explain_required_timeseries_schema():
    """
    Returns detailed specifications of the dataset schema required to build a valid LSTM model.
    """
    schema_info = {
        "required_columns": [
            "Station_ID (Unique identifier per river/meteorological gauge station)",
            "Timestamp (Hourly or daily ISO datetime index, e.g. 2026-01-01 00:00:00)",
            "Precipitation_mm (Continuous historical rainfall depth)",
            "River_Discharge_m3s (Upstream river streamflow rate)",
            "Water_Level_m (Gauge water stage height)",
            "Soil_Moisture_pct (Continuous soil moisture saturation)",
            "Flood_Occurred (Target flag at time t or t+k)"
        ],
        "required_temporal_frequency": "Continuous fixed interval (e.g. hourly, 6-hourly, daily) without missing time gaps",
        "minimum_sequence_length": "Lookback window N = 7 to 30 past timesteps per station",
        "train_test_split_rule": "Strict chronological split (e.g. Train: 2020-2023, Test: 2024)"
    }
    return schema_info


def build_pytorch_lstm_template(input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
    """
    PyTorch LSTM Architecture Template for sequential time-series forecasting.
    """
    try:
        import torch
        import torch.nn as nn

        class FloodLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super(FloodLSTM, self).__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0
                )
                self.fc = nn.Linear(hidden_dim, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # x shape: (batch_size, sequence_length, input_dim)
                lstm_out, _ = self.lstm(x)
                # Take output of last timestep
                last_out = lstm_out[:, -1, :]
                out = self.fc(last_out)
                return self.sigmoid(out)

        model = FloodLSTM(input_dim, hidden_dim, num_layers, dropout)
        logging.info(f"Instantiated PyTorch FloodLSTM template with input_dim={input_dim}, hidden_dim={hidden_dim}")
        return model
    except ImportError:
        logging.warning("PyTorch not installed. Returning code template representation.")
        return None


def run_lstm_scientific_audit(data_dir: str = "data/raw", results_dir: str = "results"):
    """
    Executes Step 8 scientific audit, evaluates time-series integrity, and saves audit reports.
    """
    os.makedirs(results_dir, exist_ok=True)
    raw_csv_path = os.path.join(data_dir, "flood_risk_dataset_india.csv")

    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset not found at '{raw_csv_path}'.")

    df_raw = pd.read_csv(raw_csv_path)
    audit = verify_timeseries_structure(df_raw)

    # Save lstm_metrics.csv
    audit_metrics_df = pd.DataFrame([{
        "Dataset": audit["dataset_name"],
        "Total Records": audit["total_records"],
        "Timestamp Column Found": audit["has_timestamp"],
        "Time-Series Validity": audit["is_time_series_valid"],
        "LSTM Applied": False,
        "Reason": "Dataset is spatial tabular data without timestamp/chronological structure",
        "Best Alternative Model": "XGBoost (Accuracy: 0.5170, F1: 0.5360) / Logistic Regression (Recall: 0.5668)"
    }])

    metrics_csv_path = os.path.join(results_dir, "lstm_metrics.csv")
    audit_metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved LSTM audit report to '{metrics_csv_path}'.")

    # Load ML comparison if available to generate combined benchmark table
    ml_comp_path = os.path.join(results_dir, "ml_model_comparison.csv")
    if os.path.exists(ml_comp_path):
        ml_df = pd.read_csv(ml_comp_path)
        lstm_row = pd.DataFrame([{
            "Model": "LSTM (Deep Learning)",
            "Accuracy": "N/A (Spatial Data)",
            "Precision": "N/A (Spatial Data)",
            "Recall": "N/A (Spatial Data)",
            "F1-score": "N/A (Spatial Data)"
        }])
        benchmark_df = pd.concat([ml_df, lstm_row], ignore_index=True)
    else:
        benchmark_df = audit_metrics_df

    benchmark_csv_path = os.path.join(results_dir, "ml_vs_lstm_benchmark.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    logging.info(f"Saved comparative benchmark table to '{benchmark_csv_path}'.")

    return audit, benchmark_df
