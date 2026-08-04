"""
Data Preprocessing Module
-------------------------
Responsible for loading raw property and geospatial data, performing data quality checks,
handling any missing or duplicate records, and saving the cleaned dataset.

Input Dataset: data/raw/california_housing.csv
Output Dataset: data/processed/cleaned_housing.csv
"""

import os
import pandas as pd
import numpy as np

def load_data(raw_data_path: str) -> pd.DataFrame:
    """Load raw dataset from disk."""
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")
    print(f"Loading raw dataset from {raw_data_path}...")
    return pd.read_csv(raw_data_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean tabular features and validate geospatial coordinate bounds."""
    print("Performing data quality checks...")

    # Drop duplicates if any exist
    initial_rows = df.shape[0]
    df_cleaned = df.drop_duplicates().copy()
    num_duplicates = initial_rows - df_cleaned.shape[0]
    print(f"Duplicates removed: {num_duplicates}")

    # Drop missing values if any exist
    df_cleaned = df_cleaned.dropna()

    # Validate coordinate bounds for California
    valid_lat = (df_cleaned['latitude'] >= 32.0) & (df_cleaned['latitude'] <= 43.0)
    valid_lon = (df_cleaned['longitude'] >= -125.0) & (df_cleaned['longitude'] <= -113.0)
    df_cleaned = df_cleaned[valid_lat & valid_lon].copy()

    print(f"Data cleaning complete. Cleaned shape: {df_cleaned.shape}")
    return df_cleaned

def main():
    base_dir = r"C:\Users\hkris\OneDrive\Desktop\Construction & Real Estate – Geospatial Valuation via Spatial Embeddings\geospatial-real-estate-valuation"
    raw_path = os.path.join(base_dir, "data", "raw", "california_housing.csv")
    processed_dir = os.path.join(base_dir, "data", "processed")
    output_path = os.path.join(processed_dir, "cleaned_housing.csv")

    os.makedirs(processed_dir, exist_ok=True)

    df_raw = load_data(raw_path)
    df_cleaned = clean_data(df_raw)

    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
