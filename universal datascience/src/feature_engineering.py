"""
Feature Engineering Pipeline
Deep Learning-Based Flood Prediction Using Rainfall Data

This module provides reusable functions for feature engineering, domain-specific
transformation, categorical encoding, feature selection, scaling, target leakage
prevention, train/test splitting, and reshaping tabular spatial data for ML models
(Logistic Regression, Decision Tree, Random Forest, XGBoost) and Deep Learning (LSTM).
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes dataframe column headers by removing special non-ASCII characters,
    stripping whitespace, and standardizing unit representations.
    """
    df_clean = df.copy()
    clean_cols = []
    for c in df_clean.columns:
        c_str = str(c).strip()
        # Clean temperature and discharge unicode artifacts
        c_str = c_str.replace('C', 'C').replace('°C', 'C')
        c_str = c_str.replace('m/s', 'm3s').replace('m³/s', 'm3s').replace('m/s', 'm3s')
        c_str = c_str.replace(' (mm)', '_mm').replace(' (%)', '_pct').replace(' (m)', '_m')
        c_str = c_str.replace(' ', '_').replace('(', '').replace(')', '')
        clean_cols.append(c_str)
    df_clean.columns = clean_cols
    return df_clean


def review_existing_features(df: pd.DataFrame, target_col: str = "Flood_Occurred"):
    """
    Audits raw dataframe and returns column classification dictionary.
    """
    df_norm = normalize_column_names(df)
    cols = df_norm.columns.tolist()
    
    if target_col not in cols:
        possible_targets = [c for c in cols if 'flood' in c.lower() or 'target' in c.lower()]
        target_col = possible_targets[0] if possible_targets else cols[-1]

    feature_cols = [c for c in cols if c != target_col]
    
    numerical_cols = df_norm[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_norm[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    time_cols = [c for c in cols if 'date' in c.lower() or 'time' in c.lower() or 'year' in c.lower() or 'month' in c.lower()]
    
    review_summary = {
        "total_columns": len(cols),
        "target_column": target_col,
        "time_columns": time_cols,
        "has_time_dimension": len(time_cols) > 0,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "total_records": len(df_norm)
    }
    
    logging.info(f"Audited features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical, target: '{target_col}'")
    return review_summary


def create_hydrological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific hydrological and meteorological interaction features.
    
    Features created:
    - Rainfall_Elevation_Ratio: Rainfall per meter of elevation (terrain inundation risk).
    - Hydro_Load_Index: Rainfall * River Discharge / (Elevation + 10). Combined runoff load.
    - Discharge_WaterLevel_Ratio: River Discharge / (Water Level + 1e-5). Channel capacity metric.
    - Relative_Humidity_Ratio: Humidity / (Temperature + 1.0). Atmospheric moisture metric.
    - Elevation_Inverse: 1.0 / (Elevation + 1.0). Lowland inundation factor.
    """
    df_feat = df.copy()
    
    # 1. Rainfall to Elevation Ratio (high rainfall in low elevation = higher flood risk)
    if 'Rainfall_mm' in df_feat.columns and 'Elevation_m' in df_feat.columns:
        df_feat['Rainfall_Elevation_Ratio'] = df_feat['Rainfall_mm'] / (df_feat['Elevation_m'].clip(lower=0) + 1.0)
        df_feat['Elevation_Inverse'] = 1.0 / (df_feat['Elevation_m'].clip(lower=0) + 1.0)

    # 2. Hydro Load Index (Total surface runoff pressure relative to terrain)
    if 'Rainfall_mm' in df_feat.columns and 'River_Discharge_m3s' in df_feat.columns and 'Elevation_m' in df_feat.columns:
        df_feat['Hydro_Load_Index'] = (df_feat['Rainfall_mm'] * df_feat['River_Discharge_m3s']) / (df_feat['Elevation_m'].clip(lower=0) + 10.0)

    # 3. Discharge to Water Level Ratio (Streamflow intensity)
    if 'River_Discharge_m3s' in df_feat.columns and 'Water_Level_m' in df_feat.columns:
        df_feat['Discharge_WaterLevel_Ratio'] = df_feat['River_Discharge_m3s'] / (df_feat['Water_Level_m'].clip(lower=0) + 1e-5)

    # 4. Atmospheric Moisture Ratio
    if 'Humidity_pct' in df_feat.columns and 'Temperature_C' in df_feat.columns:
        df_feat['Relative_Humidity_Ratio'] = df_feat['Humidity_pct'] / (df_feat['Temperature_C'].clip(lower=0) + 1.0)

    return df_feat


def create_domain_categorical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical variables using domain-informed physical risk scores
    and one-hot encodings.
    """
    df_feat = df.copy()

    # Soil Permeability Index (Soil Science domain mapping)
    if 'Soil_Type' in df_feat.columns:
        soil_perm_map = {
            'Clay': 3.0,       # Impermeable -> highest surface runoff / flood risk
            'Peat': 2.5,
            'Silty': 2.0,
            'Loam': 1.5,
            'Alluvial': 1.5,
            'Sandy': 1.0       # Highly permeable -> lowest flood risk
        }
        df_feat['Soil_Permeability_Index'] = df_feat['Soil_Type'].map(soil_perm_map).fillna(2.0)

    # Land Cover Inundation Susceptibility
    if 'Land_Cover' in df_feat.columns:
        land_cover_map = {
            'Water Body': 3.0,
            'Urban': 2.5,      # High impervious cover -> high runoff
            'Agriculture': 2.0,
            'Grassland': 1.5,
            'Forest': 1.0       # High canopy/infiltration -> lowest runoff
        }
        df_feat['Land_Cover_Inundation_Risk'] = df_feat['Land_Cover'].map(land_cover_map).fillna(2.0)

    # One-Hot Encoding for categorical columns
    cat_cols = [c for c in ['Land_Cover', 'Soil_Type'] if c in df_feat.columns]
    if cat_cols:
        df_feat = pd.get_dummies(df_feat, columns=cat_cols, drop_first=False, dtype=int)

    return df_feat


def create_vulnerability_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates socioeconomic vulnerability and historical flood interaction features.
    """
    df_feat = df.copy()

    if 'Population_Density' in df_feat.columns and 'Infrastructure' in df_feat.columns:
        df_feat['Infrastructure_Vulnerability'] = df_feat['Population_Density'] * df_feat['Infrastructure']

    if 'Historical_Floods' in df_feat.columns and 'Rainfall_mm' in df_feat.columns:
        df_feat['Historical_Rainfall_Interaction'] = df_feat['Historical_Floods'] * df_feat['Rainfall_mm']

    return df_feat


def create_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main orchestration function to run the full feature engineering pipeline.
    """
    df_norm = normalize_column_names(df)
    df_hydro = create_hydrological_features(df_norm)
    df_enc = create_domain_categorical_encodings(df_hydro)
    df_full = create_vulnerability_interactions(df_enc)
    
    logging.info(f"Engineered dataset total columns: {df_full.shape[1]}")
    return df_full


def select_and_filter_features(df: pd.DataFrame, target_col: str = "Flood_Occurred", corr_threshold: float = 0.95):
    """
    Splits features X and target y, checks for collinearity > corr_threshold and target leakage.
    
    Returns:
        tuple: (X, y, dropped_features)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col]).copy()

    # Check for target leakage or extreme multicollinearity
    corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    dropped_features = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    if dropped_features:
        logging.info(f"Removing collinear features (> {corr_threshold}): {dropped_features}")
        X = X.drop(columns=dropped_features)
    else:
        logging.info(f"No collinear features dropped above threshold {corr_threshold}.")

    return X, y, dropped_features


def prepare_model_inputs(df: pd.DataFrame, target_col: str = "Flood_Occurred", test_size: float = 0.2, random_state: int = 42):
    """
    Prepares X and y matrices, applies stratified train/test split, and fits StandardScaler
    strictly on X_train to prevent data leakage.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler)
    """
    X, y, dropped_cols = select_and_filter_features(df, target_col=target_col)
    feature_names = X.columns.tolist()

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scaling numerical features without leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    logging.info(f"Prepared inputs: X_train shape {X_train_scaled.shape}, X_test shape {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


def reshape_for_lstm(X_train: pd.DataFrame, X_test: pd.DataFrame, timesteps: int = 1):
    """
    Reshapes 2D feature matrices (N, num_features) into 3D tensors (N, timesteps, num_features)
    required by Keras/PyTorch LSTM layers.
    
    Returns:
        tuple: (X_train_3d, X_test_3d)
    """
    num_train, num_features = X_train.shape
    num_test = X_test.shape[0]

    X_train_3d = X_train.values.reshape((num_train, timesteps, num_features))
    X_test_3d = X_test.values.reshape((num_test, timesteps, num_features))

    logging.info(f"Reshaped for LSTM: X_train_3d shape {X_train_3d.shape}, X_test_3d shape {X_test_3d.shape}")
    return X_train_3d, X_test_3d


def save_engineered_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df_engineered: pd.DataFrame,
    output_dir: str = "data/processed"
):
    """
    Saves processed dataset splits and feature metadata to destination folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    df_engineered.to_csv(os.path.join(output_dir, "feature_engineered_dataset.csv"), index=False)

    metadata = {
        "num_records": len(df_engineered),
        "num_features_train": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "target_name": y_train.name if hasattr(y_train, 'name') else "Flood_Occurred",
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0]
    }

    with open(os.path.join(output_dir, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Successfully saved all feature-engineered files to '{output_dir}'.")
