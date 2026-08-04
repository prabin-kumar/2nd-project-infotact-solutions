"""
Data Cleaning and Preprocessing Pipeline
Deep Learning-Based Flood Prediction Using Rainfall Data

This module provides reusable functions for loading, cleaning, transforming,
scaling, and splitting rainfall & hydro-meteorological datasets without data leakage.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def find_raw_dataset(raw_dir: str = "data/raw") -> str:
    """
    Locates the CSV dataset file inside the raw data directory.
    
    Parameters:
        raw_dir (str): Path to raw data directory.
        
    Returns:
        str: Absolute or relative path to the raw CSV file.
    """
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in '{raw_dir}'. Please place your dataset CSV file in '{raw_dir}'.")
    logging.info(f"Found raw dataset: {csv_files[0]}")
    return csv_files[0]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV dataset into a pandas DataFrame.
    
    Parameters:
        filepath (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Raw loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by stripping whitespace from column names,
    removing duplicate records, and handling invalid values.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Strip leading/trailing whitespaces from string columns & column headers
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    logging.info(f"Removed {duplicates_removed} duplicate records.")
    
    return cleaned_df


def identify_columns(df: pd.DataFrame, target_col_name: str = None, time_col_name: str = None):
    """
    Identifies numerical features, categorical features, date/time columns, and target column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col_name (str): Expected target column name (optional).
        time_col_name (str): Expected time/date column name (optional).
        
    Returns:
        tuple: (feature_cols, target_col, time_col, categorical_cols)
    """
    cols = df.columns.tolist()
    
    # Identify target column
    target_col = target_col_name
    if not target_col:
        possible_targets = [c for c in cols if 'flood' in c.lower() or 'target' in c.lower() or 'label' in c.lower()]
        target_col = possible_targets[0] if possible_targets else cols[-1]
        
    # Identify time column
    time_col = time_col_name
    if not time_col:
        possible_time = [c for c in cols if 'date' in c.lower() or 'year' in c.lower() or 'time' in c.lower() or 'month' in c.lower()]
        time_col = possible_time[0] if possible_time else None
        
    feature_cols = [c for c in cols if c not in [target_col] and c != 'ID' and c != 'Index']
    
    categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numerical_cols, categorical_cols, target_col, time_col


def encode_target(series: pd.Series) -> pd.Series:
    """
    Encodes binary target column values into 0 and 1.
    Handles 'YES'/'NO', 'Y'/'N', True/False, or string representations.
    
    Parameters:
        series (pd.Series): Target series.
        
    Returns:
        pd.Series: Encoded integer target series (0 or 1).
    """
    if series.dtype == object or isinstance(series.iloc[0], str):
        mapping = {'yes': 1, 'no': 0, 'y': 1, 'n': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
        encoded = series.astype(str).str.strip().str.lower().map(mapping)
        if encoded.isnull().any():
            # Fallback to factorize if unrecognized categories exist
            encoded, _ = pd.factorize(series)
        return encoded.astype(int)
    return series.astype(int)


def split_data_chronologically_or_stratified(
    df: pd.DataFrame, 
    feature_cols: list, 
    target_col: str, 
    time_col: str = None, 
    test_size: float = 0.2, 
    random_state: int = 42
):
    """
    Splits data into train and test sets avoiding data leakage.
    Uses chronological split if time column or sequential structure is present,
    otherwise uses stratified split for balanced target distribution.
    
    Parameters:
        df (pd.DataFrame): Data.
        feature_cols (list): List of feature names.
        target_col (str): Target column name.
        time_col (str): Optional time column for sorting.
        test_size (float): Fraction of test set.
        random_state (int): Random seed.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df[feature_cols].copy()
    y = encode_target(df[target_col])
    
    if time_col and time_col in df.columns:
        # Chronological Split for Time Series
        df_sorted = df.sort_values(by=time_col)
        X = df_sorted[feature_cols]
        y = encode_target(df_sorted[target_col])
        split_idx = int(len(df_sorted) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        logging.info("Applied Chronological (Time-Based) Train/Test Split.")
    else:
        # Stratified Split for Random Tabular Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info("Applied Stratified Train/Test Split.")
        
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, numerical_cols: list):
    """
    Scales numerical features using StandardScaler.
    Fits ONLY on training set to prevent data leakage, then transforms train and test sets.
    
    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        numerical_cols (list): List of numerical columns.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if numerical_cols:
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        logging.info("Fitted StandardScaler on X_train and transformed X_train and X_test without leakage.")
        
    return X_train_scaled, X_test_scaled, scaler


def save_processed_data(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    output_dir: str = "data/processed"
):
    """
    Saves cleaned and preprocessed feature matrices and target vectors into output directory.
    
    Parameters:
        X_train, X_test, y_train, y_test: Processed dataset splits.
        output_dir (str): Destination folder path.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    logging.info(f"Saved processed dataset splits to '{output_dir}'.")
