"""
Feature Engineering & Spatial Embeddings Module
------------------------------------------------
This module loads the cleaned housing dataset, constructs non-linear interaction features,
spatial grid cell IDs, distance metrics to the dataset center using the Haversine formula,
and standardized continuous spatial embedding vectors.

Output:
- data/processed/spatial_features.csv
- models/spatial_scaler.pkl
- models/spatial_center.pkl
- reports/figures/spatial_features_map.png
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def create_spatial_features(df: pd.DataFrame, grid_size: float = 0.1, models_dir: str = None) -> pd.DataFrame:
    """
    Generate coordinate interaction features, spatial grid cells, 
    and distance-based features from raw latitude and longitude.
    Saves fitted StandardScaler to models/spatial_scaler.pkl.
    """
    df_feat = df.copy()

    # 1. Interaction Features
    df_feat['latitude_longitude'] = df_feat['latitude'] * df_feat['longitude']
    df_feat['latitude_squared'] = df_feat['latitude'] ** 2
    df_feat['longitude_squared'] = df_feat['longitude'] ** 2

    # 2. Spatial Grid Features (0.1 degree resolution)
    df_feat['latitude_grid'] = np.floor(df_feat['latitude'] / grid_size) * grid_size
    df_feat['longitude_grid'] = np.floor(df_feat['longitude'] / grid_size) * grid_size
    
    # Combine latitude_grid and longitude_grid into spatial_grid_id
    df_feat['spatial_grid_id'] = (
        df_feat['latitude_grid'].round(2).astype(str) + "_" + 
        df_feat['longitude_grid'].round(2).astype(str)
    )

    # 3. Distance-based Features (Distance from dataset mean center)
    mean_lat = float(df_feat['latitude'].mean())
    mean_lon = float(df_feat['longitude'].mean())
    
    df_feat['distance_from_center'] = haversine_distance(
        df_feat['latitude'], df_feat['longitude'], mean_lat, mean_lon
    )

    # Save spatial center coordinates for inference consistency
    if models_dir:
        center_path = os.path.join(models_dir, "spatial_center.pkl")
        joblib.dump({"mean_lat": mean_lat, "mean_lon": mean_lon}, center_path)
        print(f"Saved spatial center coordinates to: {center_path}")

    # 4. Standardized Spatial Embedding Vectors
    spatial_components = [
        'latitude',
        'longitude',
        'latitude_squared',
        'longitude_squared',
        'latitude_longitude',
        'distance_from_center'
    ]

    scaler = StandardScaler()
    embedded_matrix = scaler.fit_transform(df_feat[spatial_components])

    if models_dir:
        scaler_path = os.path.join(models_dir, "spatial_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Saved fitted spatial scaler to: {scaler_path}")

    embedding_cols = [
        'spatial_emb_1',
        'spatial_emb_2',
        'spatial_emb_3',
        'spatial_emb_4',
        'spatial_emb_5',
        'spatial_emb_6'
    ]

    for i, col_name in enumerate(embedding_cols):
        df_feat[col_name] = embedded_matrix[:, i]

    return df_feat

def generate_spatial_visualization(df_feat: pd.DataFrame, output_image_path: str):
    """
    Generates a 2x2 grid figure illustrating spatial feature transformations.
    Saved to reports/figures/spatial_features_map.png
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    sns.set_theme(style="whitegrid")

    sc1 = axes[0, 0].scatter(
        df_feat['longitude'], df_feat['latitude'],
        c=df_feat['distance_from_center'], cmap='magma', s=10, alpha=0.7
    )
    fig.colorbar(sc1, ax=axes[0, 0], label='Distance from Center (km)')
    axes[0, 0].set_title('Spatial Feature 1: Haversine Distance from Center (km)')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')

    sc2 = axes[0, 1].scatter(
        df_feat['longitude'], df_feat['latitude'],
        c=df_feat['latitude_longitude'], cmap='coolwarm', s=10, alpha=0.7
    )
    fig.colorbar(sc2, ax=axes[0, 1], label='Lat * Lon Interaction')
    axes[0, 1].set_title('Spatial Feature 2: Latitude × Longitude Interaction')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')

    sc3 = axes[1, 0].scatter(
        df_feat['longitude'], df_feat['latitude'],
        c=df_feat['spatial_emb_1'], cmap='viridis', s=10, alpha=0.7
    )
    fig.colorbar(sc3, ax=axes[1, 0], label='spatial_emb_1 (Standardized Lat)')
    axes[1, 0].set_title('Spatial Embedding Vector 1 (spatial_emb_1)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')

    sc4 = axes[1, 1].scatter(
        df_feat['longitude'], df_feat['latitude'],
        c=df_feat['spatial_emb_6'], cmap='plasma', s=10, alpha=0.7
    )
    fig.colorbar(sc4, ax=axes[1, 1], label='spatial_emb_6 (Standardized Distance)')
    axes[1, 1].set_title('Spatial Embedding Vector 6 (spatial_emb_6)')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

def main():
    base_dir = r"C:\Users\hkris\OneDrive\Desktop\Construction & Real Estate – Geospatial Valuation via Spatial Embeddings\geospatial-real-estate-valuation"
    input_csv = os.path.join(base_dir, "data", "processed", "cleaned_housing.csv")
    output_csv = os.path.join(base_dir, "data", "processed", "spatial_features.csv")
    models_dir = os.path.join(base_dir, "models")
    figure_path = os.path.join(base_dir, "reports", "figures", "spatial_features_map.png")

    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading cleaned dataset from {input_csv}...")
    df_cleaned = pd.read_csv(input_csv)
    original_shape = df_cleaned.shape

    print("Generating spatial features and saving spatial scaler...")
    df_processed = create_spatial_features(df_cleaned, grid_size=0.1, models_dir=models_dir)

    print(f"Saving feature-engineered dataset to {output_csv}...")
    df_processed.to_csv(output_csv, index=False)
    final_shape = df_processed.shape

    print(f"Generating spatial feature maps at {figure_path}...")
    generate_spatial_visualization(df_processed, figure_path)

    print("\n--- FEATURE ENGINEERING SUMMARY ---")
    print(f"Original features count: {original_shape[1]}")
    print(f"Final features count: {final_shape[1]}")
    print(f"New spatial features added: {final_shape[1] - original_shape[1]}")

if __name__ == "__main__":
    main()
