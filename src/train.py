"""
Model Training & Model Comparison Module
----------------------------------------
This module loads the feature-engineered housing dataset, performs an 80/20 train-test split,
trains multiple regression models (Linear Regression, Random Forest, Gradient Boosting, XGBoost),
evaluates metrics (MAE, RMSE, R2), saves all trained artifacts and the overall best model,
and generates performance comparison visualizations.

Input Dataset: data/processed/spatial_features.csv
Saved Models: models/*.pkl, models/best_model.pkl, models/best_model_name.txt
Reports: reports/model_comparison.csv
Figures: reports/figures/model_r2_comparison.png, reports/figures/best_model_predictions.png
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try importing XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost is not installed. Skipping XGBoost model.")

def get_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    base_dir = r"C:\Users\hkris\OneDrive\Desktop\Construction & Real Estate – Geospatial Valuation via Spatial Embeddings\geospatial-real-estate-valuation"
    input_csv = os.path.join(base_dir, "data", "processed", "spatial_features.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(base_dir, "reports", "figures")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Separate Target and Features
    target_col = "house_value"
    exclude_cols = [target_col, "spatial_grid_id"]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any missing or infinite values if present
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Instantiate Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror"
        )

    # Model evaluation loop
    results = []
    trained_models = {}
    test_predictions = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = get_rmse(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        })

        trained_models[name] = model
        test_predictions[name] = preds

        # Save individual model
        filename_map = {
            "Linear Regression": "linear_regression.pkl",
            "Random Forest": "random_forest.pkl",
            "Gradient Boosting": "gradient_boosting.pkl",
            "XGBoost": "xgboost.pkl"
        }
        save_path = os.path.join(models_dir, filename_map[name])
        joblib.dump(model, save_path)
        print(f"Saved {name} model to {save_path}")

    # Build Comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=["R2", "RMSE"], ascending=[False, True], inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    print("\n=== MODEL COMPARISON TABLE ===")
    print(results_df.to_string(index=False))

    # Save Comparison Table
    comparison_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison table saved to: {comparison_csv_path}")

    # Select Best Model
    best_model_name = results_df.iloc[0]["Model"]
    best_model_r2 = results_df.iloc[0]["R2"]
    best_model_obj = trained_models[best_model_name]

    print(f"\n>>> BEST MODEL SELECTED: {best_model_name} (R² = {best_model_r2:.4f}) <<<")

    best_model_path = os.path.join(models_dir, "best_model.pkl")
    best_model_name_path = os.path.join(models_dir, "best_model_name.txt")

    joblib.dump(best_model_obj, best_model_path)
    with open(best_model_name_path, "w") as f:
        f.write(best_model_name)

    print(f"Best model artifact saved to: {best_model_path}")
    print(f"Best model name recorded in: {best_model_name_path}")

    # Visualizations
    sns.set_theme(style="whitegrid")

    # 1. Bar Chart of R2 Comparison
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=results_df, x="Model", y="R2", palette="viridis"
    )
    plt.title("Model Comparison - R² Score", fontsize=14, fontweight="bold")
    plt.xlabel("Machine Learning Model", fontsize=12)
    plt.ylabel("R² Score (Coefficient of Determination)", fontsize=12)
    plt.ylim(0, 1.0)
    
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.4f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points'
        )

    r2_fig_path = os.path.join(figures_dir, "model_r2_comparison.png")
    plt.tight_layout()
    plt.savefig(r2_fig_path, dpi=300)
    plt.close()
    print(f"Saved R² comparison plot to: {r2_fig_path}")

    # 2. Predicted vs Actual Plot for Best Model
    best_preds = test_predictions[best_model_name]
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_preds, alpha=0.3, color="#1f77b4", label="Predictions")
    
    # Perfect prediction line
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect 1:1 Line")
    
    plt.title(f"Predicted vs. Actual House Values ({best_model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Actual House Value ($100k)", fontsize=12)
    plt.ylabel("Predicted House Value ($100k)", fontsize=12)
    plt.legend(loc="upper left")
    
    pred_fig_path = os.path.join(figures_dir, "best_model_predictions.png")
    plt.tight_layout()
    plt.savefig(pred_fig_path, dpi=300)
    plt.close()
    print(f"Saved predicted-vs-actual plot to: {pred_fig_path}")

if __name__ == "__main__":
    main()
