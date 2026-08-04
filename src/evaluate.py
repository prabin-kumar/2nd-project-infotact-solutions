"""
Model Evaluation Module
-----------------------
This module loads the best trained model artifact (models/best_model.pkl),
evaluates it on the test dataset split (test_size=0.2, random_state=42),
generates metric evaluation text reports, actual vs. predicted CSV artifacts,
residual plots, error distribution plots, feature importance tables, and visualizations.

Outputs:
- reports/best_model_evaluation.txt
- reports/predictions.csv
- reports/feature_importance.csv
- reports/figures/actual_vs_predicted.png
- reports/figures/residual_plot.png
- reports/figures/error_distribution.png
- reports/figures/feature_importance.png
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def get_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    base_dir = r"C:\Users\hkris\OneDrive\Desktop\Construction & Real Estate – Geospatial Valuation via Spatial Embeddings\geospatial-real-estate-valuation"
    input_csv = os.path.join(base_dir, "data", "processed", "spatial_features.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(base_dir, "reports", "figures")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load best model and model name
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    best_model_name_path = os.path.join(models_dir, "best_model_name.txt")

    print(f"Loading best model from {best_model_path}...")
    best_model = joblib.load(best_model_path)

    with open(best_model_name_path, "r") as f:
        best_model_name = f.read().strip()

    print(f"Best Model Selected: {best_model_name}")

    # 2. Load dataset and reproduce exact train/test split
    df = pd.read_csv(input_csv)
    target_col = "house_value"
    exclude_cols = [target_col, "spatial_grid_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Test Set Evaluation Samples: {X_test.shape[0]}")

    # 3. Generate predictions & metrics
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = get_rmse(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- EVALUATION METRICS ({best_model_name}) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Save metrics report
    eval_txt_path = os.path.join(reports_dir, "best_model_evaluation.txt")
    with open(eval_txt_path, "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Test Set Samples: {len(y_test)}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R-squared (R2) Score: {r2:.4f}\n")
    print(f"Saved evaluation metrics report to: {eval_txt_path}")

    # 4. Save Actual vs Predicted CSV
    predictions_df = pd.DataFrame({
        "actual_house_value": y_test.values,
        "predicted_house_value": y_pred,
        "prediction_error": y_test.values - y_pred
    })
    preds_csv_path = os.path.join(reports_dir, "predictions.csv")
    predictions_df.to_csv(preds_csv_path, index=False)
    print(f"Saved test predictions table to: {preds_csv_path}")

    # 5. Create Evaluation Plots
    sns.set_theme(style="whitegrid")

    # A. Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="#1f77b4", edgecolors='none')
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label="Ideal 1:1 Prediction")
    plt.title(f"Actual vs. Predicted House Values ({best_model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Actual House Value ($100k)", fontsize=12)
    plt.ylabel("Predicted House Value ($100k)", fontsize=12)
    plt.legend(loc="upper left")
    p_act_pred = os.path.join(figures_dir, "actual_vs_predicted.png")
    plt.savefig(p_act_pred, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_act_pred}")

    # B. Residual Plot
    residuals = y_test.values - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, color="#e377c2", edgecolors='none')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title(f"Residual Plot ({best_model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted House Value ($100k)", fontsize=12)
    plt.ylabel("Residual (Actual - Predicted)", fontsize=12)
    p_residual = os.path.join(figures_dir, "residual_plot.png")
    plt.savefig(p_residual, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_residual}")

    # C. Prediction Error Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=50, color="#2ca02c")
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title("Prediction Error Distribution (Residuals)", fontsize=14, fontweight="bold")
    plt.xlabel("Prediction Error (Actual - Predicted in $100k)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    p_err_dist = os.path.join(figures_dir, "error_distribution.png")
    plt.savefig(p_err_dist, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_err_dist}")

    # 6. Feature Importance Calculation
    print("\nCalculating Feature Importances...")
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        perm_res = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_res.importances_mean

    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    feat_imp_csv_path = os.path.join(reports_dir, "feature_importance.csv")
    feat_imp_df.to_csv(feat_imp_csv_path, index=False)
    print(f"Saved feature importance table to: {feat_imp_csv_path}")

    print("\n--- TOP 10 IMPORTANT FEATURES ---")
    print(feat_imp_df.head(10).to_string(index=False))

    # Top 15 Feature Importance Bar Plot
    top15_df = feat_imp_df.head(15).copy()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top15_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False)
    plt.title(f"Top 15 Most Important Features ({best_model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    p_feat_imp = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(p_feat_imp, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance bar chart to: {p_feat_imp}")

if __name__ == "__main__":
    main()
