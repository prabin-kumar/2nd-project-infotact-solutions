"""
Final Model Evaluation and Selection Script
Deep Learning-Based Flood Prediction Using Rainfall Data

This script loads pre-processed test datasets and pre-trained model artifacts,
evaluates all candidate models (Logistic Regression, Decision Tree, Random Forest, XGBoost),
generates unified comparison tables, confusion matrix grids, feature importance plots,
and exports all results to the results/ directory.
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_test_data(data_dir: str = "data/processed"):
    """Loads X_test.csv and y_test.csv from processed data directory."""
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        raise FileNotFoundError(f"Processed test datasets not found in '{data_dir}'.")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).iloc[:, 0]
    logging.info(f"Loaded test dataset: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    return X_test, y_test


def load_trained_models(models_dir: str = "models"):
    """Loads trained candidate model pkl artifacts and scaler from models directory."""
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    models = {}
    for name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filename}' not found in '{models_dir}'.")
        models[name] = joblib.load(filepath)
        logging.info(f"Loaded trained model artifact: '{name}' from '{filepath}'")

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return models, scaler


def generate_evaluation_summary(models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates each model on the test dataset and returns metrics DataFrame and confusion matrices dictionary.
    """
    metrics_list = []
    cms_dict = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-score": round(f1, 4)
        })
        cms_dict[name] = cm

    # Add LSTM Audit Entry
    metrics_list.append({
        "Model": "LSTM (Deep Learning)",
        "Accuracy": "N/A (Spatial Data)",
        "Precision": "N/A (Spatial Data)",
        "Recall": "N/A (Spatial Data)",
        "F1-score": "N/A (Spatial Data)"
    })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df, cms_dict


def plot_final_model_comparison(metrics_df: pd.DataFrame, output_dir: str = "results"):
    """Generates and saves professional grouped bar plot comparing metrics across all models."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter numerical models for plotting
    df_plot = metrics_df[metrics_df["Model"] != "LSTM (Deep Learning)"].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        df_plot[col] = df_plot[col].astype(float)

    df_melted = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Blues_d", ax=ax)
    ax.set_title("Final Machine Learning Model Performance Benchmark", fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Classifier Model", fontsize=12)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h / 2.),
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold', rotation=90)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    filepath = os.path.join(output_dir, "final_model_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Saved final comparison bar plot to '{filepath}'.")


def plot_final_confusion_matrices(cms_dict: dict, output_dir: str = "results"):
    """Generates 2x2 grid visualizing confusion matrices for all 4 trained models."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    model_names = list(cms_dict.keys())
    cm_colors = ["Blues", "Oranges", "Greens", "Purples"]

    for idx, name in enumerate(model_names):
        cm = cms_dict[name]
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cm_colors[idx], cbar=False, ax=ax,
                    xticklabels=["No Flood (0)", "Flood (1)"],
                    yticklabels=["No Flood (0)", "Flood (1)"])
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label (1 = Flood)", fontsize=10)
        ax.set_ylabel("True Label (1 = Flood)", fontsize=10)

    plt.suptitle("Final Model Confusion Matrices Comparison", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = os.path.join(output_dir, "final_confusion_matrices.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Saved final 2x2 confusion matrices grid to '{filepath}'.")


def plot_final_feature_importance(xgb_model, feature_names: list, top_n: int = 15, output_dir: str = "results"):
    """Generates feature importance bar plot for the top performing tree model (XGBoost)."""
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(xgb_model, "feature_importances_"):
        imp_series = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(x=imp_series.values, y=imp_series.index, hue=imp_series.index, palette="mako", legend=False, ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances — XGBoost Classifier", fontsize=13, fontweight="bold")
        ax.set_xlabel("Relative Gini Feature Importance Score", fontsize=11)
        ax.set_ylabel("Feature Name", fontsize=11)
        plt.tight_layout()

        filepath = os.path.join(output_dir, "final_feature_importance.png")
        plt.savefig(filepath, dpi=300)
        plt.close()
        logging.info(f"Saved final feature importance plot to '{filepath}'.")


def main():
    data_dir = "data/processed"
    models_dir = "models"
    output_dir = "results"

    X_test, y_test = load_test_data(data_dir)
    models, scaler = load_trained_models(models_dir)

    metrics_df, cms_dict = generate_evaluation_summary(models, X_test, y_test)

    # Save final_model_comparison.csv
    csv_path = os.path.join(output_dir, "final_model_comparison.csv")
    metrics_df.to_csv(csv_path, index=False)
    logging.info(f"Saved unified final evaluation results to '{csv_path}'.")

    # Generate plots
    plot_final_model_comparison(metrics_df, output_dir)
    plot_final_confusion_matrices(cms_dict, output_dir)
    plot_final_feature_importance(models["XGBoost"], X_test.columns.tolist(), top_n=15, output_dir=output_dir)

    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON UNIFIED RESULTS")
    print("="*70)
    print(metrics_df.to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
