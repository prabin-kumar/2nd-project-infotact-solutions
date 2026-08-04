"""
Machine Learning Training and Evaluation Module
Deep Learning-Based Flood Prediction Using Rainfall Data

This module loads feature-engineered datasets, trains machine learning models
(Logistic Regression, Decision Tree, Random Forest, XGBoost), evaluates performance,
plots confusion matrices & feature importances, and saves models and metrics.
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_processed_data(data_dir: str = "data/processed"):
    """
    Loads feature-engineered train and test datasets from data/processed directory.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    if not (os.path.exists(X_train_path) and os.path.exists(X_test_path)):
        raise FileNotFoundError(f"Processed dataset files not found in '{data_dir}'. Ensure Step 6 is executed.")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    feature_names = X_train.columns.tolist()
    logging.info(f"Loaded processed data: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """Trains Logistic Regression classifier."""
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    logging.info("Trained Logistic Regression Classifier.")
    return model


def train_decision_tree(X_train, y_train, max_depth: int = 10, random_state: int = 42):
    """Trains Decision Tree classifier."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    logging.info("Trained Decision Tree Classifier.")
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
    """Trains Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    logging.info("Trained Random Forest Classifier.")
    return model


def train_xgboost(X_train, y_train, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6, random_state: int = 42):
    """Trains XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logging.info("Trained XGBoost Classifier.")
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluates a trained classifier on the test set and calculates key metrics.
    
    Returns:
        tuple: (metrics_dict, conf_matrix)
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4)
    }

    logging.info(f"[{model_name}] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    return metrics, cm


def plot_and_save_confusion_matrix(cm, model_name: str, output_dir: str = "results"):
    """Plot and save confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No Flood (0)", "Flood (1)"],
                yticklabels=["No Flood (0)", "Flood (1)"])
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    
    filename = f"cm_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Saved confusion matrix plot to '{filepath}'.")


def plot_and_save_feature_importance(importance_series: pd.Series, model_name: str, top_n: int = 15, output_dir: str = "results"):
    """Plot and save top feature importances."""
    os.makedirs(output_dir, exist_ok=True)
    top_importances = importance_series.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_importances.values, y=top_importances.index, hue=top_importances.index, palette="viridis", legend=False, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances ({model_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_ylabel("Feature Name", fontsize=11)
    plt.tight_layout()

    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Saved feature importance plot to '{filepath}'.")


def plot_and_save_model_comparison(metrics_df: pd.DataFrame, output_dir: str = "results"):
    """Plot grouped bar chart comparing Accuracy, Precision, Recall, F1-score across models."""
    os.makedirs(output_dir, exist_ok=True)
    
    df_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Set2", ax=ax)
    ax.set_title("Machine Learning Models Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Machine Learning Model", fontsize=11)
    
    # Annotate bar scores
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', fontsize=8, color='black', rotation=90)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    filepath = os.path.join(output_dir, "model_performance_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    logging.info(f"Saved model performance comparison plot to '{filepath}'.")


def train_and_evaluate_all_models(data_dir: str = "data/processed", output_dir: str = "results", models_dir: str = "models"):
    """
    Main orchestration function to train all 4 models, evaluate, export metrics, plot, and serialize models.
    """
    X_train, X_test, y_train, y_test, feature_names = load_processed_data(data_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train),
        "Decision Tree": train_decision_tree(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train)
    }

    metrics_list = []
    cms_dict = {}

    for name, model in models.items():
        metrics, cm = evaluate_model(model, X_test, y_test, name)
        metrics_list.append(metrics)
        cms_dict[name] = cm
        plot_and_save_confusion_matrix(cm, name, output_dir)

        # Feature Importance for Random Forest & XGBoost
        if hasattr(model, "feature_importances_"):
            importance_series = pd.Series(model.feature_importances_, index=feature_names)
            plot_and_save_feature_importance(importance_series, name, top_n=15, output_dir=output_dir)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv_path = os.path.join(output_dir, "ml_model_comparison.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Saved model comparison table to '{metrics_csv_path}'.")

    plot_and_save_model_comparison(metrics_df, output_dir)

    # Save models to models/ directory
    model_filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    for name, model in models.items():
        save_path = os.path.join(models_dir, model_filename_map[name])
        joblib.dump(model, save_path)
        logging.info(f"Saved trained model '{name}' to '{save_path}'.")

    # Create dummy scaler object or fit scaler for deployment consistency
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    logging.info("Saved fitted StandardScaler to 'models/scaler.pkl'.")

    return metrics_df, models, cms_dict
