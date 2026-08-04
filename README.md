# Construction & Real Estate – Geospatial Valuation via Spatial Embeddings

## 1. Project Overview
This project presents an end-to-end Machine Learning pipeline that predicts residential property values by combining physical housing characteristics with geographical coordinates (`latitude` and `longitude`). By converting raw coordinates into non-linear interaction features, spatial grid binnings, geodesic distances, and continuous standardized spatial embeddings, the machine learning models learn spatial sub-markets and neighborhood micro-dynamics to deliver highly accurate automated property appraisals.

## 2. Problem Statement
Traditional Automated Valuation Models (AVMs) primarily rely on tabular property attributes (such as square footage, room count, and building age) and administrative boundaries (ZIP codes). However, real estate values exhibit strong spatial autocorrelation—properties located near one another share economic micro-climates, coastal access, and transportation hubs. Tabular models fail to capture non-linear geographic gradients, while raw coordinates operating on different numerical scales can cause feature dominance issues.

## 3. Project Objectives
* **Predict Property Values**: Develop machine learning regression models for continuous price prediction ($y \in \mathbb{R}^+$).
* **Leverage Geospatial Information**: Utilize latitude and longitude to capture physical location dynamics.
* **Engineered Spatial Features**: Construct interaction terms, squared coordinate transformations, and grid cells.
* **Numerical Spatial Embeddings**: Standardize geographic features into zero-mean, unit-variance vector spaces (`spatial_emb_1` to `spatial_emb_6`).
* **Model Benchmarking**: Compare Linear Regression, Random Forest, Gradient Boosting, and XGBoost models.
* **Interactive Web Deployment**: Deploy a user-friendly Streamlit web application for real-time inference.

## 4. Dataset Information
* **Dataset Name**: California Housing Dataset (1990 U.S. Census Block Groups)
* **Dataset Source**: `sklearn.datasets.fetch_california_housing`
* **Total Sample Count**: 20,640 census block groups
* **Target Variable**: `house_value` (Median house value for households within a block group, in $100,000 units)
* **Features Included**:
  * `median_income`: Median household income ($10,000s)
  * `house_age`: Median age of houses in block group (years)
  * `average_rooms`: Average rooms per household
  * `average_bedrooms`: Average bedrooms per household
  * `population`: Total block group population
  * `average_occupancy`: Average household occupancy
  * `latitude`: Geographic latitude coordinate (°N)
  * `longitude`: Geographic longitude coordinate (°W)

## 5. Technologies Used
* **Programming Language**: Python 3.10+
* **Data Processing & Manipulation**: Pandas, NumPy
* **Geospatial & Math Tools**: SciPy, Haversine Great-Circle Distance
* **Machine Learning Frameworks**: Scikit-Learn, XGBoost, Joblib
* **Visualization & UI**: Matplotlib, Seaborn, Streamlit

## 6. Project Methodology
```
Data Collection (sklearn.datasets)
       │
       ▼
Data Cleaning & Quality Audit (src/data_preprocessing.py)
       │
       ▼
Exploratory Data Analysis & Visualization (reports/figures/)
       │
       ▼
Spatial Feature Engineering & Embeddings (src/feature_engineering.py)
       │
       ▼
80/20 Train / Test Dataset Split
       │
       ▼
Model Training & Comparison (src/train.py)
       │
       ▼
Best Model Selection & Evaluation (src/evaluate.py)
       │
       ▼
Inference & Prediction Pipeline (src/predict.py)
       │
       ▼
Interactive Web Application Deployment (app.py)
```

## 7. Spatial Feature Engineering

### Overview
Location is a fundamental driver of real estate valuation. Latitude and longitude anchor properties to physical space, allowing machine learning models to capture proximity to coastal lines and urban centers.

### Spatial Features Created
1. **Raw Coordinates**: `latitude`, `longitude`
2. **Interaction Features**:
   - `latitude_longitude` = `latitude` × `longitude`
   - `latitude_squared` = `latitude²`
   - `longitude_squared` = `longitude²`
3. **Spatial Grid Partitioning**: `latitude_grid`, `longitude_grid` ($0.1^\circ$ resolution, $\approx 11 \text{ km}$ bins), `spatial_grid_id`
4. **Geodesic Distance**: `distance_from_center` (Haversine distance in km from dataset center $35.6319^\circ\text{N}, -119.5697^\circ\text{W}$)

### Understanding Spatial Embeddings
Spatial embeddings are continuous standardized vector representations (`spatial_emb_1` through `spatial_emb_6`) constructed by applying `StandardScaler` to coordinate interaction and distance features. They project raw geographic dimensions into a unified, zero-mean, unit-variance space, preventing numerical scale disparities between coordinate degrees ($\approx 35^\circ$) and squared interactions ($\ge 1000$).

## 8. Machine Learning Models
Four supervised machine learning regression algorithms were trained using an 80/20 train/test split (16,512 train samples, 4,128 test samples) with `random_state=42`:
* **Linear Regression**: Baseline parametric model establishing linear relationship bounds.
* **Random Forest Regressor**: Parallel ensemble of 200 deep decision trees (`n_estimators=200`).
* **Gradient Boosting Regressor**: Sequential boosting ensemble (`n_estimators=200`).
* **XGBoost Regressor**: Optimized gradient boosted decision tree framework (`n_estimators=200`, `max_depth=6`, `learning_rate=0.05`).

Regression is appropriate because real estate price valuation is a continuous numeric prediction task ($y \in \mathbb{R}^+$).

## 9. Empirical Model Comparison Results

Evaluated on the independent 4,128 test sample holdout set:

| Model Name | MAE ($100k) | RMSE ($100k) | R² Score | Performance Rank |
|---|---:|---:|---:|:---:|
| **XGBoost Regressor** | **0.3104** | **0.4695** | **0.8318** | 🥇 **1 (Best Model)** |
| **Random Forest Regressor** | 0.3190 | 0.4939 | 0.8139 | 🥈 **2** |
| **Gradient Boosting Regressor** | 0.3402 | 0.5045 | 0.8058 | 🥉 **3** |
| **Linear Regression** | 0.5111 | 0.7218 | 0.6024 | 4 |

**Selected Best Model**: **XGBoost Regressor** attained the highest $R^2$ score (**0.8318**), lowest RMSE (**0.4695** / $46,950 USD), and lowest MAE (**0.3104** / $31,040 USD).

## 10. Feature Importance Results

Top 10 most influential features extracted from XGBoost decision tree split gains:

| Rank | Feature | Importance Score | Category |
|:---:|---|:---:|---|
| **1** | `median_income` | **0.3887** | Economic / Tabular |
| **2** | `average_occupancy` | **0.1170** | Structural / Demographic |
| **3** | `longitude_squared` | **0.0778** | Spatial Interaction |
| **4** | `latitude_grid` | **0.0761** | Spatial Grid |
| **5** | `latitude` | **0.0582** | Raw Coordinate |
| **6** | `longitude` | **0.0512** | Raw Coordinate |
| **7** | `house_age` | **0.0477** | Structural |
| **8** | `distance_from_center` | **0.0461** | Geodesic Distance |
| **9** | `longitude_grid` | **0.0447** | Spatial Grid |
| **10** | `latitude_longitude` | **0.0351** | Spatial Interaction |

*Key Insight*: Spatial features and interaction terms account for over **35%** of total prediction weight, confirming the value of geospatial feature engineering.

## 11. Property Valuation Web Application
The project includes an interactive web interface built with **Streamlit** (`app.py`):
* **Property Inputs**: Sliders and numerical input controls for 6 physical property attributes.
* **Geospatial Map**: Real-time interactive map rendering selected latitude and longitude coordinates.
* **Automated Embeddings**: Automatic calculation of spatial interactions, grid IDs, Haversine distance, and standardized spatial embeddings.
* **Market Price Output**: Instantaneous display of estimated property value in USD.
* **Model Information**: Dynamic loading of model selection and evaluation metrics from reports.

## 12. How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Individual Python Scripts
```bash
# Data Preprocessing
python src/data_preprocessing.py

# Feature Engineering & Spatial Embeddings
python src/feature_engineering.py

# Model Training & Comparison
python src/train.py

# Model Evaluation & Artifact Generation
python src/evaluate.py

# Sample Property Inference
python src/predict.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

## 13. Project Directory Structure
```text
geospatial-real-estate-valuation/
│
├── data/
│   ├── raw/
│   │   └── california_housing.csv
│   └── processed/
│       ├── cleaned_housing.csv
│       └── spatial_features.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── best_model.pkl
│   ├── best_model_name.txt
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── xgboost.pkl
│   ├── spatial_scaler.pkl
│   └── spatial_center.pkl
│
├── reports/
│   ├── eda_report.md
│   ├── PROJECT_REPORT.md
│   ├── final_checklist.md
│   ├── model_comparison.csv
│   ├── best_model_evaluation.txt
│   ├── predictions.csv
│   ├── feature_importance.csv
│   └── figures/
│       ├── distribution_house_values.png
│       ├── house_value_vs_median_income.png
│       ├── house_value_vs_house_age.png
│       ├── house_value_vs_average_rooms.png
│       ├── house_value_vs_latitude.png
│       ├── house_value_vs_longitude.png
│       ├── correlation_heatmap.png
│       ├── geographic_scatter_plot.png
│       ├── spatial_features_map.png
│       ├── model_r2_comparison.png
│       ├── best_model_predictions.png
│       ├── actual_vs_predicted.png
│       ├── residual_plot.png
│       ├── error_distribution.png
│       └── feature_importance.png
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 14. Project Limitations
* **Geographical Scope**: Dataset is restricted to California census block groups.
* **Target Value Truncation**: Target house values are capped at $500,000 ($5.0), limiting performance on high-end luxury estates.
* **Macroeconomic Changes**: Historical 1990 census baseline price levels require inflation indexing for modern market appraisals.
* **Unobserved Micro-Attributes**: Does not incorporate individual property condition, interior finishes, or localized school district boundaries.

## 15. Future Improvements
* **POI Integration**: Incorporate OpenStreetMap distances to public transit, top-rated schools, and commercial zones.
* **Advanced Deep Learning**: Train Spatial Graph Neural Networks (GNNs) or Tile2Vec neural spatial embeddings.
* **Cloud Deployment**: Deploy Streamlit web application on Streamlit Community Cloud or AWS.
* **Live API**: Wrap inference pipeline in a FastAPI REST service.

## 16. GitHub Setup

Follow these steps to upload this project repository to your personal or organization GitHub account:

1. **Create a New GitHub Repository**:
   - Go to [GitHub](https://github.com/new).
   - Set repository name: `geospatial-real-estate-valuation`
   - Set visibility to **Public** or **Private**.
   - Do **NOT** check "Initialize this repository with a README", `.gitignore`, or license (as local files are already fully configured).

2. **Connect Local Repository & Push**:
   Open a terminal in the project root directory and execute:

   ```bash
   git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
   git branch -M main
   git push -u origin main
   ```

## 17. Conclusion
This project successfully demonstrates that encoding geographic coordinates into spatial interaction terms, grid cell binnings, geodesic Haversine distance metrics, and continuous spatial embeddings significantly improves property price valuation accuracy. The **XGBoost Regressor** achieved an $R^2$ score of **0.8318** with an RMSE of **$46,950**, establishing a robust foundation for automated geospatial real estate valuation.
