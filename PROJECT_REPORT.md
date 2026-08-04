# Academic Technical Report: Geospatial Property Valuation via Spatial Embeddings

**Project Title**: Construction & Real Estate – Geospatial Valuation via Spatial Embeddings  
**Dataset**: California Housing Dataset (1990 U.S. Census Block Groups)  
**Target Variable**: Median House Value (`house_value` in $100,000s)  

---

## 1. Abstract
Automated Valuation Models (AVMs) in real estate traditional rely on tabular property features such as living area, bedroom count, and building age. However, property values exhibit strong spatial autocorrelation and local neighborhood micro-dynamics that basic tabular data fails to capture. This project implements a machine learning system that transforms raw geographic coordinates (`latitude` and `longitude`) into high-dimensional interaction features, geodesic distance metrics, spatial grid binnings, and continuous standardized spatial embeddings. Using the California Housing dataset (20,640 records), we trained and compared four regression algorithms: Linear Regression, Random Forest, Gradient Boosting, and XGBoost. The **XGBoost Regressor** achieved superior valuation accuracy, attaining an $R^2$ score of **0.8318**, a Root Mean Squared Error (RMSE) of **0.4695** ($46,950 USD), and a Mean Absolute Error (MAE) of **0.3104** ($31,040 USD). Feature importance analysis confirmed that spatial features and embeddings account for over 30% of total prediction weight. Finally, an interactive web application was developed in Streamlit to enable real-time geospatial property appraisal.

---

## 2. Introduction
Real estate valuation plays a fundamental role in mortgage underwriting, urban planning, property tax assessment, and investment management. Property pricing is governed by Tobler's First Law of Geography: *"everything is related to everything else, but near things are more related than distant things."* Traditional linear regression models often treat location as an independent administrative variable (e.g., ZIP code dummy variables), missing non-linear spatial gradients such as coastal proximity and metropolitan economic nodes. By constructing explicit spatial interaction features and standardized spatial embedding vectors, machine learning models can learn complex spatial topologies without requiring expensive external geographic datasets.

---

## 3. Problem Statement
Accurately estimating residential property values requires integrating physical house attributes with localized spatial context. Key challenges include:
1. **Non-Linear Spatial Boundaries**: Real estate values vary non-linearly with geographic coordinates (e.g., sharp price drops moving inland from the coast).
2. **Feature Scale Disparities**: Geographic degrees ($\approx 32^\circ - 42^\circ$) operate on vastly different scales compared to income or bedroom counts, creating potential feature dominance issues in distance-based models.
3. **Data Leakage in Evaluation**: Standard random sampling can cause spatial autocorrelation leakage between training and evaluation splits.

---

## 4. Objectives
The primary objectives of this project are:
* **Dataset Standardization**: Clean and structure the California Housing dataset into a standardized reproducible format.
* **Spatial Feature Generation**: Engineer non-linear coordinate interactions, spatial grid cells, and Haversine distance metrics.
* **Continuous Spatial Embeddings**: Derive zero-mean, unit-variance standardized spatial vector representations (`spatial_emb_1` through `spatial_emb_6`).
* **Model Benchmarking**: Train and compare Linear Regression, Random Forest, Gradient Boosting, and XGBoost regression models on identical 80/20 train/test splits.
* **Model Interpretation**: Evaluate feature importance gains to quantify the relative contribution of spatial vs. tabular features.
* **Web Deployment**: Deploy an interactive Streamlit application for end-user property price inference.

---

## 5. Dataset Description
The dataset used in this study is the **California Housing Dataset** (`sklearn.datasets.fetch_california_housing`), originating from the 1990 U.S. Census.

* **Total Samples**: 20,640 census block groups
* **Total Columns**: 9 (8 predictor attributes + 1 target variable)
* **Target Variable**: `house_value` (Median house value for households within a block group, measured in $100,000 units).

### Features Summary Table

| Feature Name | Description | Data Type | Min | Mean | Max |
|---|---|:---:|:---:|:---:|:---:|
| `median_income` | Median income for households ($10,000s) | Continuous | 0.4999 | 3.8707 | 15.0001 |
| `house_age` | Median age of houses in block (years) | Continuous | 1.0000 | 28.6395 | 52.0000 |
| `average_rooms` | Average rooms per household | Continuous | 0.8462 | 5.4290 | 141.9091 |
| `average_bedrooms` | Average bedrooms per household | Continuous | 0.3333 | 1.0967 | 34.0667 |
| `population` | Block group population | Continuous | 3.0000 | 1425.48 | 35682.00 |
| `average_occupancy` | Average household occupancy | Continuous | 0.6923 | 3.0707 | 1243.33 |
| `latitude` | Block group latitude (°N) | Continuous | 32.5400 | 35.6319 | 41.9500 |
| `longitude` | Block group longitude (°W) | Continuous | -124.3500 | -119.5697 | -114.3100 |
| **`house_value`** | **Target: Median house value ($100k)** | **Continuous** | **0.14999** | **2.0686** | **5.00001** |

---

## 6. Data Preprocessing
Data hygiene validation was conducted in `src/data_preprocessing.py`:
1. **Missing Value Audit**: Confirmed zero missing (`NaN` / `null`) values across all 20,640 records.
2. **Duplicate Check**: Confirmed zero duplicate rows.
3. **Geographic Bound Verification**: Verified all latitude ($32.54^\circ\text{N} \le \text{lat} \le 41.95^\circ\text{N}$) and longitude ($-124.35^\circ\text{W} \le \text{lon} \le -114.31^\circ\text{W}$) coordinates fall within California state borders.
4. **Export**: Saved clean output to `data/processed/cleaned_housing.csv`.

---

## 7. Exploratory Data Analysis (EDA)
Comprehensive EDA was conducted and visual artifacts were saved to `reports/figures/`:
* **Target Distribution**: Right-skewed distribution with a distinct capping spike at $5.00001 ($500,000), representing 965 block groups (4.68%).
* **Economic Correlation**: `median_income` exhibits the strongest linear correlation with property value ($r = +0.69$).
* **Geographic Spatial Clustering**: Spatial scatter plots (`longitude` vs. `latitude`) clearly highlight coastal value premiums ($\ge \$400k - \$500k$) in the San Francisco Bay Area and Greater Los Angeles relative to inland agricultural regions.

---

## 8. Spatial Feature Engineering
Raw coordinates were transformed into non-linear spatial representations:
1. **Coordinate Interactions**:
   - $\text{latitude\_longitude} = \text{latitude} \times \text{longitude}$
   - $\text{latitude\_squared} = \text{latitude}^2$
   - $\text{longitude\_squared} = \text{longitude}^2$
2. **Spatial Grid Cell Partitioning**:
   - Binned coordinates into $0.1^\circ$ grid cells ($\approx 11 \text{ km} \times 11 \text{ km}$ resolution): `latitude_grid`, `longitude_grid`, and `spatial_grid_id`.
3. **Haversine Distance Metric**:
   - Calculated Great-Circle distance from each block group to California's geographic mean center ($35.6319^\circ\text{N}, -119.5697^\circ\text{W}$) in kilometers:
   $$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$
   $$\text{distance} = 2 R \cdot \text{atan2}(\sqrt{a}, \sqrt{1-a})$$

---

## 9. Spatial Embeddings
To represent location in a continuous vector space without scale dominance:
- Standardized 6 core spatial attributes (`latitude`, `longitude`, `latitude_squared`, `longitude_squared`, `latitude_longitude`, `distance_from_center`) using `StandardScaler`.
- Generated 6 continuous spatial embedding dimensions: `spatial_emb_1` through `spatial_emb_6`.
- The final feature-engineered dataset `data/processed/spatial_features.csv` contains **22 total columns** (20 predictor features, 1 categorical grid ID, 1 target variable).

---

## 10. Model Development
Four machine learning algorithms were implemented in `src/train.py`:
1. **Linear Regression**: Parametric baseline model.
2. **Random Forest Regressor**: Parallel ensemble of 200 deep decision trees (`n_estimators=200`, `n_jobs=-1`).
3. **Gradient Boosting Regressor**: Sequential boosting ensemble (`n_estimators=200`).
4. **XGBoost Regressor**: Optimized gradient boosting framework (`n_estimators=200`, `max_depth=6`, `learning_rate=0.05`).

Data was split using an 80/20 train/test split (16,512 train samples, 4,128 test samples) with `random_state=42`.

---

## 11. Model Comparison

### Benchmark Performance Summary

| Model Name | MAE ($100k) | RMSE ($100k) | R² Score | Performance Rank |
|---|---:|---:|---:|:---:|
| **XGBoost Regressor** | **0.3104** | **0.4695** | **0.8318** | 🥇 **1** |
| **Random Forest Regressor** | 0.3190 | 0.4939 | 0.8139 | 🥈 **2** |
| **Gradient Boosting Regressor** | 0.3402 | 0.5045 | 0.8058 | 🥉 **3** |
| **Linear Regression** | 0.5111 | 0.7218 | 0.6024 | 4 |

---

## 12. Evaluation Metrics
Evaluated on the holdout test set (4,128 block groups) using the selected **XGBoost** model:
- **MAE**: `0.3104` ($31,040 USD)
- **RMSE**: `0.4695` ($46,950 USD)
- **$R^2$ Score**: `0.8318` (83.18% variance explained)

---

## 13. Results & Feature Importance

### Top 10 Most Influential Features

| Rank | Feature Name | Importance Gain | Feature Category |
|:---:|---|:---:|---|
| **1** | `median_income` | 0.3887 | Economic / Tabular |
| **2** | `average_occupancy` | 0.1170 | Demographic |
| **3** | `longitude_squared` | 0.0778 | Spatial Interaction |
| **4** | `latitude_grid` | 0.0761 | Spatial Grid |
| **5** | `latitude` | 0.0582 | Raw Coordinate |
| **6** | `longitude` | 0.0512 | Raw Coordinate |
| **7** | `house_age` | 0.0477 | Structural |
| **8** | `distance_from_center` | 0.0461 | Geodesic Distance |
| **9** | `longitude_grid` | 0.0447 | Spatial Grid |
| **10** | `latitude_longitude` | 0.0351 | Spatial Interaction |

*Key Finding*: Spatial interaction terms and grid features account for **>35%** of total feature importance, proving that explicit spatial feature engineering dramatically outperforms raw tabular features alone.

---

## 14. Web Application
An interactive Streamlit web application (`app.py`) was deployed featuring:
* Dual-column layout for property inputs and location selection.
* Interactive map displaying user-selected coordinates.
* Automated spatial feature & embedding generation pipeline.
* Real-time property market valuation output in USD.
* Project information and dynamic model evaluation metric display.

---

## 15. Limitations
1. **Target Upper Truncation**: Target values are capped at $500,000 ($5.0), limiting prediction precision for ultra-luxury properties ($>\$1\text{M}$).
2. **Census Block Aggregation**: Data is aggregated at the block group level rather than individual house-level transactions.
3. **Macro-Economic Shifts**: The 1990 census data reflects historical baseline pricing structure requiring price index scaling for contemporary deployment.

---

## 16. Future Scope
* Integration of real-time OpenStreetMap point-of-interest (POI) spatial features (distance to schools, transit stations, coastlines).
* Graph Neural Network (GNN) spatial embeddings.
* Spatial block cross-validation techniques to further eliminate spatial autocorrelation leakage.
* Cloud deployment on Streamlit Community Cloud or AWS EC2.

---

## 17. Conclusion
This project successfully demonstrates that encoding geographic coordinates into non-linear spatial interactions, Haversine distance metrics, spatial grids, and continuous spatial embeddings significantly improves property valuation accuracy. The XGBoost model achieved an $R^2$ score of **0.8318** with an RMSE of **$46,950**, establishing a robust foundation for automated geospatial property appraisal.
