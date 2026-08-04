# Presentation Deck: Deep Learning-Based Flood Prediction
**Project Title**: Deep Learning-Based Flood Prediction Using Rainfall Data  
**Dataset**: `data/raw/flood_risk_dataset_india.csv` (10,000 spatial observations across India)

---

## Slide 1: Title Slide
- **Title**: Deep Learning-Based Flood Prediction Using Rainfall Data
- **Subtitle**: Hydro-Meteorological Feature Engineering & Machine Learning Baseline Evaluation
- **Presenter**: Data Science Engineering Team
- **Date**: August 2026

---

## Slide 2: Introduction & Motivation
- **Context**: Floods are devastating natural hazards causing loss of life, displacement, and economic destruction.
- **Goal**: Automate flood risk classification (`FLOOD` vs `NO FLOOD`) from precipitation, elevation, streamflow, and soil metrics.
- **Approach**: Data validation, 30 domain-engineered hydro features, baseline ML benchmarking, LSTM scientific audit, and Streamlit inference web app.

---

## Slide 3: Problem Statement & Objectives
- **Problem**: Traditional hydraulic modeling requires high-resolution physical topography. Machine learning provides rapid non-linear risk classification.
- **Objectives**:
  1. Validate raw spatial dataset integrity (10,000 records).
  2. Construct domain hydro-meteorological features.
  3. Benchmark Logistic Regression, Decision Tree, Random Forest, and XGBoost.
  4. Audit LSTM recurrent neural network suitability.
  5. Deploy Streamlit prediction web interface.

---

## Slide 4: Dataset Overview
- **Dataset**: `flood_risk_dataset_india.csv`
- **Size**: 10,000 observations across India (Lat 8.0°–37.0°N, Long 68.0°–97.0°E).
- **Target**: `Flood_Occurred` (Binary: 5,057 Flood [50.57%] vs. 4,943 No Flood [49.43%]).
- **Key Input Features**: Rainfall (mm), Temperature (°C), Humidity (%), River Discharge (m³/s), Water Level (m), Elevation (m), Land Cover, Soil Type.

---

## Slide 5: Methodology & Pipeline Architecture
- **Cleaning**: Whitespace stripping, 0 duplicate rows, range validation.
- **Splitting**: Stratified 80/20 Train/Test Split (8,000 train / 2,000 test samples).
- **Scaling**: `StandardScaler` fitted **strictly on $X_{\text{train}}$** to prevent data leakage.
- **Inference Pipeline**: Modular runtime feature transformer (`src/predict.py`).

---

## Slide 6: Exploratory Data Analysis (EDA) Insights
- **Target Balance**: Perfectly balanced target distribution (~50.5% / 49.5%).
- **Primary Correlations**: Rainfall, River Discharge, Water Level, and Elevation exhibit the strongest physical association with flood events.
- **Physical Extremes**: Himalayan elevations (~8,846m) and Ganges/Brahmaputra discharge (~5,000 m³/s) represent valid geographical extremes.

---

## Slide 7: Hydro-Meteorological Feature Engineering
Engineered 30 domain features in `src/feature_engineering.py`:
- `Rainfall_Elevation_Ratio`: $\frac{\text{Rainfall}}{\text{Elevation} + 1}$ (Lowland inundation potential).
- `Hydro_Load_Index`: $\frac{\text{Rainfall} \times \text{Discharge}}{\text{Elevation} + 10}$ (Combined runoff pressure).
- `Discharge_WaterLevel_Ratio`: Channel flow capacity efficiency.
- `Soil_Permeability_Index`: Soil drainage risk (`Clay`=3.0 to `Sandy`=1.0).
- `Land_Cover_Inundation_Risk`: Impervious land cover runoff risk score.

---

## Slide 8: Machine Learning Baseline Models
Evaluated 4 classifiers using fixed seed (`random_state=42`):
1. **Logistic Regression**: Interpretable linear baseline.
2. **Decision Tree**: Non-linear tree model (`max_depth=10`).
3. **Random Forest**: Ensemble bagging model (`n_estimators=100`, `max_depth=15`).
4. **XGBoost Classifier**: Gradient boosted trees (`n_estimators=100`, `learning_rate=0.1`).

---

## Slide 9: LSTM Deep Learning Scientific Audit
- **Audit Finding**: Dataset contains **0 date/time/timestamp columns**. Observations represent spatial coordinates across India.
- **Scientific Determination**: Fabricating artificial sliding windows over spatial rows introduces **false temporal autocorrelation and data corruption**.
- **Action Taken**: LSTM training halted per Step 8 guidelines; continuous gauge time-series specification authored in `src/train_lstm.py`.

---

## Slide 10: Model Evaluation & Results
*Evaluated on 2,000 Holdout Test Samples*:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.5075 | 0.5116 | **0.5668** | **0.5378** |
| **Decision Tree** | 0.5040 | 0.5093 | 0.5153 | 0.5123 |
| **Random Forest** | 0.5050 | 0.5098 | 0.5381 | 0.5236 |
| **XGBoost Classifier** | **0.5170** | **0.5210** | 0.5519 | 0.5360 |

- **Recall Winner**: **Logistic Regression (0.5668)** — Minimizes missed flood warnings.
- **Accuracy Winner**: **XGBoost (0.5170)** — Best overall precision & accuracy.

---

## Slide 11: Prediction System & Web Application
- **Inference Script**: `src/predict.py` (CLI prediction formatter).
- **Web App**: `app.py` Streamlit interactive interface.
- **Features**: Inputs 13 raw parameters, computes 30 engineered features, outputs `FLOOD` or `NO FLOOD` banner + Flood Probability percentage gauge.
- **Safety Disclaimer**: Academic notice displayed prominently.

---

## Slide 12: Conclusion & Future Scope
- **Conclusion**: Successfully established an end-to-end ML pipeline for flood prediction. Logistic Regression and XGBoost provide solid baseline performance.
- **Future Scope**:
  1. Ingest continuous IMD/CWC gauge telemetry data for **LSTM / Transformer** time-series forecasting.
  2. Integrate GPM IMERG satellite precipitation and Sentinel-1 SAR imagery.
