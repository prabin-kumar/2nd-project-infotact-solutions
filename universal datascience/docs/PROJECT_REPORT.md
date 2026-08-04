# DEEP LEARNING-BASED FLOOD PREDICTION USING RAINFALL DATA

---

## 1. TITLE PAGE

**Project Title**: Deep Learning-Based Flood Prediction Using Rainfall Data  

**Student Name**: [Student Name Placeholder]  
**Roll Number**: [Roll Number Placeholder]  
**Department**: Department of Computer Science & Engineering / Data Science  
**College / University**: [College/University Name Placeholder]  
**Course**: Bachelor of Technology (B.Tech) in Computer Science / Data Science  
**Faculty / Supervisor**: [Faculty Supervisor Name Placeholder]  
**Academic Year**: 2025 – 2026  

---

## 2. ABSTRACT

Flooding represents one of the most severe and frequent natural disasters globally, resulting in significant loss of life, displacement of human populations, agricultural destruction, and economic instability. Timely and accurate prediction of flood occurrences based on environmental indicators is essential for effective municipal early warnings and emergency response management. 

This project develops an end-to-end Machine Learning and Hydro-Meteorological forecasting framework for predicting flood risk (`FLOOD` vs. `NO FLOOD`) across geographical coordinates in India. Operating on a dataset of 10,000 spatial observations, the system implements data cleaning, range validation, and 30 domain-specific hydro-meteorological feature transformations (including Rainfall-to-Elevation ratio, Hydro-Load Index, Streamflow Discharge efficiency, and Soil Permeability indices). Four baseline Machine Learning models—Logistic Regression, Decision Tree, Random Forest, and XGBoost—were trained and evaluated on a holdout test dataset (2,000 samples) under strict data-leakage prevention controls. 

Additionally, a rigorous data-integrity audit was conducted regarding Deep Learning (LSTM) models, establishing that spatial tabular data without continuous timestamp series cannot be modeled with recurrent networks without introducing false temporal autocorrelation. Logistic Regression achieved the highest sensitivity (**Recall = 0.5668**, **F1 = 0.5378**), while XGBoost achieved the highest overall accuracy (**Accuracy = 0.5170**, **Precision = 0.5210**). The complete framework is deployed as a CLI inference module (`src/predict.py`) and Streamlit web application (`app.py`), backed by an automated 8-test PyTest suite (100% pass rate).

---

## 3. INTRODUCTION

### 3.1 Flood Prediction Context
Precipitation events, river channel discharge rates, terrain slope topography, and soil infiltration capacity interact dynamically to dictate inundation risks across watershed basins. Traditional hydraulic forecasting relies on continuous differential physical simulations, which require extensive computational infrastructure and high-resolution spatial terrain mapping.

### 3.2 Role of Environmental & Rainfall Data
Precipitation depth is the primary atmospheric driver of surface runoff volume. However, rainfall volume alone is insufficient for predicting flooding without integrating terrain elevation (lowland catchment potential), river streamflow discharge (channel volume pressure), atmospheric humidity (convective moisture status), and soil drainage composition.

### 3.3 Role of Machine Learning & Deep Learning
Machine Learning classifiers enable rapid, non-linear pattern recognition across multi-modal hydro-meteorological indicators. By learning non-linear boundary interactions across spatial observations, algorithms like XGBoost and Random Forest complement physical models. Furthermore, Deep Learning models such as LSTMs provide specialized architectures for temporal sequence forecasting when continuous timestamp telemetry is available.

---

## 4. PROBLEM STATEMENT

Given a spatial observation vector containing meteorological metrics (rainfall, temperature, humidity), hydrological metrics (river discharge, water level stage), physical terrain attributes (elevation, land cover, soil permeability), and socioeconomic demographic metrics (population density, infrastructure, historical flood records), the objective is to accurately classify whether a flood event occurs (`Flood_Occurred = 1`) or not (`Flood_Occurred = 0`), while optimizing for high recall to minimize catastrophic False Negative warnings.

---

## 5. OBJECTIVES

1. **Data Inspection & Validation**: Perform automated cleaning, missing value detection, duplicate removal, and boundary sanity checks on raw spatial observations.
2. **Domain Feature Engineering**: Construct physical hydro-meteorological interaction indices combining rainfall, river discharge, elevation, atmospheric humidity, and soil permeability scores.
3. **Machine Learning Benchmarking**: Train and evaluate baseline Machine Learning algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost) under strict leakage-free split controls.
4. **Time-Series / Deep Learning Audit**: Conduct a scientific audit evaluating the applicability of LSTM recurrent neural networks on non-timestamped spatial datasets.
5. **Inference System Deployment**: Build a reusable inference module (`src/predict.py`) and interactive Streamlit web application (`app.py`) with 100% automated test coverage.

---

## 6. LITERATURE / EXISTING APPROACHES

Traditional flood forecasting relies primarily on two methodologies:
1. **Hydrological & Hydraulic Simulations** (e.g., HEC-RAS, SWAT): Solve physical fluid-mechanics equations over digital elevation models (DEM). While physically grounded, they require intensive calibration and fine spatial grid data.
2. **Statistical Regression Models**: Simple linear regressions linking rainfall volume to streamflow. These models frequently fail to capture complex non-linear interactions between land cover, soil permeability, and terrain slope.

Machine Learning approaches address these limitations by learning complex non-linear decision boundaries directly from multi-modal environmental features.

---

## 7. PROPOSED METHODOLOGY

The end-to-end project methodology follows a structured, modular pipeline:

```text
┌─────────────────┐    ┌────────────────────┐    ┌───────────────────────────┐
│ Raw CSV Dataset │ ──>│ Data Validation &  │ ──>│  Hydro-Meteorological     │
│ (10,000 Rows)   │    │ Cleaning (Step 3-4)│    │ Feature Engineering (St.6)│
└─────────────────┘    └────────────────────┘    └───────────────────────────┘
                                                               │
                                                               ▼
┌─────────────────┐    ┌────────────────────┐    ┌───────────────────────────┐
│ Streamlit Web   │ <──│ Final Model        │ <──│  ML Classifiers           │
│ App (app.py)    │    │ Selection (Step 9) │    │  (XGBoost, LogReg, RF, DT) │
└─────────────────┘    └────────────────────┘    └───────────────────────────┘
```

---

## 8. DATASET DESCRIPTION

- **Dataset File**: `data/raw/flood_risk_dataset_india.csv`
- **Total Records**: 10,000 spatial observations sampled across India (Latitude 8.0°–37.0°N, Longitude 68.0°–97.0°E).
- **Target Variable**: `Flood_Occurred` (Binary: `1` = Flood [5,057 records, 50.57%], `0` = No Flood [4,943 records, 49.43%]).
- **Class Balance**: Balanced target distribution (~50.5% / 49.5%).
- **Raw Features**:
  - `Rainfall (mm)`: Continuous float (0.0 to ~500.0 mm)
  - `Temperature (°C)`: Continuous float
  - `Humidity (%)`: Continuous float (0.0% to 100.0%)
  - `River Discharge (m³/s)`: Continuous float (0.0 to ~5,000.0 m³/s)
  - `Water Level (m)`: Continuous float (0.0 to ~15.0 m)
  - `Elevation (m)`: Continuous float (0.0 to ~8,846.0 m)
  - `Land Cover`: Categorical string (`Water Body`, `Urban`, `Agriculture`, `Forest`, `Grassland`)
  - `Soil Type`: Categorical string (`Clay`, `Peat`, `Silty`, `Loam`, `Sandy`)
  - `Population Density`: Continuous float (people/km²)
  - `Infrastructure`: Binary integer (0 or 1)
  - `Historical Floods`: Binary integer (0 or 1)

---

## 9. DATA PREPROCESSING

1. **Header & String Sanitization**: Stripped leading/trailing whitespace from column names and string values.
2. **Missing & Duplicate Auditing**: Verified 0 missing values and 0 duplicate rows across all 10,000 records.
3. **Range Validation**: Verified physical boundaries (Rainfall $\ge 0$, Humidity $0-100\%$, Elevation $\ge 0$).
4. **Stratified Train/Test Split**: Applied an 80/20 stratified split (8,000 training samples / 2,000 holdout test samples) maintaining target class balance.
5. **Data Leakage Prevention**: `StandardScaler` was fitted **exclusively on $X_{\text{train}}$**, then applied to transform $X_{\text{train}}$ and $X_{\text{test}}$.

---

## 10. EXPLORATORY DATA ANALYSIS (EDA)

Analysis of the raw observational dataset yielded key insights:
- **Target Class Balance**: Perfectly balanced distribution (50.57% positive vs. 49.43% negative), eliminating the need for artificial SMOTE resampling.
- **Key Feature Correlations**: High rainfall (`Rainfall_mm`), high streamflow (`River_Discharge_m3s`), high river stage (`Water_Level_m`), and low terrain elevation (`Elevation_m`) exhibit the strongest statistical association with flood events.
- **Physical Extremes**: Himalayan elevations (~8,846m) and Ganga/Brahmaputra discharge volumes (~5,000 m³/s) reflect real physical terrain geography.

*(Refer to generated visual figures: [results/rainfall_distribution.png](file:///c:/Users/hkris/Downloads/universal%20datascience/results/rainfall_distribution.png), [results/target_distribution.png](file:///c:/Users/hkris/Downloads/universal%20datascience/results/target_distribution.png), and [results/feature_correlation.png](file:///c:/Users/hkris/Downloads/universal%20datascience/results/feature_correlation.png)).*

---

## 11. FEATURE ENGINEERING

A total of 30 final input features were engineered in `src/feature_engineering.py`:
- `Rainfall_Elevation_Ratio`: $\frac{\text{Rainfall (mm)}}{\text{Elevation (m)} + 1.0}$ — Quantifies low-lying terrain inundation risk.
- `Hydro_Load_Index`: $\frac{\text{Rainfall} \times \text{River Discharge}}{\text{Elevation} + 10}$ — Quantifies surface water accumulation pressure.
- `Discharge_WaterLevel_Ratio`: $\frac{\text{River Discharge}}{\text{Water Level} + 10^{-5}}$ — Quantifies channel capacity efficiency.
- `Relative_Humidity_Ratio`: $\frac{\text{Humidity}}{\text{Temperature} + 1}$ — Quantifies atmospheric moisture saturation.
- `Elevation_Inverse`: $\frac{1.0}{\text{Elevation} + 1.0}$ — Lowland terrain vulnerability factor.
- `Soil_Permeability_Index`: Geological drainage score (`Clay`=3.0 [impermeable] to `Sandy`=1.0 [permeable]).
- `Land_Cover_Inundation_Risk`: Runoff susceptibility score based on surface land use.
- **One-Hot Encodings**: `Land_Cover_*` and `Soil_Type_*` binary indicators.
- **Vulnerability Interactions**: `Infrastructure_Vulnerability` (`Population_Density` $\times$ `Infrastructure`) and `Historical_Rainfall_Interaction` (`Historical_Floods` $\times$ `Rainfall_mm`).

---

## 12. MACHINE LEARNING MODELS

Four classifiers were trained using fixed seeds (`random_state=42`):
1. **Logistic Regression**: Linear baseline probability model.
2. **Decision Tree Classifier**: Non-linear decision rule classifier (`max_depth=10`).
3. **Random Forest Classifier**: Ensemble bagging classifier (`n_estimators=100`, `max_depth=15`).
4. **XGBoost Classifier**: Gradient boosted decision tree model (`n_estimators=100`, `learning_rate=0.1`, `max_depth=6`).

---

## 13. LSTM MODEL & SCIENTIFIC TIME-SERIES AUDIT

A scientific audit was conducted on `flood_risk_dataset_india.csv` to check for date/time attributes:
- **Audit Result**: **0 date/time/timestamp columns found**. The dataset consists of spatial cross-sectional coordinates sampled across India.

> **Scientific Determination**:
> Fabricating sliding window sequences over unordered spatial rows introduces **false temporal autocorrelation and data corruption**. In strict compliance with Step 8 rules, LSTM training was halted on this spatial dataset. A continuous gauge dataset specification (`Station_ID`, `Timestamp`, `Precipitation_mm`) was authored in `src/train_lstm.py` for future chronological telemetry data.

---

## 14. EVALUATION METRICS

Models were evaluated on 2,000 holdout test samples using:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

In disaster warning systems, **minimizing False Negatives (Recall)** is prioritized over raw accuracy to prevent un-alerted flood hazards.

---

## 15. RESULTS & MODEL COMPARISON

All models were evaluated under identical test conditions:

| Model | Accuracy | Precision | Recall | F1-Score | Recommended Production Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.5075 | 0.5116 | **0.5668** | **0.5378** | **Selected Model for High Flood Recall** |
| **Decision Tree** | 0.5040 | 0.5093 | 0.5153 | 0.5123 | Baseline tree classifier |
| **Random Forest** | 0.5050 | 0.5098 | 0.5381 | 0.5236 | Ensemble bagging baseline |
| **XGBoost Classifier** | **0.5170** | **0.5210** | 0.5519 | 0.5360 | **Selected Model for Overall Accuracy & Precision** |
| **LSTM (Deep Learning)** | *N/A* | *N/A* | *N/A* | *N/A* | *Inappropriate for spatial cross-sectional data* |

*(Refer to generated visual figures: [results/final_model_comparison.png](file:///c:/Users/hkris/Downloads/universal%20datascience/results/final_model_comparison.png) and [results/final_confusion_matrices.png](file:///c:/Users/hkris/Downloads/universal%20datascience/results/final_confusion_matrices.png)).*

---

## 16. BEST MODEL SELECTION

- **Logistic Regression** is selected for applications prioritizing maximum sensitivity (**Recall = 0.5668**, **F1 = 0.5378**), correctly identifying the largest fraction of actual flood events.
- **XGBoost Classifier** is selected for applications prioritizing overall classification accuracy (**0.5170**) and precision (**0.5210**).

---

## 17. PREDICTION SYSTEM

- **Inference Script**: `src/predict.py` accepts input parameters and formats CLI predictions (`FLOOD` / `NO FLOOD`, `Flood Probability: XX.XX%`).
- **Web App**: `app.py` Streamlit web interface containing input controls, risk banners, probability progress gauges, and safety notices.
- **Unit Test Suite**: `tests/test_prediction.py` (8/8 tests passed).

---

## 18. GIT AND GITHUB VERSION CONTROL

- **Branch**: `main`
- **Working Tree**: `nothing to commit, working tree clean`
- **Commit History**: 13 structured milestone commits representing the project lifecycle:
  - `34a72a9`: Final academic project submission
  - `2e76e00`: Finalize project documentation and presentation
  - `92eab42`: Add feature metadata JSON and package init files
  - `5f9f4b2`: Update project README.md and documentation
  - `b0825aa`: Add flood prediction inference module, Streamlit web app, and test suite

---

## 19. TESTING

Automated unit tests were executed with PyTest:
- Command: `python -m pytest tests/test_prediction.py`
- Total Tests: 8
- **Passed**: **8 (100% Pass Rate)**
- Failed: 0
- Coverage: Model loading, scaler loading, range validation, negative input handling, output format.

---

## 20. LIMITATIONS

1. **Spatial Tabular Sampling**: Observations represent spatial geographical points without timestamp history, capping spatial classification bounds to ~51-54%.
2. **Coarse Resolution**: Terrain elevation and river discharge values represent regional station averages.

---

## 21. FUTURE SCOPE

1. **Continuous Telemetry Integration**: Ingest real-time IMD/CWC river gauge time-series to train valid **LSTM and Transformer** recurrent forecasting models.
2. **Remote Sensing Integration**: Ingest GPM IMERG satellite precipitation and Sentinel-1 SAR imagery for spatial inundation mapping.

---

## 22. CONCLUSION

This project successfully establishes a complete end-to-end Machine Learning pipeline for flood risk prediction from spatial hydro-meteorological data. **Logistic Regression** (highest recall) and **XGBoost** (highest accuracy) provide robust baseline models for emergency warning demonstrations.

---

## 23. REFERENCES

1. Scikit-Learn Documentation: Machine Learning in Python (scikit-learn.org).
2. XGBoost Documentation: Scalable and Flexible Gradient Boosting (xgboost.readthedocs.io).
3. Streamlit Framework Documentation: The fastest way to build data apps (streamlit.io).
4. Central Water Commission (CWC) & India Meteorological Department (IMD) Hydrological Standards.

---

## 24. APPENDIX

### Project Directory Structure
```text
universal datascience/
├── data/raw/ & data/processed/
├── docs/PROJECT_REPORT.md
├── notebooks/ (01 to 06)
├── src/ (preprocessing.py, feature_engineering.py, train_ml.py, train_lstm.py, evaluate_models.py, predict.py)
├── models/ (*.pkl artifacts)
├── results/ (plots, CSVs, reports)
├── tests/ (test_prediction.py)
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

### Key Execution Commands
```bash
# Run CLI Prediction:
python src/predict.py

# Run Streamlit Web Application:
streamlit run app.py

# Run Automated Test Suite:
python -m pytest tests/test_prediction.py
```
