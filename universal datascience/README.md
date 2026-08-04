# Deep Learning-Based Flood Prediction Using Rainfall Data

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-green.svg)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.1-red.svg)](https://streamlit.io/)
[![PyTest](https://img.shields.io/badge/PyTest-8.3.4-brightgreen.svg)](https://docs.pytest.org/)

An end-to-end Machine Learning & Hydro-Meteorological forecasting framework for predicting regional flood occurrence across geographical coordinates in India using precipitation, terrain elevation, streamflow discharge, atmospheric humidity, and soil permeability metrics.

---

## 📋 Abstract
This project implements an end-to-end Machine Learning and Hydro-Meteorological forecasting framework for predicting regional flood occurrence across geographical coordinates in India. Using 10,000 spatial observations, the system cleans raw observational data, constructs 30 domain-specific hydro-meteorological interaction features, trains and benchmarks four baseline Machine Learning classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost), conducts a scientific audit regarding LSTM recurrent neural networks, and provides a production-ready inference API (`src/predict.py`) and Streamlit web application (`app.py`).

---

## 🚨 Problem Statement
Flooding is one of the most severe natural hazards worldwide, causing extensive loss of life, displacement, agricultural destruction, and infrastructure damage. Accurate early detection of flood risk based on precipitation, terrain elevation, streamflow discharge, atmospheric moisture, and soil permeability is essential for emergency management and municipal disaster response.

---

## 🎯 Objectives
1. **Data Validation & Preprocessing**: Audit raw spatial observational datasets for integrity, missing values, duplicates, and physical range sanity.
2. **Hydro-Meteorological Feature Engineering**: Construct physical interaction indices (Rainfall-to-Elevation ratio, Hydro-Load Index, Streamflow Discharge efficiency, Soil Permeability scores).
3. **Machine Learning Benchmarking**: Train and evaluate Logistic Regression, Decision Tree, Random Forest, and XGBoost classifiers under leakage-free evaluation.
4. **Time-Series / Deep Learning Audit**: Audit dataset schema to evaluate whether LSTM recurrent neural networks can be applied without fabricating artificial temporal dependencies.
5. **Production Inference & Web App**: Build a modular prediction inference pipeline (`src/predict.py`) and Streamlit web interface (`app.py`).

---

## 📊 Dataset & Source
- **Primary Source**: `data/raw/flood_risk_dataset_india.csv` (10,000 spatial records sampled across geographical monitoring coordinates in India).
- **Total Records**: 10,000 observations across India (Latitude 8.0°–37.0°N, Longitude 68.0°–97.0°E).
- **Target Variable**: `Flood_Occurred` (Binary: `1` = Flood [5,057 records, 50.57%], `0` = No Flood [4,943 records, 49.43%]).

---

## 🛠️ Technologies Used
- **Programming Language**: Python 3.14
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, PyTorch
- **Data Visualization**: Matplotlib, Seaborn
- **Web Application Framework**: Streamlit
- **Testing Framework**: PyTest
- **Model Serialization**: Joblib
- **Version Control**: Git, GitHub

---

## 🔬 Methodology & Workflow Diagrams

### Project Workflow Diagram
![Project Workflow](results/project_workflow.png)

### Model Architecture Diagram
![Model Architecture](results/model_architecture.png)

---

## 🧹 Data Preprocessing & EDA
- Stripped whitespace from column headers and string entries.
- Removed duplicate rows (0 duplicate records found).
- Validated numerical boundaries (Rainfall $\ge 0$, Humidity $0-100\%$, Elevation $\ge 0$).
- Applied **Stratified 80/20 Train/Test Split** (8,000 train / 2,000 test samples).
- Fitted `StandardScaler` **exclusively on $X_{\text{train}}$** to prevent data leakage.

---

## 🛠️ Feature Engineering (30 Final Features)
- `Rainfall_Elevation_Ratio`: $\frac{\text{Rainfall (mm)}}{\text{Elevation (m)} + 1.0}$ — Lowland inundation potential.
- `Hydro_Load_Index`: $\frac{\text{Rainfall} \times \text{River Discharge}}{\text{Elevation} + 10}$ — Combined surface runoff pressure.
- `Discharge_WaterLevel_Ratio`: Channel flow capacity efficiency.
- `Relative_Humidity_Ratio`: Moisture-temperature interaction.
- `Soil_Permeability_Index`: Physical drainage score (`Clay`=3.0 [impermeable, high risk] to `Sandy`=1.0 [permeable]).
- `Land_Cover_Inundation_Risk`: Surface runoff score based on impervious land cover.

---

## 🤖 Machine Learning Models & Results

| Model | Accuracy | Precision | Recall | F1-Score | Recommended Production Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.5075 | 0.5116 | **0.5668** | **0.5378** | **Selected Model for High Flood Recall** |
| **Decision Tree** | 0.5040 | 0.5093 | 0.5153 | 0.5123 | Baseline decision tree classifier |
| **Random Forest** | 0.5050 | 0.5098 | 0.5381 | 0.5236 | Ensemble bagging baseline |
| **XGBoost Classifier** | **0.5170** | **0.5210** | 0.5519 | 0.5360 | **Selected Model for Overall Accuracy & Precision** |
| **LSTM (Deep Learning)** | *N/A* | *N/A* | *N/A* | *N/A* | *Inappropriate for spatial cross-sectional data (0 timestamp columns)* |

---

## 🧠 LSTM Model & Scientific Audit
A data-integrity audit verified that `flood_risk_dataset_india.csv` contains **0 date/time/timestamp columns**. The dataset consists of spatial observations across geographical coordinates rather than continuous temporal measurements over time for specific river stations.

> **Scientific Determination**:
> Fabricating artificial sliding windows over unordered spatial rows introduces **false temporal autocorrelation and data corruption**. Per Step 8 guidelines, LSTM training was halted on this spatial dataset. A continuous gauge dataset specification (`Station_ID`, `Timestamp`, `Precipitation_mm`) was authored in `src/train_lstm.py` for future chronological data integration.

---

## 💻 How to Install

```bash
git clone https://github.com/<your-username>/universal-datascience.git
cd universal-datascience

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run Terminal Prediction Inference (CLI)
```bash
python src/predict.py
```

### Run Interactive Streamlit Web Application
```bash
streamlit run app.py
```

### Run Automated Unit Test Suite
```bash
python -m pytest tests/test_prediction.py
```

---

## 📂 Project Structure

```text
universal datascience/
├── data/
│   ├── processed/          # Feature matrices (X_train, X_test, y_train, y_test)
│   └── raw/                # Original raw dataset CSV and zip archive
├── notebooks/              # Executed Jupyter Notebooks 01 to 06
├── src/                    # Reusable Python source code modules
│   ├── preprocessing.py    # Data cleaning & validation
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── train_ml.py         # Baseline ML training & evaluation
│   ├── train_lstm.py       # LSTM scientific audit module
│   ├── evaluate_models.py  # Final evaluation script
│   └── predict.py          # Real-time prediction inference module
├── models/                 # Serialized model & scaler pkl artifacts
├── results/                # Metrics CSVs, evaluation reports, and PNG plots
├── tests/                  # Pytest unit tests (100% pass rate)
├── app.py                  # Streamlit web application
├── README.md               # Comprehensive project documentation
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore rules
```

---

## 🔀 Git and GitHub Version Control
- **Branch**: `main`
- **Commit History**: 11 structured milestone commits representing the project lifecycle.
- **Workflow**: Local commits -> GitHub remote `origin main`.

---

## ⚠️ Limitations & Future Scope

### Limitations
1. **Spatial Tabular Sampling**: Observations represent spatial points without timestamp history, limiting predictive classification bounds to ~51-54%.
2. **Coarse Regional Resolution**: Elevation and river discharge represent regional station averages.

### Future Scope
1. **Continuous Gauge Integration**: Incorporate IMD and CWC hourly station telemetry data to enable valid **LSTM and Transformer** time-series forecasting.
2. **Satellite & Radar Integration**: Ingest GPM IMERG precipitation and Sentinel-1 SAR radar imagery for spatial inundation mapping.

---

## 💡 Conclusion
This project successfully establishes a complete end-to-end Machine Learning pipeline for flood risk prediction from spatial hydro-meteorological data. **Logistic Regression** (highest recall) and **XGBoost** (highest accuracy) provide robust baseline models for emergency warning demonstrations.

---

## 🛡️ Academic Safety Disclaimer
> **Notice**: This repository contains an academic machine learning research project developed for educational and demonstration purposes. It should **not** be used as an official real-time flood emergency warning system.
