# Viva Voce Examination Questions & Answers
**Project Title**: Deep Learning-Based Flood Prediction Using Rainfall Data  
**Dataset**: `data/raw/flood_risk_dataset_india.csv` (10,000 spatial observations across India)

---

### Q1: What is the project about?
**Answer**: This project is an end-to-end machine learning and hydro-meteorological forecasting system that predicts regional flood risk (`FLOOD` vs `NO FLOOD`) across geographical sampling locations in India using precipitation, elevation, streamflow discharge, humidity, temperature, and soil permeability metrics.

---

### Q2: What problem does it solve?
**Answer**: Flooding is a severe natural hazard that causes widespread loss of life, displacement, and economic destruction. This system automates flood risk warning classification from hydro-meteorological parameters, enabling early municipal alerts and emergency management response.

---

### Q3: Why did you choose flood prediction?
**Answer**: Flood disaster management requires rapid, objective risk estimation. Machine learning provides automated non-linear pattern recognition across complex atmospheric and hydrological indicators to complement traditional physical hydraulic simulations.

---

### Q4: What dataset did you use?
**Answer**: We used `flood_risk_dataset_india.csv` containing 10,000 spatial observations across geographical coordinates in India (Latitude 8.0°–37.0°N, Longitude 68.0°–97.0°E).

---

### Q5: What are the input features?
**Answer**: The raw dataset contains 13 input features: `Rainfall (mm)`, `Temperature (°C)`, `Humidity (%)`, `River Discharge (m³/s)`, `Water Level (m)`, `Elevation (m)`, `Land Cover`, `Soil Type`, `Population Density`, `Infrastructure`, `Historical Floods`, `Latitude`, and `Longitude`.

---

### Q6: What is the target variable?
**Answer**: The target variable is `Flood_Occurred`, a binary flag where `1` represents a Flood event (5,057 records, 50.57%) and `0` represents No Flood (4,943 records, 49.43%).

---

### Q7: How did you clean the data?
**Answer**: We stripped whitespace from column headers and categorical string values, checked for missing values (0 missing found), checked for duplicate records (0 duplicates found), and validated physical ranges (Rainfall $\ge 0$, Humidity $0-100\%$, Elevation $\ge 0$).

---

### Q8: What is Exploratory Data Analysis (EDA)?
**Answer**: EDA is the systematic process of analyzing dataset distributions, summary statistics, class balance, outliers, and inter-feature correlations prior to model training to understand underlying patterns and data quality.

---

### Q9: What is feature engineering?
**Answer**: Feature engineering is the process of creating domain-specific mathematical indicators from raw data to improve model learning. We engineered 30 features, including `Rainfall_Elevation_Ratio`, `Hydro_Load_Index`, `Discharge_WaterLevel_Ratio`, `Relative_Humidity_Ratio`, `Soil_Permeability_Index`, and `Land_Cover_Inundation_Risk`.

---

### Q10: Why did you use Logistic Regression?
**Answer**: Logistic Regression provides a clean, interpretable linear probability baseline. It achieved the highest **Recall (0.5668)** and **F1-Score (0.5378)** on the holdout test dataset.

---

### Q11: Why did you use Decision Tree?
**Answer**: Decision Tree Classifier provides a non-linear rule-based decision tree baseline (`max_depth=10`). It helps evaluate how single feature splits perform before applying ensemble methods.

---

### Q12: Why did you use Random Forest?
**Answer**: Random Forest Classifier is an ensemble bagging model (`n_estimators=100`, `max_depth=15`) that reduces decision tree variance by averaging multiple de-correlated decision trees trained on bootstrap samples.

---

### Q13: Why did you use XGBoost?
**Answer**: XGBoost (eXtreme Gradient Boosting) is an advanced gradient boosting decision tree algorithm. It achieved the highest overall **Accuracy (0.5170)** and **Precision (0.5210)** while maintaining strong Recall (0.5519) and F1-Score (0.5360).

---

### Q14: Why did you evaluate an LSTM model?
**Answer**: Long Short-Term Memory (LSTM) networks are recurrent neural networks designed for sequential time-series forecasting. We audited the dataset schema to evaluate whether LSTM was applicable.

---

### Q15: What is a time series?
**Answer**: A time series is a sequence of data points indexed in continuous chronological order at uniform time intervals (e.g., hourly or daily streamflow history for a specific station).

---

### Q16: What is data leakage?
**Answer**: Data leakage occurs when information from outside the training dataset (such as test set statistics or target variable values) is accidentally used to create features or fit scalers, leading to unrealistically optimistic performance during training that fails in production.

---

### Q17: Why is accuracy alone not sufficient for flood prediction?
**Answer**: In flood warning systems, failing to alert an actual flood (False Negative) carries severe risks of unevacuated populations and loss of life. Accuracy measures overall correctness but obscures False Negative rates. Thus, **Recall** and **F1-score** are primary metrics.

---

### Q18: What are precision, recall, and F1-score?
**Answer**:
- **Precision**: $\frac{TP}{TP + FP}$ — Fraction of predicted flood alerts that were actual floods.
- **Recall**: $\frac{TP}{TP + FN}$ — Fraction of actual flood events correctly identified.
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean balancing precision and recall.

---

### Q19: Which model performed best and why?
**Answer**: **Logistic Regression** performed best for maximum sensitivity (**Recall = 0.5668**, **F1 = 0.5378**), minimizing missed flood warnings. **XGBoost** performed best for overall accuracy (**0.5170**) and precision (**0.5210**). Both were selected as production baselines.

---

### Q20: How does the prediction system work?
**Answer**: The user inputs 13 raw parameters via CLI (`src/predict.py`) or Streamlit web UI (`app.py`). The system validates inputs, constructs the 30 engineered features via `src/feature_engineering.py`, scales features using `models/scaler.pkl`, passes them to `models/xgboost.pkl`, and displays `FLOOD` or `NO FLOOD` with the exact flood probability percentage.

---

### Q21: How did you use Git?
**Answer**: Git was used for local version control. We initialized the repository (`git init`), set the default branch to `main`, configured `.gitignore`, and created 10 structured milestone commits representing each development phase.

---

### Q22: Why is GitHub useful?
**Answer**: GitHub provides remote repository hosting, code collaboration, issue tracking, continuous integration, and transparent open-source code sharing and portfolio demonstration.

---

### Q23: What is a Git commit?
**Answer**: A Git commit is a recorded snapshot of changes made to the repository files at a specific point in time, accompanied by a descriptive commit message.

---

### Q24: What is a Git branch?
**Answer**: A Git branch is an independent line of development that allows developers to work on features or bug fixes in isolation without affecting the main codebase.

---

### Q25: What are the limitations of the project?
**Answer**:
1. The dataset contains 10,000 spatial observations without continuous timestamp history, capping spatial classification bounds to ~51-54%.
2. Elevation and river discharge represent regional averages rather than micro-topography stream channels.

---

### Q26: What is the future scope?
**Answer**:
1. Ingesting continuous 15-minute or daily IMD/CWC gauge telemetry data to train valid **LSTM and Transformer** time-series forecasting models.
2. Integrating GPM IMERG satellite rainfall and Sentinel-1 SAR radar imagery for spatial inundation mapping.
