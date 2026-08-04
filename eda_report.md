# Exploratory Data Analysis & Data Cleaning Report

## Executive Summary
This report summarizes the data quality checks, data cleaning procedures, exploratory data analysis (EDA), and geospatial findings for the **California Housing dataset**, prepared for the **Construction & Real Estate – Geospatial Valuation via Spatial Embeddings** project.

---

## 1. Data Quality & Cleaning Summary

* **Raw Dataset Location**: `data/raw/california_housing.csv`
* **Cleaned Dataset Location**: `data/processed/cleaned_housing.csv`
* **Dataset Shape**: 20,640 rows, 9 columns
* **Missing Values**: 0 across all features.
* **Duplicate Records**: 0 duplicates found.
* **Infinite Values**: 0 infinite values found.
* **Data Types**: All 9 variables are continuous numerical features (`float64`).
* **Cleaning Action Taken**: Confirmed zero nulls, zero duplicates, and valid numeric types across all features. Retained all 20,640 records and 9 columns without dropping geospatial coordinates (`latitude` and `longitude`), preserving spatial fidelity.

---

## 2. Main EDA Findings

### A. Important Patterns Found
1. **Target Feature (`house_value`) Distribution**:
   - The median house value ranges from $0.15 ($15,000) to $5.00 ($500,000).
   - The target distribution exhibits a right-skewed pattern with a prominent spike at $5.00 ($500,000), indicating upper truncation/censoring in the dataset collection methodology.

2. **Median Income Relationship**:
   - `median_income` exhibits the strongest linear correlation with `house_value` ($r \approx 0.69$). As block group median income rises, property valuation increases proportionally.

3. **House Age Impact**:
   - `house_age` displays a slight positive correlation with `house_value` ($r \approx 0.11$). High-value properties exist across both older historical block groups (e.g., historical coastal properties) and newer developments.

### B. Correlation Analysis with `house_value`
From the correlation matrix:
* `median_income`: **+0.69** (Strong positive correlation)
* `latitude`: **-0.14** (Negative correlation; southern coastal areas tend to have higher valuations than northern inland areas)
* `house_age`: **+0.11** (Slight positive correlation)
* `average_rooms`: **+0.15** (Moderate positive correlation)
* `longitude`: **-0.05** (Slight negative correlation)
* `population`: **-0.02** (Negligible linear correlation)
* `average_occupancy`: **-0.02** (Negligible linear correlation)
* `average_bedrooms`: **-0.05** (Slight negative correlation due to colinearity with rooms)

### C. Relationship Between Location and House Value
The geographic scatter plot (`longitude` vs. `latitude`) reveals distinct spatial clusters:
1. **Coastal Premium**: Properties situated directly along the Pacific coastline (San Francisco Bay Area, Los Angeles, San Diego) command significantly higher values ($\ge \$400,000 - \$500,000$).
2. **Inland Valuation Drop**: Moving east into the Central Valley (Sacramento, Fresno, Bakersfield), property values drop dramatically even for blocks with similar room counts or structural attributes.
3. **Metropolitan Clusters**: Two massive spatial price high-density nodes are visible centered around the Greater Bay Area ($37.7^\circ \text{N}, -122.4^\circ \text{W}$) and Greater Los Angeles ($34.0^\circ \text{N}, -118.2^\circ \text{W}$).

### D. Unusual & Outlier Observations
* **Capped Target Variable**: `house_value` is capped at `5.00001` ($500,000), affecting ~965 block groups (4.68% of the dataset). This artificial ceiling creates horizontal bands in scatter plots.
* **Extreme Structural Outliers**: `average_rooms` reaches up to 141.9 rooms per household, and `average_occupancy` reaches up to 1,243 members per household. These represent extreme non-residential or institutional block groups that will benefit from robust scaling or spatial embedding aggregation.

---

## 3. Why Geographical Features Improve Property Valuation

1. **Capturing Spatial Micro-Markets**: Traditional tabular models treat attributes independently, ignoring proximity. Coordinates reveal spatial sub-markets that explain why identical physical houses differ in value by hundreds of thousands of dollars depending on location.
2. **Non-Linear Coastal & Urban Distance Effects**: Latitude and longitude encode multi-dimensional spatial proximity to coastline, employment hubs, and desirable micro-climates that linear tabular models miss.
3. **Basis for Spatial Embeddings**: Raw coordinates enable spatial tiling (e.g., Uber H3 hexagon indexing) and continuous neural spatial embeddings (e.g., Spatial2Vec / Tile2Vec). These embeddings capture high-order spatial autocorrelation ($Spatial\ Context$) to boost automated valuation model (AVM) accuracy.

---

## 4. Generated Visualization Artifacts

All plot figures are stored in `reports/figures/`:

1. `reports/figures/distribution_house_values.png`
2. `reports/figures/house_value_vs_median_income.png`
3. `reports/figures/house_value_vs_house_age.png`
4. `reports/figures/house_value_vs_average_rooms.png`
5. `reports/figures/house_value_vs_latitude.png`
6. `reports/figures/house_value_vs_longitude.png`
7. `reports/figures/correlation_heatmap.png`
8. `reports/figures/geographic_scatter_plot.png`
