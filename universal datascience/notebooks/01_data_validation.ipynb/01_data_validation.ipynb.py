import pandas as pd

# Load the CSV dataset
# pyrefly: ignore [parse-error]
df = pd.read_csv("C:\Users\hkris\Downloads\universal datascience\data\raw\flood_prediction_dataset.csv")

# Display the first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display column names
print(df.columns)

# Display number of rows and columns
print(df.shape)