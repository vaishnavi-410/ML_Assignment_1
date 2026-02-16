import pandas as pd
import os

# Print current working directory (for understanding)
print("Current Working Directory:", os.getcwd())

# Correct file path
file_path = "data/Mall_Customers.csv"

# Load dataset
df = pd.read_csv(file_path)

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Shape of dataset
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Data types
print("\nData Types:")
print(df.dtypes)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
