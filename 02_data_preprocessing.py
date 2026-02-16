import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")


print("Dataset Loaded Successfully")
# Check missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Encode Gender column
label_encoder = LabelEncoder()
df['Genre'] = label_encoder.fit_transform(df['Genre'])


print("\nEncoded Genre Column:")
print(df['Genre'].head())

# Select numerical columns
numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Standardize features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\nScaled Numerical Features:")
print(df.head())
