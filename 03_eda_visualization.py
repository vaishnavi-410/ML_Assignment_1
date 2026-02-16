import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# ================================
# 1. Histogram - Age Distribution
# ================================
plt.figure()
plt.hist(df['Age'], bins=10)
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

# ================================
# 2. Histogram - Annual Income
# ================================
plt.figure()
plt.hist(df['Annual Income (k$)'], bins=10)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.show()

# ================================
# 3. Boxplot - Spending Score
# ================================
plt.figure()
plt.boxplot(df['Spending Score (1-100)'])
plt.title("Boxplot of Spending Score")
plt.ylabel("Spending Score")
plt.show()

# ================================
# 4. Gender Count Plot
# ================================
plt.figure()
sns.countplot(x='Gender', data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ================================
# 5. Correlation Heatmap
# ================================
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
