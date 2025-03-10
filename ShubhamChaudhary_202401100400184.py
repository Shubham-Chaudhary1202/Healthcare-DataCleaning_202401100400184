# # Step 1: Install necessary packages for visualizations (if not already installed)
# !pip install -U scikit-learn pandas numpy matplotlib seaborn missingno

# Step 2: Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

# # Step 3: Upload your dataset to Google Colab
# from google.colab import files
# uploaded = files.upload()

# After uploading, read the uploaded CSV
df = pd.read_csv("healthcare_data.csv")

# Step 4: Visualize missing data before cleaning
plt.figure(figsize=(12, 8))
msno.matrix(df)  # This visualizes missing data as a matrix
plt.title("Missing Data in Original Dataset", fontsize=15)
plt.show()

# Step 5: Handle Missing Data
# Fill missing numerical values with the median
numerical_cols = df.select_dtypes(include=[np.number]).columns
imputer_num = SimpleImputer(strategy='median')
df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

# Fill missing categorical values with the mode
categorical_cols = df.select_dtypes(include=[object]).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# Visualize missing data after cleaning
plt.figure(figsize=(12, 8))
msno.matrix(df)
plt.title("Missing Data After Cleaning", fontsize=15)
plt.show()

# Step 6: Handle Inconsistent Data
# Remove duplicates if any
df = df.drop_duplicates()

# Fix case inconsistency in categorical data
for col in categorical_cols:
    df[col] = df[col].str.lower()

# Step 7: Encode categorical variables using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 8: Handle Noisy Data (Outlier Detection)
# Use IsolationForest to detect and remove outliers from numerical columns
outlier_detector = IsolationForest(contamination=0.05)  # 5% of data as outliers
outliers = outlier_detector.fit_predict(df[numerical_cols])

# Only keep the rows that are not outliers (marked as 1, outliers are -1)
df_clean = df[outliers == 1]

# Step 9: Visualize Data Distribution Before and After Cleaning
# Plot distributions of a specific column before and after cleaning (for example, 'age')
# You can choose other columns based on your dataset

# Before cleaning (original data)
plt.figure(figsize=(10, 6))
sns.histplot(df[numerical_cols[0]], kde=True, color="red", label="Original Data")
plt.title("Distribution of 'Age' Before Cleaning")
plt.legend()
plt.show()

# After cleaning (cleaned data)
plt.figure(figsize=(10, 6))
sns.histplot(df_clean[numerical_cols[0]], kde=True, color="green", label="Cleaned Data")
plt.title("Distribution of 'Age' After Cleaning")
plt.legend()
plt.show()

# Step 10: Visualize the correlation matrix before and after cleaning
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Before Cleaning", fontsize=15)
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix After Cleaning", fontsize=15)
plt.show()

# Step 11: Save the cleaned data to a new CSV file
df_clean.to_csv("cleaned_healthcare_data.csv", index=False)

# To download the cleaned data from Colab
files.download("cleaned_healthcare_data.csv")
