# Let's first import the dataset 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Use the exact file name and path
data = pd.read_csv("./dataset.csv")
data.head()
#EDA and data preprocessing
#Label encoding
from sklearn.preprocessing import LabelEncoder

label_cols = [col for col in data.columns if col != 'Weekly_Study_Hours']

# Apply Label Encoding to each of them
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])
data.head()
# b.Summary statistics
print("\nSummary Statistics:")
print(data.describe(include='all'))

# c. Check for missing values
print("\nMissing values:")
print(data.isnull().sum())



data=data.drop(columns=["Student_ID","Student_Age","Sex"])

cat_cols = data.select_dtypes(include='object').columns.tolist()
num_cols = data.select_dtypes(include=np.number).columns.tolist()

# d. Plotting distributions for numeric features
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# e. Correlation heatmap
if len(num_cols) > 1:
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()