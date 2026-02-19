# Let's first import the dataset 
import pandas as pd

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