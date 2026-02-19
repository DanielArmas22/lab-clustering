# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('dataset.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Update the features to use valid column names from the dataset
features = ['Weekly_Study_Hours', 'Attendance']  # Replace with relevant numeric columns
X = df[features].values

# Convert non-numeric columns to numeric using label encoding
for feature in features:
    if df[feature].dtype == 'object':
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

# Update the feature matrix after encoding
X = df[features].values

# Visualize the distribution of features
plt.figure(figsize=(15, 6))
for i, feature in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sns.histplot(X[:, i], kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Calculate inertia for different numbers of clusters
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.show()

# Apply KMeans with the optimal number of clusters (e.g., 5)
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the clusters
plt.figure(figsize=(15, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=100, cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Weekly Study Hours')
plt.ylabel('Attendance')
plt.title('Clusters and Centroids')
plt.legend()
plt.show()

# Add cluster labels to the dataset
df['Cluster'] = labels

# Display the updated dataset
print("\nUpdated dataset with cluster labels:")
print(df.head())