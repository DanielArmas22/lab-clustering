import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import os

# Definir rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
OUTPUT_IMAGE = os.path.join(BASE_DIR, 'results', 'clusters_regiones.png')

# Cargar datos
df = pd.read_csv(DATA_PATH)

# Preprocesamiento
# Usaremos 'Weekly_Study_Hours' y 'Attendance' para la visualización 2D
# 'Attendance' debe ser convertido a numérico
le = LabelEncoder()
df['Attendance_Encoded'] = le.fit_transform(df['Attendance'].astype(str))

features = ['Weekly_Study_Hours', 'Attendance_Encoded']
X = df[features].values

# Aplicar KMeans (usaremos 3 clusters como ejemplo)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Preparar la malla (meshgrid) para visualizar las REGIONES
h = 0.05  # paso de la malla
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predecir el cluster para cada punto en la malla
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar
plt.figure(figsize=(12, 8))
plt.clf()

# Dibujar las regiones de fondo
plt.imshow(Z, interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

# Añadir Jitter (ruido aleatorio) para evitar el solapamiento de puntos
# Esto permite ver los 145 registros aunque tengan los mismos valores
X_jitter = X + np.random.normal(0, 0.15, size=X.shape)

# Dibujar los puntos de datos con jitter y color según el cluster real
plt.scatter(X_jitter[:, 0], X_jitter[:, 1], c=labels, s=40, edgecolor='white', linewidth=0.5, alpha=0.8, cmap='viridis')

# Dibujar los centroides
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='X', c='red', edgecolor='black', label='Centroides')

plt.title(f'Distribución de Estudiantes (N=145) con Jitter\nVisualización de Clusters por Regiones (K={n_clusters})')
plt.xlabel('Weekly Study Hours')
plt.ylabel('Attendance (Encoded)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

# Guardar la imagen
plt.savefig(OUTPUT_IMAGE)
print(f"Gráfico guardado en: {OUTPUT_IMAGE}")
plt.show()
