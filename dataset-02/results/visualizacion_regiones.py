import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

# Definir rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Estudiantes.csv')
OUTPUT_IMAGE = os.path.join(BASE_DIR, 'results', 'clusters_estudiantes_02.png')

# Cargar datos
df = pd.read_csv(DATA_PATH)

# Seleccionar variables numéricas para el clustering (Lectura vs Escritura por ejemplo)
features = ['Lectura', 'Escritura']
X = df[features].values

# Aplicar KMeans
# Usaremos 5 clusters para ver una segmentación más detallada
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Preparar la malla (meshgrid) para visualizar las REGIONES
h = 0.5  # paso de la malla (ajustado para escala 0-100)
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
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

# Añadir Jitter para ver la densidad de los 1000 registros
X_jitter = X + np.random.normal(0, 0.5, size=X.shape)

# Dibujar los puntos de datos
plt.scatter(X_jitter[:, 0], X_jitter[:, 1], c=labels, s=20, edgecolor='white', linewidth=0.3, alpha=0.7, cmap='viridis')

# Dibujar los centroides
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='red', edgecolor='black', label='Centroides')

# Añadir etiquetas a cada cluster
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1] + 2, f'Cluster {i}', fontsize=12, 
             fontweight='bold', color='black', ha='center',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.3'))

plt.title(f'Visualización de Clusters por Regiones - Nuevo Dataset (N=1000)\nSegmentación por Puntajes de Lectura vs Escritura (K={n_clusters})')
plt.xlabel('Puntaje de Lectura')
plt.ylabel('Puntaje de Escritura')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

# Guardar la imagen
plt.savefig(OUTPUT_IMAGE)
print(f"Gráfico guardado en: {OUTPUT_IMAGE}")
plt.show()
