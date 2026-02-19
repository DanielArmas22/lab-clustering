import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# 1. Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INPUT = os.path.join(BASE_DIR, 'data', 'Estudiantes.csv')
DATA_OUTPUT = os.path.join(BASE_DIR, 'results', 'estudiantes_con_clusters.csv')

def ejecutar_clustering():
    # 2. Cargar datos
    print("Cargando datos...")
    df = pd.read_csv(DATA_INPUT)
    
    # 3. Selección de características numéricas
    # Usaremos Matematicas, Lectura y Escritura para un clustering tridimensional
    columnas_clustering = ['Matematicas', 'Lectura', 'Escritura']
    X = df[columnas_clustering]
    
    # 4. Escalamiento de datos (Buena práctica para KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Aplicación del Algoritmo KMeans
    # Determinamos 5 grupos de rendimiento académico
    n_clusters = 5
    print(f"Ejecutando KMeans con {n_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init='k-means++', 
        n_init=10, 
        max_iter=300, 
        random_state=42
    )
    
    # Asignación de etiquetas
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 6. Guardar resultados
    if not os.path.exists(os.path.dirname(DATA_OUTPUT)):
        os.makedirs(os.path.dirname(DATA_OUTPUT))
        
    df.to_csv(DATA_OUTPUT, index=False)
    print(f"Resultados guardados exitosamente en: {DATA_OUTPUT}")
    
    # Mostrar resumen de clusters
    print("\nResumen por Cluster (Promedios):")
    resumen = df.groupby('Cluster')[columnas_clustering].mean()
    print(resumen)

if __name__ == "__main__":
    ejecutar_clustering()
