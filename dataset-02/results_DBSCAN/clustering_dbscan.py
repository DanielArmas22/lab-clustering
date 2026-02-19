"""Implementación de DBSCAN para el dataset de Estudiantes (Dataset 02).

Este script aplica el algoritmo de clustering DBSCAN sobre las notas de los estudiantes.
A diferencia de K-Means, DBSCAN no requiere especificar el número de clusters
y puede detectar ruido (puntos que no pertenecen a ningún grupo).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Configuración de rutas
DATASET_PATH = Path(__file__).parent.parent / "data" / "Estudiantes.csv"
OUTPUT_PATH = Path(__file__).parent / "estudiantes_con_dbscan.csv"

def scale_data(df):
    """Prepara y escala los datos para DBSCAN."""
    # Seleccionamos las columnas numéricas para el clustering (Notas)
    features = ['Matematicas', 'Lectura', 'Escritura']
    X = df[features].copy()
    
    # DBSCAN es muy sensible a la escala de las distancias
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def main():
    if not DATASET_PATH.exists():
        print(f"Error: No se encontró el dataset en {DATASET_PATH}")
        return

    df_raw = pd.read_csv(DATASET_PATH)
    
    # 1. Escalar datos
    X_scaled = scale_data(df_raw)
    
    # 2. Aplicar DBSCAN
    # eps: Distancia máxima entre dos muestras para que se consideren vecinas
    # min_samples: El número de muestras en un vecindario para que un punto sea un 'core'
    # Ajustamos eps después de varias pruebas empíricas para este tipo de datos de notas
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)
    
    # 3. Adjuntar resultados
    df_out = df_raw.copy()
    df_out['Cluster_DBSCAN'] = clusters
    
    # Análisis rápido de resultados
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"\nResultados DBSCAN:")
    print(f"Clusters encontrados: {n_clusters}")
    print(f"Puntos de ruido (outliers): {n_noise} ({n_noise/len(df_out)*100:.1f}%)")
    
    print("\nDistribución por cluster:")
    print(df_out['Cluster_DBSCAN'].value_counts().sort_index())

    # 4. Guardar resultado
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nArchivo guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
