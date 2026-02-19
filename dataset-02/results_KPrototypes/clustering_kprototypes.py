"""Implementación de K-Prototypes para el dataset de Estudiantes (Dataset 02).

K-Prototypes es un algoritmo que combina K-Means (para datos numéricos) 
y K-Modes (para datos categóricos), permitiendo hacer clustering 
directamente sobre datasets mixtos sin necesidad de encoding previo exhaustivo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

# Configuración de rutas
DATASET_PATH = Path(__file__).parent.parent / "data" / "Estudiantes.csv"
OUTPUT_PATH = Path(__file__).parent / "estudiantes_con_kprototypes.csv"

def main():
    if not DATASET_PATH.exists():
        print(f"Error: No se encontró el dataset en {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    
    # 1. Preparación de datos
    # Columnas Numéricas: Matematicas, Lectura, Escritura
    # Columnas Categóricas: Genero, Etnia, Nivel educativo de los padres, Examen de preparacion
    
    # Identificamos índices de columnas categóricas para el algoritmo
    categorical_idx = [0, 1, 2, 3] # Posiciones de las columnas object
    
    # Escalamos solo las numéricas para no sesgar por magnitud
    scaler = StandardScaler()
    df_transformed = df.copy()
    num_cols = ['Matematicas', 'Lectura', 'Escritura']
    df_transformed[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Convertimos a matriz para k-prototypes
    X = df_transformed.values
    
    # 2. Ejecutar K-Prototypes
    # Usaremos 4 clusters como punto de partida razonable para este dataset
    kproto = KPrototypes(n_clusters=4, init='Cao', verbose=1, random_state=42)
    clusters = kproto.fit_predict(X, categorical=categorical_idx)
    
    # 3. Guardar resultados
    df['Cluster_KProto'] = clusters
    
    print("\nCentroides (Prototypes) encontrados:")
    print(kproto.cluster_centroids_)
    
    print("\nDistribución por cluster:")
    print(df['Cluster_KProto'].value_counts().sort_index())
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nArchivo guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
