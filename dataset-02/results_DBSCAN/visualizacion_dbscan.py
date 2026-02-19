"""Script de visualización de regiones para DBSCAN (Dataset 02).

Este script genera visualizaciones 2D de los clusters identificados por DBSCAN,
incluyendo la detección de ruido (outliers).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configuración de rutas
DATASET_PATH = Path(__file__).parent / "estudiantes_con_dbscan.csv"
OUTPUT_DIR = Path(__file__).parent / "viz"

def generate_dbscan_plots():
    if not DATASET_PATH.exists():
        print(f"Error: No se encontró el dataset con clusters en {DATASET_PATH}")
        return

    # Crear carpeta para visualizaciones si no existe
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    
    # Configuramos el estilo
    sns.set_theme(style="white")
    
    # Definimos las combinaciones de ejes para comparar
    comparaciones = [
        ('Matematicas', 'Lectura'),
        ('Lectura', 'Escritura'),
        ('Matematicas', 'Escritura')
    ]

    plt.figure(figsize=(18, 5))

    for i, (x_col, y_col) in enumerate(comparaciones, 1):
        plt.subplot(1, 3, i)
        
        # Graficamos los puntos normales
        sns.scatterplot(
            data=df[df['Cluster_DBSCAN'] != -1],
            x=x_col,
            y=y_col,
            hue='Cluster_DBSCAN',
            palette='viridis',
            alpha=0.6,
            legend='full'
        )
        
        # Graficamos el ruido (Outliers) en Rojo con una X
        noise = df[df['Cluster_DBSCAN'] == -1]
        if not noise.empty:
            plt.scatter(
                noise[x_col], 
                noise[y_col], 
                c='red', 
                marker='x', 
                s=100, 
                label='Ruido (Outliers)',
                linewidths=2
            )
        
        plt.title(f'DBSCAN: {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if i == 1:
            plt.legend(title='Clusters')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'regiones_dbscan.png')
    plt.show()

    print(f"Visualización de regiones generada en: {OUTPUT_DIR / 'regiones_dbscan.png'}")

if __name__ == "__main__":
    generate_dbscan_plots()
