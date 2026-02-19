"""Script de visualización para K-Prototypes (Dataset 02).

Visualiza la relación entre notas y variables categóricas 
dentro de los clusters identificados por K-Prototypes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configuración de rutas
DATASET_PATH = Path(__file__).parent / "estudiantes_con_kprototypes.csv"
OUTPUT_DIR = Path(__file__).parent / "viz"

def generate_kproto_plots():
    if not DATASET_PATH.exists():
        print(f"Error: No se encontró el dataset en {DATASET_PATH}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Boxplot de Notas por Cluster (Visualizar separación numérica)
    plt.figure(figsize=(15, 6))
    df_melted = df.melt(id_vars=['Cluster_KProto'], 
                        value_vars=['Matematicas', 'Lectura', 'Escritura'],
                        var_name='Materia', value_name='Nota')
    
    sns.boxplot(data=df_melted, x='Cluster_KProto', y='Nota', hue='Materia')
    plt.title('Distribución de Notas por Cluster (K-Prototypes)')
    plt.savefig(OUTPUT_DIR / 'kproto_notas_boxplot.png')
    plt.show()

    # 2. Análisis Categórico: Nivel educativo de los padres por Cluster
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x='Cluster_KProto', hue='Nivel educativo de los padres')
    plt.title('Nivel Educativo de los Padres por Cluster')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kproto_padres_dist.png')
    plt.show()

    # 3. Preparación para el examen por Cluster
    plt.figure(figsize=(10, 6))
    props = df.groupby('Cluster_KProto')['Examen de preparacion'].value_counts(normalize=True).unstack()
    props.plot(kind='bar', stacked=True, color=['#4CAF50', '#FF5722'])
    plt.title('Proporción de Preparación para Examen por Cluster')
    plt.ylabel('Proporción')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kproto_preparacion_prop.png')
    plt.show()

    print(f"Visualizaciones de K-Prototypes generadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_kproto_plots()
