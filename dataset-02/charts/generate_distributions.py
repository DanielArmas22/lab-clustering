"""Script para generar gráficos de distribución del dataset-02.

Este script analiza las variables numéricas (notas) y categóricas (perfil)
del dataset Estudiantes.csv.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configuración de rutas
DATASET_PATH = Path(__file__).parent.parent / "data" / "Estudiantes.csv"
OUTPUT_DIR = Path(__file__).parent / "img"

def generate_plots():
    if not DATASET_PATH.exists():
        print(f"Error: No se encontró el dataset en {DATASET_PATH}")
        return

    # Crear carpeta para imágenes si no existe
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    sns.set_theme(style="whitegrid")

    # 1. Distribución de Notas (Numéricas)
    cols_notas = ['Matematicas', 'Lectura', 'Escritura']
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(cols_notas, 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[col], bins=15, kde=True, color='teal')
        plt.title(f'Distribución de {col}')
        plt.xlabel('Puntaje')
        plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dist_notas.png')
    plt.close()

    # 2. Distribución por Género
    plt.figure(figsize=(8, 6))
    df['Genero'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=140)
    plt.title('Distribución por Género')
    plt.ylabel('')
    plt.savefig(OUTPUT_DIR / 'dist_genero.png')
    plt.close()

    # 3. Nivel educativo de los padres (Categórica)
    plt.figure(figsize=(12, 6))
    sns.countplot(y='Nivel educativo de los padres', data=df, palette='viridis', order=df['Nivel educativo de los padres'].value_counts().index)
    plt.title('Nivel Educativo de los Padres')
    plt.xlabel('Cantidad')
    plt.ylabel('Nivel Educativo')
    plt.savefig(OUTPUT_DIR / 'dist_padres_educacion.png')
    plt.close()

    # 4. Etnia vs Preparación
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Etnia', hue='Examen de preparacion', data=df, palette='Set2')
    plt.title('Distribución de Etnia y Preparación para el Examen')
    plt.xlabel('Grupo Étnico')
    plt.ylabel('Cantidad')
    plt.savefig(OUTPUT_DIR / 'dist_etnia_preparacion.png')
    plt.close()

    print(f"Gráficos de dataset-02 generados exitosamente en: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_plots()
