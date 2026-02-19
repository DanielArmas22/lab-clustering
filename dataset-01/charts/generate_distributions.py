"""Script para generar gráficos de distribución del dataset.

Este script analiza las variables numéricas y categóricas para visualizar
cómo se distribuyen los datos de los estudiantes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configuración de rutas
# Asumimos que el dataset está en la carpeta superior del proyecto
DATASET_PATH = Path(__file__).parent.parent.parent / "dataset.csv"
OUTPUT_DIR = Path(__file__).parent / "img"

def generate_plots():
    if not DATASET_PATH.exists():
        print(f"Error: No se encontró el dataset en {DATASET_PATH}")
        return

    # Crear carpeta para imágenes si no existe
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    sns.set_theme(style="whitegrid")

    # 1. Distribución de Horas de Estudio (Numérica)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Weekly_Study_Hours'], bins=15, kde=True, color='skyblue')
    plt.title('Distribución de Horas de Estudio Semanales')
    plt.xlabel('Horas')
    plt.ylabel('Frecuencia')
    plt.savefig(OUTPUT_DIR / 'dist_study_hours.png')
    plt.close()

    # 2. Distribución de Edades (Categórica)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Student_Age', data=df, palette='viridis', order=sorted(df['Student_Age'].unique()))
    plt.title('Distribución por Rango de Edad')
    plt.xlabel('Rango de Edad')
    plt.ylabel('Cantidad de Estudiantes')
    plt.savefig(OUTPUT_DIR / 'dist_age.png')
    plt.close()

    # 3. Distribución de Género
    plt.figure(figsize=(8, 6))
    df['Sex'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=90)
    plt.title('Proporción de Género')
    plt.ylabel('')
    plt.savefig(OUTPUT_DIR / 'dist_gender.png')
    plt.close()

    # 4. Distribución de Notas (Grade)
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Grade', data=df, palette='magma', order=sorted(df['Grade'].unique()))
    plt.title('Distribución de Calificaciones (Grades)')
    plt.xlabel('Calificación')
    plt.ylabel('Cantidad')
    plt.savefig(OUTPUT_DIR / 'dist_grades.png')
    plt.close()

    print(f"Gráficos generados exitosamente en: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_plots()
