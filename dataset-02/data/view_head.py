"""Script para visualizar los primeros registros del dataset de Estudiantes.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de ruta
DATASET_PATH = Path(__file__).parent / "Estudiantes.csv"
OUTPUT_IMAGE = Path(__file__).parent / "head_preview.png"

def main():
    if not DATASET_PATH.exists():
        print(f"Error: No se encontró el archivo {DATASET_PATH}")
        return

    # Leer el dataset
    df = pd.read_csv(DATASET_PATH)
    head_df = df.head(10)

    # Crear una figura para la tabla
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.axis('tight')

    # Crear la tabla usando matplotlib
    table = ax.table(cellText=head_df.values, 
                    colLabels=head_df.columns, 
                    cellLoc='center', 
                    loc='center')
    
    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        else:
            cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')

    plt.title("Vista previa de los 10 primeros registros", pad=20, weight='bold')

    # Guardar como imagen
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\n--- Vista de los 10 primeros estudiantes ---")
    print(head_df.to_string(index=False))
    print(f"\nImagen generada con éxito en: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
