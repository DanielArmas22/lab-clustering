"""Clustering con K-Means sobre dataset.csv.

Basado en la lógica de referencia de rimenri.py y example.py.

Ejecución:
  python clustering.py

Requisitos:
  pandas, numpy, scikit-learn, matplotlib, seaborn
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# ----------------------
# Configuración mínima
# ----------------------
DATASET_PATH = Path(__file__).with_name("data/dataset.csv")

# Columnas a eliminar (no aportan al clustering o son IDs)
DROP_COLUMNS = ["Student_ID", "Student_Age", "Sex"]

# Elbow: calcula inercia para k=1..MAX_K y grafica
RUN_ELBOW = True
MAX_K = 10

# Distribuciones: calcula histogramas y countplots (como en rimenri.py)
RUN_DISTRIBUTIONS = True
DIST_COLS_NUM = ["Weekly_Study_Hours"]
DIST_COLS_CAT = ["Student_Age", "Sex", "Grade"]

# K a usar en el entrenamiento final (ajústalo según el codo del Elbow)
N_CLUSTERS = 3

# Guardar dataset con la columna Cluster añadida
SAVE_OUTPUT = True
OUTPUT_PATH = Path(__file__).with_name("results").joinpath("dataset_with_clusters.csv")

# Graficar una vista 2D simple (si existen ambas columnas)
# Nota: es solo una proyección; el clustering se entrena en todas las features.
PLOT_2D = True
PLOT_X_COL = "Weekly_Study_Hours"
PLOT_Y_COL = "Grade"


def _label_encode_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Aplica LabelEncoder a columnas tipo object (categóricas).

    Devuelve el df transformado + encoders por columna.
    """

    df_encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            # Convertimos a string para tolerar valores mixtos (ej. "Yes" y "6")
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders


def _plot_distributions(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> None:
    """Genera gráficos de distribución (histogramas) y de conteo (categóricos)
    similares a los de rimenri.py.
    """

    # 1) Gráficos de distribución para variables numéricas
    if num_cols:
        plt.figure(figsize=(15, 6))
        for i, col in enumerate(num_cols, 1):
            if col in df.columns:
                plt.subplot(1, len(num_cols), i)
                plt.subplots_adjust(hspace=0.5, wspace=0.5)
                # Usamos histplot (sucesor de distplot) con KDE para el efecto visual
                sns.histplot(df[col], bins=20, kde=True)
                plt.title(f"Distribución de {col}")
        plt.tight_layout()
        plt.show()

    # 2) Gráficos de conteo para variables categóricas
    if cat_cols:
        plt.figure(figsize=(15, 5))
        n_cat = len(cat_cols)
        for i, col in enumerate(cat_cols, 1):
            if col in df.columns:
                plt.subplot(1, n_cat, i)
                sns.countplot(y=col, data=df)
                plt.title(f"Conteo de {col}")
        plt.tight_layout()
        plt.show()


def _plot_elbow(X: np.ndarray, max_k: int) -> list[float]:
    inertia: list[float] = []

    for k in range(1, max_k + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            tol=0.0001,
            random_state=111,
            algorithm="elkan",
        )
        model.fit(X)
        inertia.append(float(model.inertia_))

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(1, max_k + 1), inertia, "o")
    plt.plot(np.arange(1, max_k + 1), inertia, "-", alpha=0.6)
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.title("Elbow Method (K-Means)")
    plt.tight_layout()
    plt.show()

    return inertia


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {DATASET_PATH}")

    df_raw = pd.read_csv(DATASET_PATH)

    print("\nPrimeras filas del dataset:")
    print(df_raw.head())

    print("\nValores nulos por columna:")
    print(df_raw.isnull().sum())

    # 4.5) Gráficos de distribución (Analizando datos como rimenri.py)
    if RUN_DISTRIBUTIONS:
        _plot_distributions(df_raw, DIST_COLS_NUM, DIST_COLS_CAT)

    # 1) Selección de features
    drop_cols = [c for c in DROP_COLUMNS if c in df_raw.columns]
    df = df_raw.drop(columns=drop_cols)

    # 2) Label encoding de categóricas (similar a example.py)
    df_encoded, _encoders = _label_encode_dataframe(df)

    # 3) Matriz X para clustering
    X = df_encoded.values

    # 4) Elbow (opcional)
    if RUN_ELBOW:
        _plot_elbow(X, MAX_K)

    # 5) Entrenamiento KMeans
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        random_state=111,
        algorithm="elkan",
    )
    kmeans.fit(X)

    labels = kmeans.labels_

    # 6) Adjuntar clusters al dataframe original para trazabilidad
    df_out = df_raw.copy()
    df_out["Cluster"] = labels

    print("\nDistribución de clusters:")
    print(df_out["Cluster"].value_counts().sort_index())

    # 7) Plot 2D opcional
    if PLOT_2D and (PLOT_X_COL in df_encoded.columns) and (PLOT_Y_COL in df_encoded.columns):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_out.assign(**{PLOT_X_COL: df_encoded[PLOT_X_COL], PLOT_Y_COL: df_encoded[PLOT_Y_COL]}),
            x=PLOT_X_COL,
            y=PLOT_Y_COL,
            hue="Cluster",
            palette="viridis",
            s=80,
        )
        plt.title(f"K-Means (k={N_CLUSTERS}) - {PLOT_X_COL} vs {PLOT_Y_COL} (codificado)")
        plt.tight_layout()
        plt.show()

    # 8) Guardado opcional
    if SAVE_OUTPUT:
        df_out.to_csv(OUTPUT_PATH, index=False)
        print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
