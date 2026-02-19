# ============================================
# CLUSTERING (K-Means) PARA DATASET DE ESTUDIANTES
# - Soporta variables categóricas y numéricas
# - Hace limpieza/transformaciones típicas (Age rango, %)
# - Elige K con Elbow + Silhouette
# - Visualiza clusters con PCA (2D)
# ============================================

# 1) Importando librerías
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 2) Leyendo datos (ajusta la ruta)
# Si tu CSV está en otro lado, cambia la ruta. Ej:
# df = pd.read_csv(r"c:\ruta\students.csv")
df = pd.read_csv("datasets.csv")

print("Primeras filas:")
display(df.head() if "display" in globals() else df.head())

print("\nDescripción (numéricas si existen):")
print(df.describe(include="all"))

print("\nNulos por columna:")
print(df.isnull().sum())


# =========================================================
# 3) Limpieza / Feature Engineering (recomendado para tu dataset)
# =========================================================

def age_to_numeric(x):
    """
    Convierte:
      - '19-22' -> 20.5 (promedio)
      - '18' -> 18
      - NaN -> NaN
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            return (float(a) + float(b)) / 2.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def percent_to_float(x):
    """
    Convierte:
      - '50%' -> 50.0
      - 50 -> 50.0
      - NaN -> NaN
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "")
    try:
        return float(s)
    except:
        return np.nan

# Copia para no tocar el original
data = df.copy()

# Si existe Student_Age como en tu ejemplo, conviértelo a número
if "Student_Age" in data.columns:
    data["Student_Age_num"] = data["Student_Age"].apply(age_to_numeric)

# Si existe Scholarship como '50%', conviértelo
if "Scholarship" in data.columns:
    data["Scholarship_num"] = data["Scholarship"].apply(percent_to_float)

# Weekly_Study_Hours puede venir como texto -> numérico
if "Weekly_Study_Hours" in data.columns:
    data["Weekly_Study_Hours"] = pd.to_numeric(data["Weekly_Study_Hours"], errors="coerce")


# =========================================================
# 4) Selección de variables para clustering
#    - No uses IDs ni el target (Grade) para clustering (salvo que quieras)
# =========================================================

drop_cols = []
for c in ["Student_ID", "Grade"]:
    if c in data.columns:
        drop_cols.append(c)

X = data.drop(columns=drop_cols)

# Si creamos columnas numéricas derivadas, podemos decidir quitar las originales:
# - Quitar Student_Age original (texto) y Scholarship original (texto) si ya tenemos las numéricas
if "Student_Age_num" in X.columns and "Student_Age" in X.columns:
    X = X.drop(columns=["Student_Age"])
if "Scholarship_num" in X.columns and "Scholarship" in X.columns:
    X = X.drop(columns=["Scholarship"])

print("\nColumnas usadas para clustering:")
print(list(X.columns))


# =========================================================
# 5) Definir preprocesamiento: OneHot para categóricas, Escalado para numéricas
# =========================================================
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nNuméricas:", num_cols)
print("Categóricas:", cat_cols)

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# =========================================================
# 6) Elegir K (Elbow + Silhouette)
# =========================================================
K_RANGE = range(2, 11)  # prueba de 2 a 10 (ajusta si tu dataset es grande)
inertias = []
sil_scores = []

# Transformamos una sola vez para evaluar rápido
X_trans = preprocess.fit_transform(X)

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=300, random_state=42)
    labels = km.fit_predict(X_trans)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_trans, labels))

# Gráfica Elbow
plt.figure(figsize=(10, 4))
plt.plot(list(K_RANGE), inertias, marker="o")
plt.title("Elbow (Inertia) vs K")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Gráfica Silhouette
plt.figure(figsize=(10, 4))
plt.plot(list(K_RANGE), sil_scores, marker="o")
plt.title("Silhouette Score vs K")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Elegimos el mejor K por silhouette (suele ser buena heurística)
best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
print(f"\nMejor K sugerido por Silhouette: {best_k}")


# =========================================================
# 7) Entrenar KMeans final con best_k
# =========================================================
kmeans_model = KMeans(
    n_clusters=best_k,
    init="k-means++",
    n_init=30,
    max_iter=300,
    random_state=42
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("kmeans", kmeans_model)
])

labels = pipe.fit_predict(X)

# Agregar clusters al dataframe original (df) para análisis
df_clusters = df.copy()
df_clusters["Cluster"] = labels

print("\nDataset con clusters (primeras filas):")
display(df_clusters.head() if "display" in globals() else df_clusters.head())


# =========================================================
# 8) Visualización 2D con PCA (sobre datos transformados)
# =========================================================
X_trans_final = pipe.named_steps["preprocess"].transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_trans_final.toarray() if hasattr(X_trans_final, "toarray") else X_trans_final)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Cluster"] = labels

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=80)
plt.title("Clusters visualizados con PCA (2D)")
plt.grid(True)
plt.show()


# =========================================================
# 9) Perfilado rápido de clusters (resumen)
# =========================================================
# Para numéricas: media por cluster
num_in_original = [c for c in df_clusters.columns if pd.api.types.is_numeric_dtype(df_clusters[c]) and c != "Cluster"]
if len(num_in_original) > 0:
    print("\nPromedios de variables numéricas por cluster:")
    print(df_clusters.groupby("Cluster")[num_in_original].mean())

# Para categóricas: moda por cluster (la categoría más frecuente)
cat_in_original = [c for c in df_clusters.columns if (c not in num_in_original) and (c not in ["Student_ID", "Cluster"])]
if len(cat_in_original) > 0:
    print("\nModa de variables categóricas por cluster:")
    modes = df_clusters.groupby("Cluster")[cat_in_original].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    print(modes)

# (Opcional) Guardar a CSV
df_clusters.to_csv("students_with_clusters.csv", index=False)
print("\n✅ Guardado: students_with_clusters.csv")
