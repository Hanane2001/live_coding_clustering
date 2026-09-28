import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================================
# 1. CHARGEMENT
# ==========================================================
df = pd.read_csv("Mall_Customers.csv")

print("Premieres lignes:")
print(df.head())

print("\nDimensions:")
print(df.shape)

print("\nInformations:")
df.info()

print("\nStatistiques:")
print(df.describe())


# ==========================================================
# 2. QUALITE DES DONNEES
# ==========================================================

print("\nValeurs manquantes:")
print(df.isnull().sum())

print("\nDoublons:")
print(df.duplicated().sum())


# ==========================================================
# 3. SELECTION DES VARIABLES
# ==========================================================

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]


# ==========================================================
# 4. STANDARDISATION
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================================
# 5. METHODE ELBOW
# ==========================================================

inertias = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertias.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertias,
    marker="o"
)

plt.xlabel("Nombre de clusters K")
plt.ylabel("Inertie")
plt.title("Methode Elbow")

plt.show()


# ==========================================================
# 6. K-MEANS
# ==========================================================

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ==========================================================
# 7. ANALYSE DES CLUSTERS
# ==========================================================

cluster_analysis = df.groupby("Cluster")[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print("\nAnalyse des clusters :")
print(cluster_analysis)


# ==========================================================
# 8. TAILLE DES CLUSTERS
# ==========================================================

print("\nNombre de clients par cluster:")
print(df["Cluster"].value_counts().sort_index())


# ==========================================================
# 9. VISUALISATION
# ==========================================================

centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["Cluster"],
    s=50
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Annual Income (standardise)")
plt.ylabel("Spending Score (standardise)")
plt.title("Segmentation des clients")

plt.show()


# ==========================================================
# 10. CLIENTS AVEC LE REVENU LE PLUS ELEVE
# ==========================================================

highest_income_cluster = cluster_analysis[
    "Annual Income (k$)"
].idxmax()

highest_income = cluster_analysis[
    "Annual Income (k$)"
].max()

print(
    f"\nCluster avec le revenu moyen le plus eleve: "
    f"{highest_income_cluster}"
)

print(
    f"Revenu moyen : {highest_income:.2f} k$"
)


# ==========================================================
# 11. CLIENTS AVEC LE SCORE DE DEPENSE LE PLUS ELEVE
# ==========================================================

highest_spending_cluster = cluster_analysis[
    "Spending Score (1-100)"
].idxmax()

highest_spending = cluster_analysis[
    "Spending Score (1-100)"
].max()

print(
    f"\nCluster avec le score de depense le plus eleve: "
    f"{highest_spending_cluster}"
)

print(
    f"Score moyen : {highest_spending:.2f}"
)