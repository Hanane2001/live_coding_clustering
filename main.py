import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Chargement des données dans un DataFrame
df = pd.read_csv('Mall_Customers.csv')

# Exploration initiale
print("--- 5 premières lignes ---")
print(df.head())

print("\n--- Informations sur les colonnes et types ---")
print(df.info())

print("\n--- Statistiques descriptives ---")
print(df.describe())

# Vérification des valeurs manquantes et des doublons
print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())

print("\n--- Nombre de doublons ---")
print(df.duplicated().sum())

# Sélection des variables numériques pertinentes pour le clustering
# On retient le Revenu Annuel et le Score de Dépense
X_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[X_cols]

# calculer moye et ecart type
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(
    "\nDonnées standardisées (5 premières lignes) :\n",
    pd.DataFrame(X_scaled, columns=X_cols).head(),
)

# Etape 2
# Tester plusieurs valeurs de K (de 1 à 10)
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X_scaled)
    inertias.append(kmeans_test.inertia_)

# Utiliser la méthode Elbow (du Coude) pour déterminer K
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker='o', linestyle='--', color='b')
plt.title('Méthode du Coude (Elbow Method)')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('Inertie intra-cluster (SSE)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Appliquer KMeans avec le nombre de clusters choisi (K=5)
k_optimal = 8
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)

# Ajouter les labels des clusters a chaque client
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n--- Aperçu du DataFrame avec la colonne Cluster ---")
print(df.head())

# 7. ANALYSE DES CLUSTERS

cluster_analysis = df.groupby("Cluster")[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print("\nAnalyse des clusters :")
print(cluster_analysis)


# 8. TAILLE DES CLUSTERS

print("\nNombre de clients par cluster:")
print(df["Cluster"].value_counts().sort_index())


# 9. VISUALISATION
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


# 10. CLIENTS AVEC LE REVENU LE PLUS ELEVE
highest_income_cluster = cluster_analysis["Annual Income (k$)"].idxmax()
highest_income = cluster_analysis["Annual Income (k$)"].max()

print(f"\nCluster avec le revenu moyen le plus eleve: {highest_income_cluster}")
print(f"Revenu moyen : {highest_income:.2f} k$")


# 11. CLIENTS AVEC LE SCORE DE DEPENSE LE PLUS ELEVE
highest_spending_cluster = cluster_analysis["Spending Score (1-100)"].idxmax()
highest_spending = cluster_analysis["Spending Score (1-100)"].max()

print(f"\nCluster avec le score de depense le plus eleve: {highest_spending_cluster}")
print(f"Score moyen: {highest_spending:.2f}")