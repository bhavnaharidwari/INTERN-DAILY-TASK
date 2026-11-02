# Task 1: Implement k-Means clustering to group employees based on productivity
# Project: SalaryInsight - Predictive Modeling for Informed Salary Predictions
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # so plots can be saved without a screen
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from typing import Literal

DistanceMetric = Literal["euclidean", "cityblock", "cosine"]
# Step 1: load the data

df = pd.read_csv("employee_data.csv")
print("Shape of data:", df.shape)
print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Performance_Rating and Training_Hours have 25 missing values each.
# I fill them with the median because it is not affected by outliers.
df["Performance_Rating"] = df["Performance_Rating"].fillna(df["Performance_Rating"].median())
df["Training_Hours"] = df["Training_Hours"].fillna(df["Training_Hours"].median())


# Step 2: choose the productivity features and scale them
features = ["Performance_Rating", "Weekly_Hours", "Training_Hours",
            "Certifications", "Work_Life_Balance"]
X = df[features]

# k-Means uses distance, so all columns must be on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: my own simple k-Means function
# (sklearn KMeans only supports euclidean, so I wrote a small one
#  where I can change the distance metric)
def my_kmeans(data, k, metric: DistanceMetric = "euclidean", max_iter=100, seed=0):
    rng = np.random.RandomState(seed)

    # pick k random points as the starting centers
    start_idx = rng.choice(len(data), k, replace=False)
    centers = data[start_idx]

    for i in range(max_iter):
        # distance of every point to every center
        distances = cdist(data, centers, metric=metric)

        # each point goes to its nearest center
        labels = np.argmin(distances, axis=1)

        # new center = mean of the points in that cluster
        new_centers = centers.copy()
        for c in range(k):
            points = data[labels == c]
            if len(points) > 0:
                new_centers[c] = points.mean(axis=0)

        # stop if the centers did not move
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    # inertia = total distance of points from their center
    distances = cdist(data, centers, metric=metric)
    labels = np.argmin(distances, axis=1)
    inertia = distances[np.arange(len(data)), labels].sum()
    return labels, centers, inertia


def best_of_runs(data, k, metric: DistanceMetric, n_runs=10, seed_start=0):
    # run with different starting points and keep the best (lowest inertia)
    best = None
    for seed in range(seed_start, seed_start + n_runs):
        labels, centers, inertia = my_kmeans(data, k, metric, seed=seed)
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    if best is None:
        raise ValueError("n_runs must be at least 1")
    return best


# Step 4: test different distance metrics
k = 3   # starting value, the best k is found in Task 2
metrics: list[DistanceMetric] = ["euclidean", "cityblock", "cosine"]   # cityblock = manhattan

results = {}
print("\n--- Comparing distance metrics (k = %d) ---" % k)
for m in metrics:
    labels, centers, inertia = best_of_runs(X_scaled, k, m)
    score = silhouette_score(X_scaled, labels, metric=m)
    results[m] = labels
    sizes = np.bincount(labels)
    print(f"{m:10s} | silhouette = {score:.3f} | cluster sizes = {sizes}")

# How similar are the clusters from different metrics?
# adjusted rand index: 1 = same clusters, 0 = random
print("\nAgreement between metrics (Adjusted Rand Index):")
for i in range(len(metrics)):
    for j in range(i + 1, len(metrics)):
        ari = adjusted_rand_score(results[metrics[i]], results[metrics[j]])
        print(f"  {metrics[i]} vs {metrics[j]}: {ari:.3f}")



# Step 5: check stability of the clusters
# Test A: single random start each time (can be unlucky)
# Test B: best of 10 starts each time (what I actually use)
# I compare every run with the first run using ARI.
# ARI close to 1 means the clusters are stable.
print("\n--- Stability check ---")
for m in metrics:
    # Test A
    first, _, _ = my_kmeans(X_scaled, k, m, seed=0)
    single = [adjusted_rand_score(first, my_kmeans(X_scaled, k, m, seed=s)[0])
              for s in range(1, 11)]

    # Test B (each batch uses different seeds)
    first_b = best_of_runs(X_scaled, k, m, seed_start=0)[0]
    multi = [adjusted_rand_score(first_b, best_of_runs(X_scaled, k, m, seed_start=s * 100)[0])
             for s in range(1, 6)]

    print(f"{m:10s} | 1 start: avg ARI = {np.mean(single):.3f} (min {np.min(single):.3f})"
          f" | best of 10 starts: avg ARI = {np.mean(multi):.3f} (min {np.min(multi):.3f})")

# Using only one random start can give a bad result, so for the final
# clustering I use 10 starts and keep the best one (see best_of_runs).
# Here I use euclidean distance as the final choice.
final_labels, final_centers, _ = best_of_runs(X_scaled, k, "euclidean")
df["Cluster"] = final_labels


# Step 6: plot the clusters (PCA squeezes 5 columns into 2 so we can draw it)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, m in zip(axes, metrics):
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=results[m], cmap="viridis", s=8)
    ax.set_title(f"k-Means with {m} distance")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
plt.tight_layout()
plt.savefig("task1_clusters_by_metric.png", dpi=120)
plt.close()


# Step 7: save the result so Task 2 and Task 3 can use it
df.to_csv("employee_clustered_k3.csv", index=False)
print("\nSaved: employee_clustered_k3.csv and task1_clusters_by_metric.png")
