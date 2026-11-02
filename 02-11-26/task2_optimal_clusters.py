# Task 2: Find the optimal number of clusters using the Elbow Method
# and the Silhouette Score
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# Step 1: load and prepare data (same steps as Task 1)
df = pd.read_csv("employee_data.csv")
df["Performance_Rating"] = df["Performance_Rating"].fillna(df["Performance_Rating"].median())
df["Training_Hours"] = df["Training_Hours"].fillna(df["Training_Hours"].median())

features = ["Performance_Rating", "Weekly_Hours", "Training_Hours",
            "Certifications", "Work_Life_Balance"]
X_scaled = StandardScaler().fit_transform(df[features])


# Step 2: try different values of k

k_values = list(range(2, 11))
inertias = []
sil_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))


# Step 3: print a table of results

results = pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": sil_scores})
# how much did inertia drop compared to the previous k (in %)
results["inertia_drop_%"] = results["inertia"].pct_change().mul(-100)
print(results.round(3).to_string(index=False))


# Step 4: elbow plot + silhouette plot

best_k_sil = k_values[int(np.argmax(sil_scores))]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(k_values, inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid(alpha=0.3)

axes[1].plot(k_values, sil_scores, marker="o", color="darkorange")
axes[1].axvline(best_k_sil, color="red", linestyle="--", label=f"best k = {best_k_sil}")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Average silhouette score")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("task2_elbow_and_silhouette.png", dpi=120)
plt.close()


# Step 5: find the elbow in a simple way
x = np.array(k_values, dtype=float)
y = np.array(inertias)
# scale both to 0-1 so the distance is fair
x_n = (x - x.min()) / (x.max() - x.min())
y_n = (y - y.min()) / (y.max() - y.min())
# distance from the line joining the first and last point
dist = np.abs((y_n[-1] - y_n[0]) * x_n - (x_n[-1] - x_n[0]) * y_n
              + x_n[-1] * y_n[0] - y_n[-1] * x_n[0])
best_k_elbow = k_values[int(np.argmax(dist))]

print("\nElbow method suggests     k =", best_k_elbow)
print("Silhouette score suggests k =", best_k_sil)


# Step 6: decide the final k based on both methods
if best_k_elbow == best_k_sil:
    final_k = best_k_sil
    reason = "both methods agree"
else:
    # silhouette at the elbow k vs best silhouette
    sil_at_elbow = sil_scores[k_values.index(best_k_elbow)]
    if max(sil_scores) - sil_at_elbow < 0.02:
        final_k = best_k_elbow          # almost same quality, fewer/more useful clusters
        reason = "silhouette at the elbow is almost as good as the best one"
    else:
        final_k = best_k_sil
        reason = "silhouette score is clearly better"

print(f"\nFinal choice: k = {final_k}  ({reason})")

# Step 7: detailed check of the chosen k (silhouette per cluster)
# A cluster with a low average score or many negative values is weak.

final_model = KMeans(n_clusters=final_k, n_init=10, random_state=42)
final_labels = np.asarray(final_model.fit_predict(X_scaled), dtype=np.intp)
sample_sil = np.asarray(
    silhouette_samples(X_scaled, final_labels), dtype=np.float64
)

print("\nSilhouette details for each cluster (k = %d):" % final_k)
for c in range(final_k):
    vals = sample_sil[final_labels == c]
    print(f"  Cluster {c}: size = {len(vals):4d} | avg silhouette = {vals.mean():.3f}"
          f" | negative values = {(vals < 0).sum()}")

# silhouette plot for the final k
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
for c in range(final_k):
    vals = np.sort(sample_sil[final_labels == c])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * len(vals), str(c))
    y_lower = y_upper + 10
ax.axvline(sil_scores[k_values.index(final_k)], color="red", linestyle="--")
ax.set_title(f"Silhouette plot for k = {final_k}")
ax.set_xlabel("Silhouette value")
ax.set_ylabel("Cluster")
plt.tight_layout()
plt.savefig("task2_silhouette_plot_final_k.png", dpi=120)
plt.close()

# Step 8: save results for Task 3

results.to_csv("task2_k_evaluation_table.csv", index=False)
with open("best_k.txt", "w") as f:
    f.write(str(final_k))
print("\nSaved: best_k.txt, task2_k_evaluation_table.csv and 2 plots")
