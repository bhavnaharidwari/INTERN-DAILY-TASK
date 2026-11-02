# Task 3: Analyze clusters and their characteristics
#   - find the key factors that separate the clusters
#   - look at employee engagement levels
#   - interpret productivity trends
# Project: SalaryInsight - Predictive Modeling for Informed Salary Predictions

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

# Step 1: load data and prepare it (same as Task 1 and 2)
df = pd.read_csv("employee_data.csv")
df["Performance_Rating"] = df["Performance_Rating"].fillna(df["Performance_Rating"].median())
df["Training_Hours"] = df["Training_Hours"].fillna(df["Training_Hours"].median())

features = ["Performance_Rating", "Weekly_Hours", "Training_Hours",
            "Certifications", "Work_Life_Balance"]
X_scaled = StandardScaler().fit_transform(df[features])

# take the best k from Task 2 (use 5 if the file is missing)
try:
    k = int(open("best_k.txt").read())
except FileNotFoundError:
    k = 5
print("Using k =", k)

model = KMeans(n_clusters=k, n_init=10, random_state=42)
df["Cluster"] = model.fit_predict(X_scaled)


# Step 2: average values of every column in each cluster
# Satisfaction is text (High/Low), so I turn it into 1/0 to take the average
df["High_Satisfaction"] = (df["Satisfaction"] == "High").astype(int)

num_cols = features + ["Age", "Experience_Years", "Salary_LPA", "High_Satisfaction"]
profile = df.groupby("Cluster")[num_cols].mean()
profile.insert(0, "Size", df["Cluster"].value_counts().sort_index())
profile["High_Satisfaction"] = profile["High_Satisfaction"] * 100   # to percent
profile = profile.rename(columns={"High_Satisfaction": "High_Satisfaction_%"})

print("\n=== Cluster profile (averages) ===")
print(profile.round(2).to_string())


# Step 3: key distinguishing factors
# Which column differs the most between clusters?
# I use ANOVA F-score: bigger F = column separates the clusters more.

print("\n=== Key distinguishing factors (ANOVA) ===")
rows = []
for col in features + ["Age", "Experience_Years", "Salary_LPA"]:
    groups = [df[df["Cluster"] == c][col] for c in range(k)]
    f_val, p_val = f_oneway(*groups)
    rows.append((col, f_val, p_val))
anova = pd.DataFrame(rows, columns=["Feature", "F_score", "p_value"])
anova = anova.sort_values("F_score", ascending=False)
print(anova.round(4).to_string(index=False))

# Also: how far is each cluster from the overall average? (z-score of the mean)
# This tells what makes each cluster special.
overall_mean = df[features].mean()
overall_std = df[features].std()
z_table = (df.groupby("Cluster")[features].mean() - overall_mean) / overall_std
print("\nCluster averages as distance from overall mean (in std units):")
print(z_table.round(2).to_string())


# Step 4: engagement levels
wlb = (df["Work_Life_Balance"] - 1) / 4
train = df["Training_Hours"] / df["Training_Hours"].max()
sat = df["High_Satisfaction"]
df["Engagement_Score"] = (wlb + train + sat) / 3

eng = df.groupby("Cluster")["Engagement_Score"].mean()
# label the clusters low / medium / high using quantiles of the score
df["Engagement_Level"] = pd.qcut(df["Engagement_Score"], 3, labels=["Low", "Medium", "High"])
eng_level = pd.crosstab(df["Cluster"], df["Engagement_Level"], normalize="index") * 100

print("\n=== Engagement ===")
print("Average engagement score per cluster:")
print(eng.round(3).to_string())
print("\n% of employees in each engagement level:")
print(eng_level.round(1).to_string())


# Step 5: productivity trends
# Productivity = performance rating, and also performance per weekly hour

df["Perf_per_Hour"] = df["Performance_Rating"] / df["Weekly_Hours"] * 40   # rating per 40 hrs
prod = df.groupby("Cluster")[["Performance_Rating", "Weekly_Hours", "Perf_per_Hour"]].mean()
print("\n=== Productivity ===")
print(prod.round(2).to_string())

# trend: does more training / more certifications mean better performance?
print("\nCorrelation with Performance_Rating:")
print(df[["Training_Hours", "Certifications", "Weekly_Hours", "Work_Life_Balance",
          "Experience_Years", "Salary_LPA"]].corrwith(df["Performance_Rating"]).round(3).to_string())

# trend across experience: average performance by experience band
df["Exp_Band"] = pd.cut(df["Experience_Years"], [-1, 5, 10, 20, 40],
                        labels=["0-5", "6-10", "11-20", "20+"])
print("\nAverage performance by experience band:")
print(df.groupby("Exp_Band", observed=True)["Performance_Rating"].mean().round(2).to_string())

# salary link
print("\nSalary by cluster (LPA):")
print(df.groupby("Cluster")["Salary_LPA"].agg(["mean", "median", "min", "max"]).round(2).to_string())

# department mix
print("\nDepartment mix inside each cluster (%):")
print((pd.crosstab(df["Cluster"], df["Department"], normalize="index") * 100).round(1).to_string())


# Step 6: plots
# 6a - heatmap of cluster profile (z-scores)
fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(z_table.values, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, rotation=30, ha="right")
ax.set_yticks(range(k))
ax.set_yticklabels([f"Cluster {c}" for c in range(k)])
for i in range(k):
    for j in range(len(features)):
        ax.text(j, i, f"{z_table.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, label="std units from overall mean")
ax.set_title("What makes each cluster different")
plt.tight_layout()
plt.savefig("task3_cluster_heatmap.png", dpi=120)
plt.close()

# 6b - PCA scatter
pca = PCA(n_components=2)
pts = pca.fit_transform(X_scaled)
plt.figure(figsize=(7, 5))
plt.scatter(pts[:, 0], pts[:, 1], c=df["Cluster"], cmap="tab10", s=8)
plt.title("Employee clusters (PCA view)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig("task3_pca_clusters.png", dpi=120)
plt.close()

# 6c - engagement, productivity, salary per cluster
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(eng.index, eng.values, color="seagreen")
axes[0].set_title("Engagement score")
axes[1].bar(prod.index, prod["Performance_Rating"], color="steelblue")
axes[1].set_title("Average performance rating")
axes[2].bar(profile.index, profile["Salary_LPA"], color="darkorange")
axes[2].set_title("Average salary (LPA)")
for ax in axes:
    ax.set_xlabel("Cluster")
    ax.set_xticks(range(k))
plt.tight_layout()
plt.savefig("task3_engagement_productivity_salary.png", dpi=120)
plt.close()

# Step 7: save results

profile.round(2).to_csv("task3_cluster_profile.csv")
df.to_csv("employee_clustered_final.csv", index=False)
print("\nSaved: task3_cluster_profile.csv, employee_clustered_final.csv and 3 plots")
