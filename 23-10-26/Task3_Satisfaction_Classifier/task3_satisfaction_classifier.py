# Task 3 - predict employee satisfaction (High / Low)
# Classes are imbalanced (~27% High) so I oversample the minority class.
# Oversampling is only done on the training part of each fold, otherwise the
# validation data would leak into training.
import numpy as np, pandas as pd, os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, classification_report)
from sklearn.base import clone
from sklearn.utils import resample

os.makedirs("task3_outputs", exist_ok=True)
df = pd.read_csv("employee_data.csv")
y = (df["Satisfaction"] == "High").astype(int)
X = df.drop(columns=["Satisfaction"])
cat_cols = X.select_dtypes("object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
print(y.value_counts(normalize=True).round(3))

def make_prep():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    return ColumnTransformer([("num", num, num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

algos = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
}

# copy minority rows until both classes are the same size
def oversample(Xt, yt, seed=42):
    minority = yt.value_counts().idxmin()
    idx_min = yt[yt == minority].index
    idx_maj = yt[yt != minority].index
    extra = resample(idx_min, replace=True, n_samples=len(idx_maj), random_state=seed)
    idx = np.concatenate([idx_maj, extra])
    np.random.default_rng(seed).shuffle(idx)
    return Xt.loc[idx], yt.loc[idx]

def scores(p, Xv, yv):
    pred = p.predict(Xv)
    prob = p.predict_proba(Xv)[:, 1]
    return {"Accuracy": accuracy_score(yv, pred),
            "Precision": precision_score(yv, pred, zero_division=0),
            "Recall": recall_score(yv, pred), "F1": f1_score(yv, pred),
            "ROC_AUC": roc_auc_score(yv, prob)}

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)

# 5-fold CV for every algorithm, with and without balancing
rows = []
for balanced in (False, True):
    for name, model in algos.items():
        fold_scores = []
        for tr, va in skf.split(X_dev, y_dev):
            Xt, yt = X_dev.iloc[tr], y_dev.iloc[tr]
            if balanced:
                Xt, yt = oversample(Xt, yt)
            p = Pipeline([("prep", make_prep()), ("m", clone(model))]).fit(Xt, yt)
            fold_scores.append(scores(p, X_dev.iloc[va], y_dev.iloc[va]))
        f = pd.DataFrame(fold_scores)
        row = {"Model": name, "Balanced": "Yes" if balanced else "No"}
        for c in f:
            row[c + "_mean"] = f[c].mean()
            row[c + "_std"] = f[c].std()
        rows.append(row)

cv = pd.DataFrame(rows).round(4)
cv.to_csv("task3_results.csv", index=False)
print(cv[["Model", "Balanced", "Accuracy_mean", "Recall_mean", "F1_mean", "F1_std", "ROC_AUC_mean"]].to_string(index=False))

# effect of balancing
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(len(algos))
for ax, metric in zip(axes, ["Recall", "F1"]):
    before = cv[cv.Balanced == "No"][metric + "_mean"].values
    after = cv[cv.Balanced == "Yes"][metric + "_mean"].values
    ax.bar(x - 0.19, before, 0.38, label="Original")
    ax.bar(x + 0.19, after, 0.38, label="Balanced")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(").replace(" ", "\n", 1) for n in algos], fontsize=8)
    ax.set_title(metric + " for High class (CV mean)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("task3_outputs/1_balancing_effect.png", dpi=150)
plt.close()

# pick the best one using balanced CV F1, then test it once on held-out data
best_name = cv[cv.Balanced == "Yes"].sort_values("F1_mean", ascending=False).iloc[0].Model
print("\nBest model:", best_name)
Xb, yb = oversample(X_dev, y_dev)
final = Pipeline([("prep", make_prep()), ("m", clone(algos[best_name]))]).fit(Xb, yb)
print(classification_report(y_test, final.predict(X_test), target_names=["Low", "High"]))
pd.Series(scores(final, X_test, y_test)).round(4).to_csv("task3_holdout_metrics.csv", header=["value"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, final.predict(X_test)),
                       display_labels=["Low", "High"]).plot(ax=ax[0], colorbar=False)
ax[0].set_title(best_name + " - confusion matrix (test)")
for name, model in algos.items():
    p = Pipeline([("prep", make_prep()), ("m", clone(model))]).fit(Xb, yb)
    prob = p.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax[1].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, prob):.2f})")
ax[1].plot([0, 1], [0, 1], "k--")
ax[1].set_title("ROC curves (test)")
ax[1].set_xlabel("False positive rate")
ax[1].set_ylabel("True positive rate")
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("task3_outputs/2_confusion_roc.png", dpi=150)
plt.close()

# robustness: repeat the CV with 10 different seeds
f1s = []
for seed in range(10):
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_dev, y_dev):
        Xt, yt = oversample(X_dev.iloc[tr], y_dev.iloc[tr], seed)
        p = Pipeline([("prep", make_prep()), ("m", clone(algos[best_name]))]).fit(Xt, yt)
        f1s.append(f1_score(y_dev.iloc[va], p.predict(X_dev.iloc[va])))
print(f"Repeated CV F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
plt.figure(figsize=(6, 4))
plt.hist(f1s, bins=12, edgecolor="k")
plt.axvline(np.mean(f1s), color="r", ls="--")
plt.title(best_name + ": F1 over 50 folds")
plt.xlabel("F1")
plt.tight_layout()
plt.savefig("task3_outputs/3_robustness.png", dpi=150)
plt.close()
