# Task 1 - compare random forest with the other models I tried before
import numpy as np, pandas as pd, os, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("task1_outputs", exist_ok=True)
df = pd.read_csv("employee_data.csv")

X = df.drop(columns=["Salary_LPA", "Satisfaction"])
y = df["Salary_LPA"]
cat_cols = X.select_dtypes("object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# fill missing values + scale numbers, one-hot the text columns
def make_prep():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    return ColumnTransformer([("num", num, num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10),
    "Lasso": Lasso(alpha=0.05),
    "Decision Tree (deep)": DecisionTreeRegressor(random_state=42),
    "Decision Tree (depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
}
pipes = {name: Pipeline([("prep", make_prep()), ("m", m)]) for name, m in models.items()}

# keep 20% aside as unseen test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(10, shuffle=True, random_state=42)

rows = []
cv_scores = {}
for name, p in pipes.items():
    r2s = cross_val_score(p, X_train, y_train, cv=kf, scoring="r2")
    rmses = -cross_val_score(p, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    cv_scores[name] = r2s
    p.fit(X_train, y_train)
    train_rmse = mean_squared_error(y_train, p.predict(X_train)) ** 0.5
    test_pred = p.predict(X_test)
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
    rows.append({"Model": name, "CV_R2_mean": r2s.mean(), "CV_R2_std": r2s.std(),
                 "CV_RMSE_mean": rmses.mean(), "CV_RMSE_std": rmses.std(),
                 "Train_RMSE": train_rmse, "Test_RMSE": test_rmse,
                 "Test_MAE": mean_absolute_error(y_test, test_pred),
                 "Test_R2": r2_score(y_test, test_pred),
                 "Overfit_Gap": test_rmse - train_rmse})

res = pd.DataFrame(rows).round(4)
res.to_csv("task1_results.csv", index=False)
print(res.to_string(index=False))

labels = [n.replace(" (", "\n(") for n in res.Model]

# stability plot
plt.figure(figsize=(10, 5))
plt.boxplot(cv_scores.values(), labels=labels, showmeans=True)
plt.ylabel("R2 over 10 folds")
plt.title("Stability (10-fold CV R2) - smaller box = more stable")
plt.xticks(fontsize=8)
plt.tight_layout()
plt.savefig("task1_outputs/1_stability_boxplot.png", dpi=150)
plt.close()

# train vs test error (bias / variance)
x = np.arange(len(res))
plt.figure(figsize=(10, 5))
plt.bar(x - 0.19, res.Train_RMSE, 0.38, label="Train RMSE")
plt.bar(x + 0.19, res.Test_RMSE, 0.38, label="Test RMSE")
plt.xticks(x, labels, fontsize=8)
plt.ylabel("RMSE (LPA)")
plt.title("Bias-variance: train vs test error")
plt.legend()
plt.tight_layout()
plt.savefig("task1_outputs/2_bias_variance.png", dpi=150)
plt.close()

# learning curves for 3 models
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, name in zip(axes, ["Linear Regression", "Decision Tree (deep)", "Random Forest"]):
    sizes, tr, va = learning_curve(pipes[name], X_train, y_train, cv=5,
                                   train_sizes=np.linspace(0.15, 1, 6),
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
    ax.plot(sizes, -tr.mean(1), "o-", label="Train")
    ax.plot(sizes, -va.mean(1), "o-", label="Validation")
    ax.set_title(name)
    ax.set_xlabel("Training samples")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("RMSE (LPA)")
axes[0].legend()
plt.tight_layout()
plt.savefig("task1_outputs/3_learning_curves.png", dpi=150)
plt.close()

# which features did the forest use most
rf = pipes["Random Forest"]
feat_names = [n.split("__")[1] for n in rf["prep"].get_feature_names_out()]
imp = pd.Series(rf["m"].feature_importances_, index=feat_names).sort_values().tail(10)
imp.plot.barh(figsize=(7, 4), title="Random Forest - top 10 features")
plt.tight_layout()
plt.savefig("task1_outputs/4_rf_feature_importance.png", dpi=150)
plt.close()

best = res.sort_values("Test_RMSE").iloc[0]
print("\nBest on test data:", best.Model, "RMSE", best.Test_RMSE, "R2", best.Test_R2)
