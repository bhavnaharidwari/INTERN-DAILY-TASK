"""
Task 8: Machine Learning Modeling, Benchmarking & Reporting Pipeline
SalaryInsight: Predictive Modeling for Informed Salary Predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc
)

# Classifiers
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
import joblib

# Aesthetic settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300

DATA_PATH = '/home/user/data/statistics_practice_dataset_2000.csv'
OUTPUT_DIR = '/home/user/visualizations/task_8'
SAVED_MODELS_DIR = '/home/user/saved_models'
DATA_OUTPUT_DIR = '/home/user/data'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def load_and_engineer_features():
    """Load dataset, engineer targets and categorical/numerical features."""
    print("Loading dataset for ML Modeling...")
    df = pd.read_csv(DATA_PATH)
    
    # Feature Engineering
    # Convert JoinDate to JoinYear / Tenure
    df['JoinDate'] = pd.to_datetime(df['JoinDate'])
    df['JoinYear'] = df['JoinDate'].dt.year
    df['TenureYears'] = 2026 - df['JoinYear']
    
    # Target 1: Multi-class Salary Tier
    # Entry (< 700k), Mid (700k - 1.2M), Senior/Executive (> 1.2M)
    bins = [0, 700000, 1200000, np.inf]
    labels = ['Entry (<700k)', 'Mid (700k-1.2M)', 'Senior/Executive (>1.2M)']
    df['SalaryTier'] = pd.cut(df['AnnualSalary'], bins=bins, labels=labels)
    
    # Target 2: Binary High Earner (>= 1,000,000)
    df['HighEarner'] = (df['AnnualSalary'] >= 1000000).astype(int)
    
    # Feature columns
    num_features = [
        'Age', 'ExperienceYears', 'PerformanceScore', 'WeeklyHours',
        'ProjectsCompleted', 'SatisfactionScore', 'CommuteKm', 'BMI',
        'MonthlySalesK', 'DiscountPercent', 'TenureYears'
    ]
    cat_features = ['Department', 'City', 'Gender']
    
    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=True)
    feature_names = df_encoded.columns.tolist()
    
    return df, df_encoded, feature_names, num_features, cat_features


def benchmark_classification_models(df, X_df, feature_names):
    """Train and evaluate 7 Classification algorithms on Salary Tier."""
    print("\n" + "="*50)
    print("BENCHMARKING CLASSIFICATION MODELS (Salary Tier)")
    print("="*50)
    
    le = LabelEncoder()
    y = le.fit_transform(df['SalaryTier'])
    target_names = le.classes_
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, 'classification_scaler.joblib'))
    joblib.dump(le, os.path.join(SAVED_MODELS_DIR, 'label_encoder.joblib'))
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42, eval_metric='mlogloss'),
        'Support Vector Machine': SVC(probability=True, kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9)
    }
    
    results = []
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted') if y_prob is not None else np.nan
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        results.append({
            'Model': name,
            'Test Accuracy': round(acc, 4),
            'CV Accuracy Mean': round(cv_scores.mean(), 4),
            'CV Accuracy Std': round(cv_scores.std(), 4),
            'Precision (Weighted)': round(float(np.asarray(prec).item()), 4),
            'Recall (Weighted)': round(float(np.asarray(rec).item()), 4),
            'F1-Score (Weighted)': round(float(np.asarray(f1).item()), 4),
            'ROC-AUC (OVR)': round(roc_auc, 4)
        })
        
    results_df = pd.DataFrame(results).sort_values(by='F1-Score (Weighted)', ascending=False)
    print("\n=== CLASSIFICATION BENCHMARK RESULTS ===\n", results_df.to_string(index=False))
    
    results_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'model_benchmark_results.csv'), index=False)
    
    # Save best classifier
    best_model_name = results_df.iloc[0]['Model']
    best_classifier = trained_models[best_model_name]
    joblib.dump(best_classifier, os.path.join(SAVED_MODELS_DIR, 'salary_tier_classifier.joblib'))
    print(f"\nBest Classifier Saved: {best_model_name} -> {os.path.join(SAVED_MODELS_DIR, 'salary_tier_classifier.joblib')}")
    
    return results_df, trained_models, X_test_scaled, y_test, target_names, feature_names, X_train_scaled, y_train


def benchmark_regression_models(df, X_df):
    """Train and evaluate regression models for Annual Salary prediction."""
    print("\n" + "="*50)
    print("BENCHMARKING REGRESSION MODELS (Annual Salary Prediction)")
    print("="*50)
    
    y = df['AnnualSalary'].values
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
        X_df, y, y_log, test_size=0.20, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, 'regression_scaler.joblib'))
    
    reg_models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, random_state=42),
        'XGBoost Regressor': XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    reg_results = []
    trained_regs = {}
    
    for name, model in reg_models.items():
        # Train on log salary for stable variance
        model.fit(X_train_scaled, y_train_log)
        trained_regs[name] = model
        
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test_log, y_pred_log)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        reg_results.append({
            'Model': name,
            'MAE (INR)': round(mae, 2),
            'RMSE (INR)': round(rmse, 2),
            'R2 Score (Log Scale)': round(r2, 4),
            'MAPE (%)': round(mape, 2)
        })
        
    reg_results_df = pd.DataFrame(reg_results)
    print("\n=== REGRESSION BENCHMARK RESULTS ===\n", reg_results_df.to_string(index=False))
    reg_results_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'regression_benchmark_results.csv'), index=False)
    
    # Save best regressor
    best_reg_name = 'XGBoost Regressor' if 'XGBoost Regressor' in trained_regs else 'Random Forest Regressor'
    joblib.dump(trained_regs[best_reg_name], os.path.join(SAVED_MODELS_DIR, 'salary_regressor.joblib'))
    print(f"Best Regressor Saved: {best_reg_name}")
    
    return reg_results_df


def plot_model_comparison(results_df):
    """Generate comparative bar chart of classification models."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df_sorted = results_df.sort_values(by='Test Accuracy', ascending=True)
    y_pos = np.arange(len(df_sorted))
    
    bars1 = ax.barh(y_pos - 0.2, df_sorted['Test Accuracy'], height=0.35, label='Test Accuracy', color='#2563eb', alpha=0.9)
    bars2 = ax.barh(y_pos + 0.2, df_sorted['F1-Score (Weighted)'], height=0.35, label='F1-Score (Weighted)', color='#10b981', alpha=0.9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['Model'], fontweight='bold')
    ax.set_xlabel('Score (0.0 to 1.0)')
    ax.set_title('Classification Model Benchmark Comparison (Salary Tier)', pad=15)
    ax.set_xlim(0, 1.0)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.1%}", va='center', fontsize=9, color='#1e3a8a')
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.1%}", va='center', fontsize=9, color='#065f46')
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'model_comparison_barchart.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_confusion_matrices_grid(trained_models, X_test, y_test, target_names):
    """Plot multi-panel confusion matrix grid for top 4 classifiers."""
    top_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Logistic Regression']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for ax, name in zip(axes.flat, top_models):
        if name in trained_models:
            model = trained_models[name]
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                cbar=False, annot_kws={'size': 13, 'weight': 'bold'}
            )
            ax.set_title(f"{name} Confusion Matrix", fontsize=13, pad=10)
            ax.set_xlabel("Predicted Tier")
            ax.set_ylabel("Actual Tier")
            
    plt.suptitle("Confusion Matrices for Primary Salary Tier Classifiers", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'confusion_matrices_grid.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_roc_auc_curves(trained_models, X_test, y_test, target_names):
    """Plot Multiclass One-vs-Rest ROC curves for best models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2563eb', '#10b981', '#f59e0b', '#dc2626']
    top_models = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Logistic Regression']
    
    for (name, color) in zip(top_models, colors):
        if name in trained_models:
            model = trained_models[name]
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                # Compute macro ROC curve
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(len(target_names)):
                    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Average micro/macro
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(target_names))]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(target_names)):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= len(target_names)
                macro_auc = auc(all_fpr, mean_tpr)
                
                ax.plot(all_fpr, mean_tpr, label=f"{name} (Macro AUC = {macro_auc:.3f})", color=color, linewidth=2.5)
                
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Chance Baseline (AUC = 0.50)')
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Multi-Class Benchmark', pad=15)
    ax.legend(loc="lower right", frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'roc_auc_curves.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_feature_importances(trained_models, feature_names):
    """Plot Top 10 feature importances from Tree-based models."""
    rf = trained_models.get('Random Forest')
    if rf is not None:
        importances = rf.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        feat_df.to_csv(os.path.join(DATA_OUTPUT_DIR, 'feature_importance.csv'), index=False)
        
        top10 = feat_df.head(10).sort_values(by='Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(top10['Feature'], top10['Importance'], color='#3b82f6', edgecolor='#1d4ed8')
        
        ax.set_title("Top 10 Feature Importances Driving Salary Tiers (Random Forest)", pad=15)
        ax.set_xlabel("Relative Feature Importance")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.003, bar.get_y() + bar.get_height()/2, f"{w:.1%}", va='center', fontsize=10, fontweight='bold', color='#1e3a8a')
            
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'feature_importances_top10.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved: {plot_path}")


def plot_salary_tier_and_equity_audit(df):
    """Plot Salary Tier Distribution & HR Equity Audit (Gender/Department)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Salary Tier distribution
    tier_counts = df['SalaryTier'].value_counts()
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    ax1.pie(tier_counts, labels=tier_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.04, 0.04, 0.04))
    ax1.set_title("Employee Headcount Distribution by Salary Tier", pad=15)
    
    # 2. Gender & Department Median Salary (Equity Audit)
    dept_gender = df.groupby(['Department', 'Gender'])['AnnualSalary'].median().unstack()
    dept_gender = dept_gender[['Male', 'Female']]
    dept_gender.plot(kind='bar', ax=ax2, color=['#2563eb', '#ec4899'], width=0.7)
    ax2.set_title("Median Annual Salary by Department & Gender (Equity Audit)", pad=15)
    ax2.set_ylabel("Median Annual Salary (INR)")
    ax2.set_xlabel("Department")
    ax2.legend(title="Gender", frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=20)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'gender_and_dept_equity_audit.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def export_model_metadata(feature_names, num_features, cat_features):
    """Export metadata JSON for seamless loading in Streamlit."""
    meta = {
        'feature_names': feature_names,
        'num_features': num_features,
        'cat_features': cat_features,
        'salary_tiers': ['Entry (<700k)', 'Mid (700k-1.2M)', 'Senior/Executive (>1.2M)'],
        'cluster_personas': {
            0: "Senior Veterans & Technical Leaders",
            1: "Early-Career High-Potential Associates",
            2: "Mid-Career Core Operations & Delivery"
        }
    }
    meta_path = os.path.join(SAVED_MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Saved Metadata: {meta_path}")


def main():
    print("="*70)
    print("STARTING TASK 8: ML MODELING, BENCHMARKING & REPORTING ASSETS")
    print("="*70)
    
    df, X_df, feature_names, num_features, cat_features = load_and_engineer_features()
    results_df, trained_models, X_test, y_test, target_names, feature_names, X_train, y_train = benchmark_classification_models(df, X_df, feature_names)
    reg_results_df = benchmark_regression_models(df, X_df)
    
    plot_model_comparison(results_df)
    plot_confusion_matrices_grid(trained_models, X_test, y_test, target_names)
    plot_roc_auc_curves(trained_models, X_test, y_test, target_names)
    plot_feature_importances(trained_models, feature_names)
    plot_salary_tier_and_equity_audit(df)
    export_model_metadata(feature_names, num_features, cat_features)
    
    print("\nTASK 8 ASSETS GENERATED SUCCESSFULLY! Stored in:", OUTPUT_DIR)
    print("="*70)


if __name__ == '__main__':
    main()
