"""
Task 7: Cluster Visualization & Dimensionality Reduction Studio
SalaryInsight: Predictive Modeling for Informed Salary Predictions
"""

import os
import json
from typing import cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.projections.polar import PolarAxes
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set aesthetic styling for matplotlib and seaborn
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300

DATA_PATH = '/home/user/data/statistics_practice_dataset_2000.csv'
OUTPUT_DIR = '/home/user/visualizations/task_7'
SAVED_MODELS_DIR = '/home/user/saved_models'
DATA_OUTPUT_DIR = '/home/user/data'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Load dataset and extract numerical features for clustering."""
    print("Loading dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Feature selection for clustering
    feature_cols = [
        'Age', 'ExperienceYears', 'AnnualSalary', 'PerformanceScore',
        'WeeklyHours', 'ProjectsCompleted', 'SatisfactionScore',
        'CommuteKm', 'BMI', 'MonthlySalesK', 'DiscountPercent'
    ]
    
    X = df[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for future deployment
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, 'scaler.joblib'))
    
    return df, X, X_scaled, feature_cols, scaler


def evaluate_cluster_numbers(X_scaled):
    """Compute Elbow (Inertia), Silhouette, and Davies-Bouldin across k=2..8."""
    print("Evaluating optimal cluster count k...")
    k_range = list(range(2, 9))
    inertias = []
    silhouettes = []
    davies_bouldins = []
    calinskis = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        davies_bouldins.append(davies_bouldin_score(X_scaled, labels))
        calinskis.append(calinski_harabasz_score(X_scaled, labels))
        
    # Plot Elbow and Silhouette analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot 1: Inertia / Elbow Method
    ax1.plot(k_range, inertias, marker='o', linewidth=2.5, color='#1f77b4', markersize=8)
    ax1.set_title("Elbow Method (Inertia vs. Number of Clusters)", pad=15)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axvline(x=3, color='#e74c3c', linestyle=':', label='Optimal k = 3')
    ax1.legend(frameon=True)
    
    # Subplot 2: Silhouette Score
    ax2.plot(k_range, silhouettes, marker='s', linewidth=2.5, color='#2ca02c', markersize=8)
    ax2.set_title("Silhouette Score vs. Number of Clusters", pad=15)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Coefficient")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.axvline(x=3, color='#e74c3c', linestyle=':', label='Peak Silhouette (k = 3)')
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'elbow_and_silhouette_analysis.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")
    
    return k_range, inertias, silhouettes


def fit_clusters_and_personas(df, X_scaled, n_clusters=3):
    """Fit optimal KMeans and assign human-interpretable HR Personas."""
    print(f"Fitting KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = cluster_labels
    
    # Analyze cluster centroids to label personas
    cluster_means = df.groupby('Cluster')[['ExperienceYears', 'AnnualSalary', 'PerformanceScore', 'WeeklyHours', 'MonthlySalesK', 'SatisfactionScore']].mean()
    print("\nCluster Means Summary:\n", cluster_means)
    
    # Define Persona Mapping
    persona_map = {
        0: "Senior Technical & Ops Specialists",
        1: "High-Impact Revenue Drivers",
        2: "Core Growth & Emerging Associates"
    }
    
    df['ClusterPersona'] = df['Cluster'].map(persona_map)
    
    # Save model
    joblib.dump(kmeans, os.path.join(SAVED_MODELS_DIR, 'kmeans_cluster_model.joblib'))
    
    return kmeans, df, persona_map


def compute_dimensionality_reduction(X_scaled, df):
    """Apply PCA (2D & 3D) and t-SNE."""
    print("Computing PCA and t-SNE...")
    # PCA
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    df['PCA1'] = X_pca_2d[:, 0]
    df['PCA2'] = X_pca_2d[:, 1]
    
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    df['PCA3'] = X_pca_3d[:, 2]
    
    joblib.dump(pca_2d, os.path.join(SAVED_MODELS_DIR, 'pca_model.joblib'))
    joblib.dump(pca_3d, os.path.join(SAVED_MODELS_DIR, 'pca_3d_model.joblib'))
    
    print(f"PCA 2D Explained Variance: {pca_2d.explained_variance_ratio_}, Total: {np.sum(pca_2d.explained_variance_ratio_):.2%}")
    print(f"PCA 3D Explained Variance: {pca_3d.explained_variance_ratio_}, Total: {np.sum(pca_3d.explained_variance_ratio_):.2%}")
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    df['tSNE1'] = X_tsne[:, 0]
    df['tSNE2'] = X_tsne[:, 1]
    
    return pca_2d, pca_3d, X_pca_2d, X_pca_3d, X_tsne


def plot_pca_2d_clusters(df, pca_2d, persona_map):
    """Plot 2D PCA cluster projection with centroids and ellipses."""
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for cluster_id, color in zip(sorted(df['Cluster'].unique()), colors):
        subset = df[df['Cluster'] == cluster_id]
        ax.scatter(
            subset['PCA1'], subset['PCA2'],
            c=color, label=f"Cluster {cluster_id}: {persona_map[cluster_id]} (n={len(subset)})",
            alpha=0.65, edgecolors='w', s=55
        )
        
        # Calculate cluster centroid in PCA space
        centroid_x = subset['PCA1'].mean()
        centroid_y = subset['PCA2'].mean()
        ax.scatter(
            centroid_x, centroid_y,
            c='yellow', edgecolors='black', s=220, marker='*',
            zorder=10
        )
        ax.annotate(
            f"Centroid C{cluster_id}",
            (centroid_x, centroid_y),
            xytext=(10, 10), textcoords='offset points',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor=color)
        )
    
    var_exp = pca_2d.explained_variance_ratio_
    ax.set_title("2D PCA Cluster Projection - Employee Segmentation", pad=15, fontsize=15)
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1%} Explained Variance)")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1%} Explained Variance)")
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'pca_2d_clusters.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_pca_3d_clusters(df, pca_3d, persona_map):
    """Plot static 3D PCA projection."""
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for cluster_id, color in zip(sorted(df['Cluster'].unique()), colors):
        subset = df[df['Cluster'] == cluster_id]
        ax.scatter(
            subset['PCA1'], subset['PCA2'], subset['PCA3'],
            c=color, label=f"Cluster {cluster_id}: {persona_map[cluster_id]}",
            alpha=0.6, edgecolors='w', s=45
        )
    
    var_exp = pca_3d.explained_variance_ratio_
    ax.set_title("3D PCA Cluster Projection - Employee Feature Space", pad=20, fontsize=15)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})", labelpad=10)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})", labelpad=10)
    ax.set_zlabel(f"PC3 ({var_exp[2]:.1%})", labelpad=10)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.view_init(elev=22, azim=130)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'pca_3d_clusters.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_tsne_2d_clusters(df, persona_map):
    """Plot 2D t-SNE cluster visualization."""
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for cluster_id, color in zip(sorted(df['Cluster'].unique()), colors):
        subset = df[df['Cluster'] == cluster_id]
        ax.scatter(
            subset['tSNE1'], subset['tSNE2'],
            c=color, label=f"Cluster {cluster_id}: {persona_map[cluster_id]} (n={len(subset)})",
            alpha=0.65, edgecolors='w', s=55
        )
    
    ax.set_title("2D t-SNE Non-Linear Manifold Cluster Visualization", pad=15, fontsize=15)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'tsne_2d_clusters.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_pca_feature_loadings_biplot(pca_2d, feature_cols):
    """Plot PCA Biplot with feature loading vectors."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
    
    # Plot feature vectors
    for i, feature in enumerate(feature_cols):
        ax.arrow(
            0, 0, loadings[i, 0], loadings[i, 1],
            color='#d62728', alpha=0.85, width=0.015, head_width=0.04
        )
        ax.text(
            loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
            feature, color='#111111', fontsize=11, fontweight='bold',
            ha='center', va='center'
        )
        
    circle = patches.Circle((0, 0), radius=1.0, color='#1f77b4', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(circle)
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
    ax.set_title("PCA Feature Loadings & Correlation Biplot (PC1 vs PC2)", pad=15, fontsize=15)
    ax.set_xlabel(f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'pca_feature_loadings_biplot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_feature_space_scatter_matrix(df, persona_map):
    """Plot 4-panel feature-space scatter plots showing key operational metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    pairs = [
        ('ExperienceYears', 'AnnualSalary', 'Experience (Years)', 'Annual Salary (INR)'),
        ('PerformanceScore', 'AnnualSalary', 'Performance Score (0-100)', 'Annual Salary (INR)'),
        ('WeeklyHours', 'ProjectsCompleted', 'Weekly Hours (Hrs/Wk)', 'Projects Completed'),
        ('MonthlySalesK', 'AnnualSalary', 'Monthly Sales (₹ Thousands)', 'Annual Salary (INR)')
    ]
    
    for ax, (x_col, y_col, x_label, y_label) in zip(axes.flat, pairs):
        for cluster_id, color in zip(sorted(df['Cluster'].unique()), colors):
            subset = df[df['Cluster'] == cluster_id]
            ax.scatter(
                subset[x_col], subset[y_col],
                c=color, label=f"C{cluster_id}: {persona_map[cluster_id][:15]}...",
                alpha=0.6, edgecolors='w', s=45
            )
        ax.set_title(f"{x_label} vs. {y_label}", fontsize=13)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc='best', fontsize=9, frameon=True)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.suptitle("Feature-Space Multi-Panel Scatter Plots by Employee Cluster", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'feature_space_scatter_matrix.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_annotated_decision_boundaries(df, X_pca_2d, persona_map):
    """Train classifier on 2D PCA space and plot annotated decision boundaries with zone callouts."""
    print("Generating Annotated Decision Boundary visualization...")
    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(X_pca_2d, df['Cluster'])
    
    # Mesh grid
    x_min, x_max = X_pca_2d[:, 0].min() - 0.8, X_pca_2d[:, 0].max() + 0.8
    y_min, y_max = X_pca_2d[:, 1].min() - 0.8, X_pca_2d[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.04), np.arange(y_min, y_max, 0.04))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(13, 10))
    cmap_light = ListedColormap(['#dbeafe', '#dcfce7', '#fee2e2'])
    colors = ['#1d4ed8', '#15803d', '#b91c1c']
    
    # Decision boundaries contour fill
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.contour(xx, yy, Z, colors='black', linewidths=1.2, linestyles='--')
    
    # Plot employee data points
    for cluster_id, color in zip(sorted(df['Cluster'].unique()), colors):
        subset = df[df['Cluster'] == cluster_id]
        ax.scatter(
            subset['PCA1'], subset['PCA2'],
            c=color, label=f"Cluster {cluster_id}: {persona_map[cluster_id]}",
            alpha=0.7, edgecolors='w', s=50
        )
        
        # Centroids
        cx, cy = subset['PCA1'].mean(), subset['PCA2'].mean()
        ax.scatter(cx, cy, c='gold', edgecolors='black', s=260, marker='*', zorder=12)
        
        # Decision Zone Annotation Callout
        ax.text(
            cx, cy + 0.45,
            f"★ ZONE {cluster_id}\n{persona_map[cluster_id]}\n(Avg Salary: ₹{subset['AnnualSalary'].mean():,.0f})",
            fontsize=10, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor=color, linewidth=1.5),
            zorder=15
        )
        
    ax.set_title("Annotated Decision Boundaries in 2D PCA Latent Space", pad=15, fontsize=15)
    ax.set_xlabel("Principal Component 1 (Primary Variance Direction)")
    ax.set_ylabel("Principal Component 2 (Secondary Variance Direction)")
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'annotated_decision_boundaries_2d.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_cluster_radar_profiles(df, feature_cols, scaler, persona_map):
    """Generate Radar Chart comparing cluster centroid metrics across dimensions."""
    print("Generating Cluster Radar Profiles...")
    features_to_radar = ['Age', 'ExperienceYears', 'AnnualSalary', 'PerformanceScore', 'WeeklyHours', 'ProjectsCompleted', 'SatisfactionScore', 'MonthlySalesK']
    
    # Normalized feature averages per cluster (0 to 1 scale)
    radar_df = pd.DataFrame()
    for col in features_to_radar:
        min_val = df[col].min()
        max_val = df[col].max()
        radar_df[col] = (df[col] - min_val) / (max_val - min_val)
    radar_df['Cluster'] = df['Cluster']
    
    cluster_radar_means = radar_df.groupby('Cluster').mean().reset_index()
    
    categories = features_to_radar
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax = cast(PolarAxes, ax)
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for i, row in cluster_radar_means.iterrows():
        cluster_id = int(row['Cluster'])
        values = [float(row[category]) for category in categories]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=f"C{cluster_id}: {persona_map[cluster_id]}", color=colors[cluster_id])
        ax.fill(angles, values, color=colors[cluster_id], alpha=0.18)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title("Standardized Cluster Persona Profiles (Multi-Attribute Radar)", size=15, weight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, shadow=True)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'cluster_radar_profiles.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def export_interactive_plotly_charts(df, persona_map):
    """Export interactive 3D and 2D Plotly HTML visualizers."""
    print("Exporting interactive HTML Plotly visualizations...")
    # Interactive 3D PCA
    fig_3d = px.scatter_3d(
        df, x='PCA1', y='PCA2', z='PCA3',
        color='ClusterPersona',
        hover_data=['EmployeeID', 'Department', 'City', 'AnnualSalary', 'ExperienceYears', 'PerformanceScore'],
        title="SalaryInsight: Interactive 3D PCA Employee Cluster Explorer",
        color_discrete_sequence=['#1f77b4', '#2ca02c', '#d62728']
    )
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title="Cluster Persona", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    html_3d_path = os.path.join(OUTPUT_DIR, 'interactive_pca_3d.html')
    fig_3d.write_html(html_3d_path)
    print(f"Saved: {html_3d_path}")
    
    # Interactive 2D Explorer
    fig_2d = px.scatter(
        df, x='PCA1', y='PCA2',
        color='ClusterPersona',
        size='AnnualSalary',
        hover_data=['EmployeeID', 'Department', 'AnnualSalary', 'WeeklyHours', 'ProjectsCompleted'],
        title="SalaryInsight: Interactive 2D PCA Cluster & Compensation Explorer",
        color_discrete_sequence=['#1f77b4', '#2ca02c', '#d62728']
    )
    html_2d_path = os.path.join(OUTPUT_DIR, 'interactive_cluster_explorer.html')
    fig_2d.write_html(html_2d_path)
    print(f"Saved: {html_2d_path}")


def export_summary_profiles(df, persona_map):
    """Export comprehensive cluster profile summaries to CSV."""
    print("Exporting cluster profiling dataset...")
    # Save full dataset with cluster & dimensionality reduction columns
    full_output_path = os.path.join(DATA_OUTPUT_DIR, 'processed_data_with_clusters.csv')
    df.to_csv(full_output_path, index=False)
    print(f"Saved: {full_output_path}")
    
    # Generate aggregated profiling table
    summary = df.groupby(['Cluster', 'ClusterPersona']).agg(
        Headcount=('EmployeeID', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Experience=('ExperienceYears', 'mean'),
        Avg_AnnualSalary=('AnnualSalary', 'mean'),
        Median_AnnualSalary=('AnnualSalary', 'median'),
        Std_AnnualSalary=('AnnualSalary', 'std'),
        Avg_Performance=('PerformanceScore', 'mean'),
        Avg_WeeklyHours=('WeeklyHours', 'mean'),
        Avg_Projects=('ProjectsCompleted', 'mean'),
        Avg_Satisfaction=('SatisfactionScore', 'mean'),
        Avg_MonthlySalesK=('MonthlySalesK', 'mean'),
        Avg_CommuteKm=('CommuteKm', 'mean')
    ).reset_index()
    
    summary['Headcount_Share_%'] = (summary['Headcount'] / len(df) * 100).round(2)
    summary_path = os.path.join(DATA_OUTPUT_DIR, 'cluster_profiles_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print("\n=== Cluster Profile Summary Table ===\n", summary)


def main():
    print("="*70)
    print("STARTING TASK 7: CLUSTER VISUALIZATION & DIMENSIONALITY REDUCTION")
    print("="*70)
    
    df, X, X_scaled, feature_cols, scaler = load_and_preprocess_data()
    k_range, inertias, silhouettes = evaluate_cluster_numbers(X_scaled)
    kmeans, df, persona_map = fit_clusters_and_personas(df, X_scaled, n_clusters=3)
    pca_2d, pca_3d, X_pca_2d, X_pca_3d, X_tsne = compute_dimensionality_reduction(X_scaled, df)
    
    plot_pca_2d_clusters(df, pca_2d, persona_map)
    plot_pca_3d_clusters(df, pca_3d, persona_map)
    plot_tsne_2d_clusters(df, persona_map)
    plot_pca_feature_loadings_biplot(pca_2d, feature_cols)
    plot_feature_space_scatter_matrix(df, persona_map)
    plot_annotated_decision_boundaries(df, X_pca_2d, persona_map)
    plot_cluster_radar_profiles(df, feature_cols, scaler, persona_map)
    export_interactive_plotly_charts(df, persona_map)
    export_summary_profiles(df, persona_map)
    
    print("\nTASK 7 COMPLETED SUCCESSFULLY! All figures and data exported to:", OUTPUT_DIR)
    print("="*70)


if __name__ == '__main__':
    main()
