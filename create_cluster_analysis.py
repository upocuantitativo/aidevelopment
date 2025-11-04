#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster Analysis with Country Labels and Ellipses
Visualize country groupings based on predictive variables
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Set academic plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11

# Load data and results
df = pd.read_excel('DATA_GHAB2.xlsx')
with open('resultados/modelos/all_results.pkl', 'rb') as f:
    results = pickle.load(f)

target = 'G_GPD_PCAP_SLOPE'
res = results[target]
predictors = res['predictors']

# Prepare data - only complete cases
mask = df[target].notna()
for var in predictors:
    mask &= df[var].notna()

df_clean = df[mask].copy()
X = df_clean[predictors].values

# Check if Country column exists
country_col = None
for col in ['Country', 'country', 'Country Name', 'COUNTRY']:
    if col in df_clean.columns:
        country_col = col
        break

if country_col is None:
    # Try to use index
    countries = df_clean.index.tolist()
    print("Warning: No country column found, using index")
else:
    countries = df_clean[country_col].tolist()

print(f"Analyzing {len(countries)} countries")
print(f"Using {len(predictors)} predictive variables")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 8)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Choose k=4 as a reasonable balance
optimal_k = 4
print(f"\nUsing {optimal_k} clusters")

# Perform clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

# Reduce to 2D for visualization using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08, wspace=0.25)

# Define colors for clusters
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']

# Function to draw confidence ellipse
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# PLOT 1: Clusters with ellipses and country labels
for cluster_id in range(optimal_k):
    mask_cluster = clusters == cluster_id
    x_cluster = X_pca[mask_cluster, 0]
    y_cluster = X_pca[mask_cluster, 1]

    # Plot points
    ax1.scatter(x_cluster, y_cluster,
               c=colors[cluster_id], s=80, alpha=0.7,
               edgecolors='black', linewidth=0.5,
               label=f'Cluster {cluster_id+1}', zorder=2)

    # Draw ellipse
    if len(x_cluster) > 2:
        confidence_ellipse(x_cluster, y_cluster, ax1, n_std=2,
                         edgecolor=colors[cluster_id], linewidth=2.5,
                         facecolor=colors[cluster_id], alpha=0.15, zorder=1)

# Add country labels
for i, (x, y, country, cluster) in enumerate(zip(X_pca[:, 0], X_pca[:, 1], countries, clusters)):
    # Truncate long country names
    country_short = str(country)[:15]
    ax1.annotate(country_short, (x, y),
                fontsize=6, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=colors[cluster], alpha=0.7, linewidth=0.5))

ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
              fontweight='bold')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
              fontweight='bold')
ax1.set_title('Country Clusters Based on Gender & Development Indicators',
             fontweight='bold', fontsize=12)
ax1.legend(loc='best', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# PLOT 2: Cluster characteristics (heatmap of means)
cluster_means = []
for cluster_id in range(optimal_k):
    mask_cluster = clusters == cluster_id
    cluster_mean = X_scaled[mask_cluster].mean(axis=0)
    cluster_means.append(cluster_mean)

cluster_means = np.array(cluster_means)

# Select top 10 most important variables for heatmap
top_10_vars = predictors[:10]
top_10_indices = [predictors.index(var) for var in top_10_vars]
cluster_means_top10 = cluster_means[:, top_10_indices]

# Truncate variable names
var_names_short = [var[:30] + '...' if len(var) > 30 else var for var in top_10_vars]

im = ax2.imshow(cluster_means_top10.T, cmap='RdBu_r', aspect='auto',
               vmin=-2, vmax=2)
ax2.set_xticks(range(optimal_k))
ax2.set_xticklabels([f'C{i+1}' for i in range(optimal_k)], fontsize=10)
ax2.set_yticks(range(len(top_10_vars)))
ax2.set_yticklabels(var_names_short, fontsize=7)
ax2.set_xlabel('Cluster', fontweight='bold')
ax2.set_title('Cluster Characteristics\n(Top 10 Variables, Standardized)',
             fontweight='bold', fontsize=11)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
cbar.set_label('Standardized Value', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Add values to heatmap
for i in range(len(top_10_vars)):
    for j in range(optimal_k):
        text = ax2.text(j, i, f'{cluster_means_top10[j, i]:.1f}',
                       ha="center", va="center", color="black", fontsize=7)

# Save figure
plt.savefig('resultados/graficos_finales/cluster_analysis.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("OK Saved: resultados/graficos_finales/cluster_analysis.png")

# Save cluster assignments
cluster_df = pd.DataFrame({
    'Country': countries,
    'Cluster': clusters + 1,  # 1-indexed for readability
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})
cluster_df = cluster_df.sort_values('Cluster')
cluster_df.to_csv('resultados/cluster_assignments.csv', index=False)
print("OK Saved: resultados/cluster_assignments.csv")

# Print cluster summaries
print("\n" + "="*60)
print("CLUSTER ANALYSIS SUMMARY")
print("="*60)
for cluster_id in range(optimal_k):
    mask_cluster = clusters == cluster_id
    countries_in_cluster = [c for c, m in zip(countries, mask_cluster) if m]
    print(f"\nCluster {cluster_id + 1} ({len(countries_in_cluster)} countries):")
    print("  " + ", ".join([str(c)[:20] for c in countries_in_cluster[:8]]))
    if len(countries_in_cluster) > 8:
        print(f"  ... and {len(countries_in_cluster) - 8} more")

print("\n" + "="*60)
print("CLUSTER INTERPRETATION")
print("="*60)
print("Clusters represent groups of countries with similar patterns in:")
print("  - Gender indicators")
print("  - Health outcomes")
print("  - Education metrics")
print("  - Development indicators")
