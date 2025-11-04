#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Improved SHAP Analysis with Bar Plot and Detailed Importance
Similar style to academic papers
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Set academic plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 8

# Load data and results
df = pd.read_excel('DATA_GHAB2.xlsx')
with open('resultados/modelos/all_results.pkl', 'rb') as f:
    results = pickle.load(f)

target = 'G_GPD_PCAP_SLOPE'
res = results[target]
best_name, best_res = res['best']
predictors = res['predictors']

# Prepare data
mask = df[target].notna()
for var in predictors:
    mask &= df[var].notna()

X = df.loc[mask, predictors]
y = df.loc[mask, target].values

# Get model
model = best_res['model']

print(f"Generating improved SHAP analysis for {best_name}...")

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Get mean absolute SHAP values for each feature
shap_importance = np.abs(shap_values).mean(axis=0)
feature_names = X.columns.tolist()

# Create DataFrame with importance
importance_df = pd.DataFrame({
    'Variable': feature_names,
    'Importance': shap_importance
}).sort_values('Importance', ascending=True)  # Ascending for horizontal bars

# Take top 15
top_15 = importance_df.tail(15)

# Create figure with better spacing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.06, wspace=0.35)

# LEFT PLOT: Bar plot of SHAP importance (similar to liñan style)
y_pos = np.arange(len(top_15))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_15)))

bars = ax1.barh(y_pos, top_15['Importance'],
               color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

# Truncate long variable names for display - MORE AGGRESSIVE
var_labels = []
for var in top_15['Variable']:
    if len(var) > 45:
        var_labels.append(var[:42] + '...')
    else:
        var_labels.append(var)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(var_labels, fontsize=7)
ax1.set_xlabel('Mean |SHAP Value|',
              fontsize=9, fontweight='bold')
ax1.set_title('Top 15 Features by SHAP Importance',
             fontsize=10, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='x')

# Add values on bars
for i, (idx, val) in enumerate(zip(y_pos, top_15['Importance'])):
    ax1.text(val + val*0.02, idx, f'{val:.2f}',
            va='center', fontsize=7, fontweight='bold')

# RIGHT PLOT: SHAP beeswarm plot (top 15)
# Get indices of top 15 features
top_15_names = top_15['Variable'].tolist()
top_15_indices = [i for i, name in enumerate(feature_names) if name in top_15_names]

# Create subset of SHAP values and data for top 15
shap_values_top15 = shap_values[:, top_15_indices]
X_top15 = X.iloc[:, top_15_indices]

# Reorder to match top 15 order (descending importance)
reorder_indices = [top_15_names.index(name) for name in X_top15.columns]
inv_reorder = [reorder_indices.index(i) for i in range(len(reorder_indices))]
shap_values_top15 = shap_values_top15[:, inv_reorder]
X_top15 = X_top15.iloc[:, inv_reorder]

# Rename columns to short names
X_top15_renamed = X_top15.copy()
X_top15_renamed.columns = var_labels[::-1]  # Reverse to match SHAP order

# Create beeswarm plot
shap.summary_plot(shap_values_top15, X_top15_renamed,
                 plot_type="dot", show=False, max_display=15,
                 plot_size=(8, 8))

# Customize the plot
ax2 = plt.gca()
ax2.set_xlabel('SHAP Value', fontsize=9, fontweight='bold')
ax2.set_title('SHAP Values Distribution\n(Red = High, Blue = Low)',
             fontsize=10, fontweight='bold', pad=10)
# Adjust y-tick labels
ax2.tick_params(axis='y', labelsize=7)
ax2.tick_params(axis='x', labelsize=7)
plt.savefig('resultados/graficos_finales/shap_G_GPD_PCAP_SLOPE.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("OK Saved: resultados/graficos_finales/shap_G_GPD_PCAP_SLOPE.png")

plt.close()

# Create a summary table CSV
top_15_reversed = importance_df.tail(15).sort_values('Importance', ascending=False)
top_15_reversed.to_csv('resultados/top_15_shap_importance.csv', index=False)
print("OK Saved: resultados/top_15_shap_importance.csv")

print("\nTop 15 Variables by SHAP Importance:")
for i, (idx, row) in enumerate(top_15_reversed.iterrows(), 1):
    var_name = row['Variable'][:70]
    print(f"  {i:2d}. {var_name:70s} {row['Importance']:.4f}")
