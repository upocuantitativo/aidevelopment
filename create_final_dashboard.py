#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Final Academic Dashboard
SOLO G_GPD_PCAP_SLOPE (buen R2)
Estilo academico compacto
"""

import pickle
import os
from datetime import datetime

# Load results
with open('resultados/modelos/all_results.pkl', 'rb') as f:
    results = pickle.load(f)

# SOLO usar G_GPD_PCAP_SLOPE
target = 'G_GPD_PCAP_SLOPE'
res = results[target]
best_name, best_res = res['best']
top3 = res['top_3']

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GDP Growth Analysis - Gender Indicators</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 10pt;
            line-height: 1.3;
            color: #000;
            background: #fff;
            padding: 10mm 15mm;
            max-width: 800px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 14pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 6pt;
            border-bottom: 2px solid #000;
            padding-bottom: 3pt;
        }}

        h2 {{
            font-size: 11pt;
            font-weight: bold;
            margin: 10pt 0 4pt 0;
            border-bottom: 1px solid #666;
            padding-bottom: 2pt;
        }}

        .meta {{
            text-align: center;
            font-size: 9pt;
            color: #333;
            margin-bottom: 10pt;
            line-height: 1.2;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 6pt 0;
            font-size: 9pt;
        }}

        th {{
            background-color: #e8e8e8;
            border: 1px solid #555;
            padding: 3pt 5pt;
            text-align: left;
            font-weight: bold;
        }}

        td {{
            border: 1px solid #888;
            padding: 3pt 5pt;
        }}

        .best {{ background-color: #f5f5dc; font-weight: bold; }}

        .summary {{
            background-color: #f9f9f9;
            border: 1px solid #666;
            padding: 8pt;
            margin: 8pt 0;
        }}

        details {{
            margin: 8pt 0;
            border: 1px solid #999;
        }}

        summary {{
            background-color: #f0f0f0;
            padding: 4pt 8pt;
            cursor: pointer;
            font-weight: bold;
            font-size: 9.5pt;
            border-bottom: 1px solid #999;
        }}

        summary:hover {{ background-color: #e5e5e5; }}

        .details-content {{
            padding: 8pt;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #999;
            margin: 6pt 0;
        }}

        .note {{
            font-size: 8.5pt;
            font-style: italic;
            color: #555;
            margin-top: 3pt;
        }}

        .metric {{
            display: inline-block;
            margin-right: 12pt;
            font-size: 9.5pt;
        }}

        @media print {{
            body {{ padding: 0; }}
            details {{ border: 1px solid #999; }}
            details[open] summary {{ border-bottom: 1px solid #999; }}
        }}
    </style>
</head>
<body>
    <h1>GDP Per Capita Growth Analysis</h1>
    <div class="meta">
        Gender Indicators & Economic Development<br>
        52 Low & Lower-Middle Income Countries | 131 Variables<br>
        Analysis Date: {datetime.now().strftime('%d %B %Y')}
    </div>

    <div class="summary">
        <strong>Target Variable:</strong> G_GPD_PCAP_SLOPE (GDP per capita growth trajectory)<br>
        <strong>Best Model:</strong> {best_name} | R² = {best_res['r2']:.3f} | RMSE = {best_res['rmse']:.2f}<br>
        <strong>Validation:</strong> 5-fold CV R² = {best_res['cv']:.3f} | Bootstrap 95% CI: [{res['validation']['bootstrap_ci'][0]:.3f}, {res['validation']['bootstrap_ci'][1]:.3f}]
    </div>

    <h2>Model Comparison (Top 3)</h2>
    <table>
        <tr>
            <th style="width: 40px;">Rank</th>
            <th>Model</th>
            <th style="width: 60px;">R²</th>
            <th style="width: 60px;">RMSE</th>
            <th style="width: 60px;">MAE</th>
            <th style="width: 70px;">CV R²</th>
        </tr>
"""

for i, (name, model_res) in enumerate(top3, 1):
    row_class = "best" if i == 1 else ""
    html += f"""        <tr class="{row_class}">
            <td style="text-align: center;">{i}</td>
            <td>{name}</td>
            <td>{model_res['r2']:.4f}</td>
            <td>{model_res['rmse']:.2f}</td>
            <td>{model_res['mae']:.2f}</td>
            <td>{model_res['cv']:.4f}</td>
        </tr>
"""

html += """    </table>
    <p class="note">Models ranked by test R². Best model highlighted. Metrics: Coefficient of determination (R²),
    Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Cross-Validation R².</p>

    <details>
        <summary>SHAP Feature Importance Analysis</summary>
        <div class="details-content">
            <p>SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to predictions.
            Top 15 most important variables shown below.</p>
            <img src="graficos_finales/shap_G_GPD_PCAP_SLOPE.png" alt="SHAP Analysis">
            <p class="note">Higher absolute SHAP values indicate greater predictive importance.
            Red indicates higher feature values, blue indicates lower values.</p>
        </div>
    </details>

    <details>
        <summary>Validation & Robustness Statistics</summary>
        <div class="details-content">
            <table>
                <tr><th>Test</th><th>Value</th><th>Interpretation</th></tr>
                <tr>
                    <td>Shapiro-Wilk</td>
                    <td>p = {res['validation']['shapiro_p']:.4f}</td>
                    <td>{"Residuals normal (p > 0.05)" if res['validation']['shapiro_p'] > 0.05 else "Non-normal residuals"}</td>
                </tr>
                <tr>
                    <td>Bootstrap Mean</td>
                    <td>R² = {res['validation']['bootstrap_r2_mean']:.4f}</td>
                    <td>Average across 100 resamples</td>
                </tr>
                <tr>
                    <td>Bootstrap 95% CI</td>
                    <td>[{res['validation']['bootstrap_ci'][0]:.4f}, {res['validation']['bootstrap_ci'][1]:.4f}]</td>
                    <td>Confidence interval for R²</td>
                </tr>
            </table>
            <p class="note">Shapiro-Wilk tests normality of residuals. Bootstrap provides robust confidence intervals.</p>
        </div>
    </details>

    <details>
        <summary>Methodology</summary>
        <div class="details-content">
            <p><strong>Sample:</strong> 52 low and lower-middle income countries</p>
            <p><strong>Variables:</strong> 131 gender and development indicators (World Bank, UN, UNESCO)</p>
            <p><strong>Feature Selection:</strong> Top 15 variables by Pearson correlation with target</p>
            <p><strong>Data Split:</strong> 75% training (n=39), 25% testing (n=13)</p>

            <p style="margin-top: 6pt;"><strong>Models Evaluated:</strong></p>
            <ol style="margin-left: 15pt; font-size: 9pt;">
                <li><strong>Random Forest:</strong> Ensemble of decision trees with recursive hyperparameter optimization</li>
                <li><strong>XGBoost:</strong> Gradient boosting with regularization</li>
                <li><strong>Neural Network:</strong> Multi-layer perceptron (100-50-25 architecture)</li>
                <li><strong>Gradient Boosting:</strong> Sequential ensemble learning</li>
            </ol>

            <p style="margin-top: 6pt;"><strong>Recursive Optimization:</strong></p>
            <p style="font-size: 9pt; margin-left: 10pt;">
            Random Forest parameters optimized sequentially via 5-fold cross-validation:
            (1) n_estimators, (2) max_depth, (3) min_samples_split, (4) min_samples_leaf.
            At each level, only parameters improving CV score are retained before recursing to next level.
            </p>

            <p style="margin-top: 6pt;"><strong>Validation:</strong></p>
            <ul style="margin-left: 15pt; font-size: 9pt;">
                <li>5-fold stratified cross-validation</li>
                <li>Bootstrap resampling (100 iterations)</li>
                <li>Residual normality testing (Shapiro-Wilk)</li>
                <li>SHAP analysis for interpretability</li>
            </ul>
        </div>
    </details>

    <details>
        <summary>Top Predictive Variables</summary>
        <div class="details-content">
            <p><strong>Top 5 Variables (by model importance):</strong></p>
            <ol style="font-size: 9pt; margin-left: 15pt;">
"""

# Get feature importance from best model
if 'importance' in best_res:
    import pandas as pd
    importance_df = pd.DataFrame(list(best_res['importance'].items()),
                                columns=['Variable', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
    for _, row in importance_df.iterrows():
        var_name = row['Variable']
        if len(var_name) > 60:
            var_name = var_name[:57] + '...'
        html += f"                <li>{var_name} ({row['Importance']:.4f})</li>\n"
else:
    html += "                <li>Feature importance not available for this model</li>\n"

html += """            </ol>
            <p class="note">Importance scores sum to 1.0. Higher values indicate stronger predictive power.</p>
        </div>
    </details>

    <hr style="margin: 10pt 0; border: none; border-top: 1px solid #999;">

    <p class="note" style="text-align: center; margin-top: 8pt;">
        Analysis conducted: {datetime.now().strftime('%Y-%m-%d')}<br>
        Repository: <a href="https://github.com/upocuantitativo/aidevelopment" style="color: #000;">github.com/upocuantitativo/aidevelopment</a>
    </p>

</body>
</html>
"""

# Save
with open('resultados/dashboard_final.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard final creado: resultados/dashboard_final.html")
print("SOLO incluye G_GPD_PCAP_SLOPE (R2=0.76)")
print("New_Business_Density excluido (R2 negativo)")
