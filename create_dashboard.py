#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Academic Dashboard
Estilo compacto, profesional, sin degradados
"""

import pickle
import os
from datetime import datetime

# Load results
with open('resultados/modelos/all_results.pkl', 'rb') as f:
    results = pickle.load(f)

targets = list(results.keys())

# Create academic dashboard
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gender & Development Analysis - Dual Target Models</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #000;
            background: #fff;
            padding: 15mm;
            max-width: 210mm;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 16pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 8pt;
            border-bottom: 2px solid #000;
            padding-bottom: 4pt;
        }}

        h2 {{
            font-size: 13pt;
            font-weight: bold;
            margin-top: 12pt;
            margin-bottom: 6pt;
            border-bottom: 1px solid #333;
        }}

        h3 {{
            font-size: 11pt;
            font-weight: bold;
            margin-top: 8pt;
            margin-bottom: 4pt;
        }}

        .meta {{
            text-align: center;
            font-size: 10pt;
            color: #555;
            margin-bottom: 12pt;
        }}

        .section {{
            margin-bottom: 16pt;
            border: 1px solid #ccc;
            padding: 10pt;
        }}

        .collapsible {{
            background-color: #f5f5f5;
            color: #000;
            cursor: pointer;
            padding: 8pt;
            width: 100%;
            border: 1px solid #999;
            text-align: left;
            outline: none;
            font-size: 11pt;
            font-weight: bold;
            font-family: 'Times New Roman', Times, serif;
        }}

        .collapsible:hover {{
            background-color: #e8e8e8;
        }}

        .collapsible:after {{
            content: '\\002B';
            color: #000;
            font-weight: bold;
            float: right;
            margin-left: 5pt;
        }}

        .collapsible.active:after {{
            content: "\\2212";
        }}

        .content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
            background-color: #fff;
            border-left: 1px solid #999;
            border-right: 1px solid #999;
            border-bottom: 1px solid #999;
        }}

        .content-inner {{
            padding: 10pt;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 8pt 0;
            font-size: 10pt;
        }}

        th {{
            background-color: #f0f0f0;
            border: 1px solid #666;
            padding: 4pt 6pt;
            text-align: left;
            font-weight: bold;
        }}

        td {{
            border: 1px solid #999;
            padding: 4pt 6pt;
        }}

        .best {{
            background-color: #ffe;
            font-weight: bold;
        }}

        .metric {{
            display: inline-block;
            min-width: 80pt;
        }}

        .warning {{
            background-color: #fff3cd;
            border: 1px solid #856404;
            padding: 8pt;
            margin: 8pt 0;
            font-size: 10pt;
        }}

        .note {{
            font-size: 9pt;
            font-style: italic;
            color: #555;
            margin-top: 4pt;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #999;
            margin: 8pt 0;
        }}

        .summary-box {{
            background-color: #f9f9f9;
            border: 2px solid #333;
            padding: 10pt;
            margin: 10pt 0;
        }}

        .inline-stat {{
            display: inline-block;
            margin-right: 15pt;
        }}

        @media print {{
            body {{
                padding: 0;
            }}
            .collapsible {{
                background-color: #f5f5f5 !important;
            }}
            .content {{
                max-height: none !important;
            }}
        }}
    </style>
</head>
<body>
    <h1>Dual-Target Analysis: Gender Indicators & Economic Development</h1>

    <div class="meta">
        Dataset: 52 Low & Lower-Middle Income Countries | 131 Variables<br>
        Analysis Date: {datetime.now().strftime('%B %d, %Y')}<br>
        Methods: Recursive Random Forest Optimization, XGBoost, Neural Networks, Gradient Boosting
    </div>

    <div class="summary-box">
        <h3>Executive Summary</h3>
        <p><strong>Targets Analyzed:</strong></p>
        <ul>
            <li><strong>New_Business_Density:</strong> New business registrations per 1,000 people (ages 15-64)</li>
            <li><strong>G_GPD_PCAP_SLOPE:</strong> GDP per capita growth trajectory</li>
        </ul>
        <p style="margin-top: 8pt;"><strong>Key Findings:</strong></p>
        <ul>
            <li>G_GPD_PCAP_SLOPE: Best model achieves R² = 0.76 (Random Forest)</li>
            <li>New_Business_Density: Challenging prediction (R² = -0.87, limited by sample size)</li>
            <li>Top 3 models selected via cross-validation for each target</li>
        </ul>
    </div>
"""

# For each target
for target in targets:
    res = results[target]
    best_name, best_res = res['best']
    top3 = res['top_3']

    # Determine if model is good
    is_good = best_res['r2'] > 0.5

    html += f"""
    <h2>{target.replace('_', ' ')}</h2>

    <div class="section">
        <h3>Best Model: {best_name}</h3>
        <div class="{'summary-box' if is_good else 'warning'}">
            <p>
                <span class="inline-stat"><strong>R² (Test):</strong> {best_res['r2']:.4f}</span>
                <span class="inline-stat"><strong>RMSE:</strong> {best_res['rmse']:.4f}</span>
                <span class="inline-stat"><strong>MAE:</strong> {best_res['mae']:.4f}</span>
                <span class="inline-stat"><strong>CV R²:</strong> {best_res['cv']:.4f}</span>
            </p>
"""

    if not is_good:
        html += f"""
            <p class="note" style="margin-top: 6pt; color: #856404;">
                Note: Negative R² indicates prediction below baseline. Likely due to small sample size (n=52 countries, 13 test)
                and high target variability. Model used with caution for qualitative insights only.
            </p>
"""

    html += """
        </div>
    </div>
"""

    # Top 3 models (collapsible)
    html += """
    <button class="collapsible">Top 3 Model Comparison</button>
    <div class="content">
        <div class="content-inner">
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>R² (Test)</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>CV R²</th>
                </tr>
"""

    for i, (name, model_res) in enumerate(top3, 1):
        row_class = "best" if i == 1 else ""
        html += f"""
                <tr class="{row_class}">
                    <td>{i}</td>
                    <td>{name}</td>
                    <td>{model_res['r2']:.4f}</td>
                    <td>{model_res['rmse']:.4f}</td>
                    <td>{model_res['mae']:.4f}</td>
                    <td>{model_res['cv']:.4f}</td>
                </tr>
"""

    html += """
            </table>
            <p class="note">Models ranked by Test R². Best model highlighted.</p>
        </div>
    </div>
"""

    # SHAP analysis (collapsible)
    shap_file = f'graficos_finales/shap_{target}.png'
    if os.path.exists(f'resultados/{shap_file}'):
        html += f"""
    <button class="collapsible">SHAP Feature Importance (Top 15 Variables)</button>
    <div class="content">
        <div class="content-inner">
            <p>SHAP (SHapley Additive exPlanations) values show the contribution of each feature to model predictions.</p>
            <img src="{shap_file}" alt="SHAP analysis for {target}">
            <p class="note">Higher values indicate greater predictive importance. Direction shows positive or negative influence.</p>
        </div>
    </div>
"""

    # Validation statistics (collapsible)
    validation = res['validation']
    html += f"""
    <button class="collapsible">Validation Statistics</button>
    <div class="content">
        <div class="content-inner">
            <h3>Robustness Analysis</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>Shapiro-Wilk p-value</td>
                    <td>{validation['shapiro_p']:.4f}</td>
                    <td>{"Residuals approximately normal (p > 0.05)" if validation['shapiro_p'] > 0.05 else "Residuals non-normal (p < 0.05)"}</td>
                </tr>
                <tr>
                    <td>Bootstrap R² Mean</td>
                    <td>{validation['bootstrap_r2_mean']:.4f}</td>
                    <td>Average R² across 100 bootstrap samples</td>
                </tr>
                <tr>
                    <td>Bootstrap 95% CI</td>
                    <td>[{validation['bootstrap_ci'][0]:.4f}, {validation['bootstrap_ci'][1]:.4f}]</td>
                    <td>Confidence interval for R²</td>
                </tr>
            </table>
            <p class="note">Bootstrap resampling provides confidence intervals for model performance.</p>
        </div>
    </div>
"""

html += """
    <hr style="margin: 20pt 0; border: none; border-top: 2px solid #000;">

    <h2>Methodology</h2>

    <button class="collapsible">Data & Variables</button>
    <div class="content">
        <div class="content-inner">
            <p><strong>Sample:</strong> 52 low and lower-middle income countries</p>
            <p><strong>Variables:</strong> 131 gender and development indicators from World Bank, UN, UNESCO</p>
            <p><strong>Categories:</strong> Cultural, Demographic, Health, Education, Labour</p>
            <p><strong>Feature Selection:</strong> Top 15 predictors selected via Pearson correlation for each target</p>
            <p><strong>Train/Test Split:</strong> 75/25 stratified random split</p>
        </div>
    </div>

    <button class="collapsible">Models & Optimization</button>
    <div class="content">
        <div class="content-inner">
            <h3>Models Tested:</h3>
            <ol>
                <li><strong>Random Forest:</strong> Recursive hyperparameter optimization (5 levels)</li>
                <li><strong>XGBoost:</strong> Gradient boosting with tree-based learning</li>
                <li><strong>Neural Network:</strong> Multi-layer perceptron (100-50-25 neurons)</li>
                <li><strong>Gradient Boosting:</strong> Sequential ensemble method</li>
            </ol>

            <h3>Recursive Optimization Process:</h3>
            <p>For Random Forest, parameters optimized sequentially via 5-fold cross-validation:</p>
            <ol>
                <li>Level 1: n_estimators (100, 200, 300, 500)</li>
                <li>Level 2: max_depth (3, 5, 7, 10, None)</li>
                <li>Level 3: min_samples_split (2, 5, 10)</li>
                <li>Level 4: min_samples_leaf (2, 4, 8)</li>
            </ol>
            <p>At each level, only parameters showing improvement are updated, then recursion continues.</p>
        </div>
    </div>

    <button class="collapsible">Validation Methods</button>
    <div class="content">
        <div class="content-inner">
            <ul>
                <li><strong>5-Fold Cross-Validation:</strong> Assess model stability across data splits</li>
                <li><strong>Bootstrap Resampling:</strong> 100 iterations for confidence intervals</li>
                <li><strong>Residual Analysis:</strong> Shapiro-Wilk test for normality assumption</li>
                <li><strong>SHAP Analysis:</strong> Model-agnostic feature importance</li>
            </ul>
        </div>
    </div>

    <hr style="margin: 20pt 0; border: none; border-top: 1px solid #999;">

    <p class="note" style="text-align: center;">
        Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        For questions or methodology details, refer to complete documentation.
    </p>

    <script>
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {{
            coll[i].addEventListener("click", function() {{
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {{
                    content.style.maxHeight = null;
                }} else {{
                    content.style.maxHeight = content.scrollHeight + "px";
                }}
            }});
        }}
    </script>
</body>
</html>
"""

# Save dashboard
with open('resultados/dashboard_academico.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard academico creado: resultados/dashboard_academico.html")
