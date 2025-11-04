#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALISIS COMPLETO DUAL TARGET
Random Forest, XGBoost, Neural Networks + SHAP + Validacion
52 paises, 2 targets, optimizacion recursiva
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, shapiro
from scipy import stats

# Plotting
import matplotlib
matplotlib.use('Agg')  # No GUI
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: XGBoost not available")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available - install with: pip install shap")

# Create directories
os.makedirs('resultados/graficos_finales', exist_ok=True)
os.makedirs('resultados/modelos', exist_ok=True)
os.makedirs('resultados/validacion', exist_ok=True)

print("="*80)
print("ANALISIS COMPLETO - DUAL TARGET")
print("52 paises, 2 variables objetivo")
print("="*80)

# Load data
df = pd.read_excel('DATA_GHAB2.xlsx')
targets = ['New_Business_Density', 'G_GPD_PCAP_SLOPE']

print(f"\nDatos: {len(df)} paises, {len(df.columns)} variables")
print(f"Targets: {targets}")

# Functions
def get_predictors(df, target, n=15):
    """Get top N predictors by correlation"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    candidates = [c for c in numeric_cols if c not in targets]

    correlations = []
    for var in candidates:
        mask = df[target].notna() & df[var].notna()
        if mask.sum() < 10:
            continue
        try:
            r, p = pearsonr(df.loc[mask, var], df.loc[mask, target])
            correlations.append({'var': var, 'r': abs(r), 'p': p})
        except:
            continue

    corr_df = pd.DataFrame(correlations).sort_values('r', ascending=False)
    return corr_df.head(n)['var'].tolist()

def recursive_optimize_rf(X_train, y_train, depth=0, current_params=None, max_depth=5, verbose=True):
    """Recursive optimization for Random Forest"""
    if depth >= max_depth:
        return current_params

    param_search = [
        {'n_estimators': [100, 200, 300, 500]},
        {'max_depth': [3, 5, 7, 10, None]},
        {'min_samples_split': [2, 5, 10]},
        {'min_samples_leaf': [2, 4, 8]},
    ]

    if depth >= len(param_search):
        return current_params

    if current_params is None:
        current_params = {'n_estimators': 100, 'random_state': 42}
        best_score = -np.inf
    else:
        model = RandomForestRegressor(**current_params, n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        best_score = scores.mean()

    param_grid = param_search[depth]
    param_name = list(param_grid.keys())[0]
    param_values = param_grid[param_name]

    if verbose:
        print(f"   Level {depth+1}: {param_name}")

    improved = False
    best_value = current_params.get(param_name, param_values[0])

    for value in param_values:
        test_params = current_params.copy()
        test_params[param_name] = value

        model = RandomForestRegressor(**test_params, n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        score = scores.mean()

        if score > best_score:
            best_score = score
            best_value = value
            improved = True

    if improved:
        current_params[param_name] = best_value
        if verbose:
            print(f"      -> {param_name}={best_value} (CV={best_score:.4f})")
        return recursive_optimize_rf(X_train, y_train, depth+1, current_params, max_depth, verbose)
    else:
        return current_params

def train_models(X_train, X_test, y_train, y_test, target_name):
    """Train all models"""
    results = {}

    # 1. Random Forest (optimized)
    print(f"\n  1. Random Forest (recursive optimization)...")
    rf_params = recursive_optimize_rf(X_train, y_train, max_depth=4, verbose=False)
    rf_model = RandomForestRegressor(**rf_params, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    results['Random Forest'] = {
        'model': rf_model,
        'y_pred': y_pred_rf,
        'r2': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'cv': cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2').mean(),
        'params': rf_params
    }
    print(f"     R2={results['Random Forest']['r2']:.4f}, CV={results['Random Forest']['cv']:.4f}")

    # 2. XGBoost
    if XGB_AVAILABLE:
        print(f"\n  2. XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)

        results['XGBoost'] = {
            'model': xgb_model,
            'y_pred': y_pred_xgb,
            'r2': r2_score(y_test, y_pred_xgb),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'mae': mean_absolute_error(y_test, y_pred_xgb),
            'cv': cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2').mean()
        }
        print(f"     R2={results['XGBoost']['r2']:.4f}, CV={results['XGBoost']['cv']:.4f}")

    # 3. Neural Network
    print(f"\n  3. Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        max_iter=2000,
        random_state=42,
        early_stopping=True
    )
    nn_model.fit(X_train_scaled, y_train)
    y_pred_nn = nn_model.predict(X_test_scaled)

    results['Neural Network'] = {
        'model': nn_model,
        'y_pred': y_pred_nn,
        'r2': r2_score(y_test, y_pred_nn),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
        'mae': mean_absolute_error(y_test, y_pred_nn),
        'cv': cross_val_score(nn_model, X_train_scaled, y_train, cv=5, scoring='r2').mean(),
        'scaler': scaler
    }
    print(f"     R2={results['Neural Network']['r2']:.4f}, CV={results['Neural Network']['cv']:.4f}")

    # 4. Gradient Boosting
    print(f"\n  4. Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.01,
        subsample=0.8, random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    results['Gradient Boosting'] = {
        'model': gb_model,
        'y_pred': y_pred_gb,
        'r2': r2_score(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'cv': cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2').mean()
    }
    print(f"     R2={results['Gradient Boosting']['r2']:.4f}, CV={results['Gradient Boosting']['cv']:.4f}")

    return results

def validation_analysis(y_test, y_pred, model_name, target_name):
    """Complete validation analysis"""

    residuals = y_test.values - y_pred

    # 1. Residual normality test
    _, p_shapiro = shapiro(residuals)

    # 2. Bootstrap confidence interval
    n_bootstrap = 100
    bootstrap_r2 = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), len(y_test), replace=True)
        bootstrap_r2.append(r2_score(y_test.values[indices], y_pred[indices]))

    ci_lower = np.percentile(bootstrap_r2, 2.5)
    ci_upper = np.percentile(bootstrap_r2, 97.5)

    return {
        'residuals': residuals,
        'shapiro_p': p_shapiro,
        'bootstrap_r2_mean': np.mean(bootstrap_r2),
        'bootstrap_ci': (ci_lower, ci_upper)
    }

# MAIN ANALYSIS
all_results = {}

for target in targets:
    print(f"\n{'='*80}")
    print(f"TARGET: {target}")
    print(f"{'='*80}")

    # Get predictors
    print(f"\n1. Selecting top 15 predictors...")
    predictors = get_predictors(df, target, n=15)
    print(f"   OK: {len(predictors)} predictors selected")

    # Prepare data
    data_clean = df[predictors + [target]].dropna()
    X = data_clean[predictors]
    y = data_clean[target]

    # Special handling for New_Business_Density (log transform)
    if target == 'New_Business_Density':
        offset = abs(y.min()) + 1 if y.min() <= 0 else 0
        y_transform = np.log(y + offset)
        use_transform = True
    else:
        y_transform = y
        use_transform = False
        offset = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transform, test_size=0.25, random_state=42
    )

    print(f"   Data: {len(X_train)} train, {len(X_test)} test")

    # Train models
    print(f"\n2. Training models...")
    models_results = train_models(X_train, X_test, y_train, y_test, target)

    # Rank models
    ranked = sorted(models_results.items(), key=lambda x: x[1]['r2'], reverse=True)

    print(f"\n3. Model Rankings:")
    for i, (name, res) in enumerate(ranked, 1):
        print(f"   {i}. {name:20s} R2={res['r2']:.4f} CV={res['cv']:.4f}")

    # Get top 3
    top_3 = ranked[:3]
    best_model_name, best_model_result = ranked[0]

    # Validation for best model
    print(f"\n4. Validation analysis (best model: {best_model_name})...")

    # Transform predictions back if needed
    y_pred_best = best_model_result['y_pred']
    if use_transform:
        y_test_original = np.exp(y_test) - offset
        y_pred_original = np.exp(y_pred_best) - offset
    else:
        y_test_original = y_test
        y_pred_original = y_pred_best

    validation = validation_analysis(y_test, y_pred_best, best_model_name, target)

    print(f"   Shapiro normality test: p={validation['shapiro_p']:.4f}")
    print(f"   Bootstrap R2: {validation['bootstrap_r2_mean']:.4f}")
    print(f"   95% CI: [{validation['bootstrap_ci'][0]:.4f}, {validation['bootstrap_ci'][1]:.4f}]")

    # SHAP analysis for best model
    if SHAP_AVAILABLE and best_model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
        print(f"\n5. SHAP analysis...")
        try:
            explainer = shap.TreeExplainer(best_model_result['model'])
            shap_values = explainer.shap_values(X_test)

            # Save SHAP plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(f'resultados/graficos_finales/shap_{target}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   SHAP analysis saved")
        except Exception as e:
            print(f"   SHAP failed: {e}")

    # Store results
    all_results[target] = {
        'predictors': predictors,
        'models': models_results,
        'top_3': top_3,
        'best': (best_model_name, best_model_result),
        'validation': validation,
        'X_test': X_test,
        'y_test': y_test_original if use_transform else y_test,
        'y_pred_best': y_pred_original if use_transform else y_pred_best,
        'use_transform': use_transform
    }

# Save results
with open('resultados/modelos/all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print(f"\n{'='*80}")
print("RESUMEN FINAL")
print(f"{'='*80}")

for target in targets:
    best_name, best_res = all_results[target]['best']
    print(f"\n{target}:")
    print(f"  Mejor modelo: {best_name}")
    print(f"  R2 Test: {best_res['r2']:.4f}")
    print(f"  CV: {best_res['cv']:.4f}")
    print(f"  RMSE: {best_res['rmse']:.4f}")

print(f"\nResultados guardados en: resultados/modelos/all_results.pkl")
print(f"Graficos SHAP en: resultados/graficos_finales/")
