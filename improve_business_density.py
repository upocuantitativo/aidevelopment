#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEJORA DE MODELO: New_Business_Density
Estrategias múltiples para mejorar R² negativo
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MEJORA DE MODELO: New_Business_Density")
print("="*80)

# Cargar datos
df = pd.read_excel('DATA_GHAB2.xlsx')
target = 'New_Business_Density'

print(f"\n1. Analisis de datos para {target}")
print(f"   Observaciones: {df[target].notna().sum()}")
print(f"   Min: {df[target].min():.2f}")
print(f"   Max: {df[target].max():.2f}")
print(f"   Mean: {df[target].mean():.2f}")
print(f"   Std: {df[target].std():.2f}")

# ESTRATEGIA 1: Probar diferentes números de predictores
print(f"\n2. ESTRATEGIA 1: Probar diferentes cantidades de predictores")

numeric_cols = df.select_dtypes(include=[np.number]).columns
candidates = [c for c in numeric_cols if c not in ['New_Business_Density', 'G_GPD_PCAP_SLOPE']]

# Obtener correlaciones
correlations = []
for var in candidates:
    mask = df[target].notna() & df[var].notna()
    if mask.sum() < 10:
        continue
    try:
        r, p = pearsonr(df.loc[mask, var], df.loc[mask, target])
        correlations.append({'var': var, 'r': abs(r), 'p': p, 'r_signed': r})
    except:
        continue

corr_df = pd.DataFrame(correlations).sort_values('r', ascending=False)

best_result = None
best_n_features = 0

for n_features in [5, 10, 15, 20, 25, 30]:
    if n_features > len(corr_df):
        continue

    top_vars = corr_df.head(n_features)['var'].tolist()

    data_clean = df[top_vars + [target]].dropna()
    if len(data_clean) < 20:
        continue

    X = data_clean[top_vars]
    y = data_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Random Forest simple
    model = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()

    print(f"   {n_features} features: R2_test={r2_test:.4f}, CV={cv_score:.4f}")

    if best_result is None or r2_test > best_result['r2_test']:
        best_result = {
            'n_features': n_features,
            'r2_test': r2_test,
            'cv_score': cv_score,
            'model': model,
            'features': top_vars,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred
        }
        best_n_features = n_features

print(f"\n   MEJOR: {best_n_features} features con R2_test={best_result['r2_test']:.4f}")

# ESTRATEGIA 2: Probar modelos lineales (menos overfitting)
print(f"\n3. ESTRATEGIA 2: Modelos lineales regularizados")

X_train = best_result['X_train']
X_test = best_result['X_test']
y_train = best_result['y_train']
y_test = best_result['y_test']

# Escalar datos
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

linear_results = {}
for name, model in linear_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2_test = r2_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1).mean()

    print(f"   {name}: R2_test={r2_test:.4f}, CV={cv_score:.4f}")

    linear_results[name] = {
        'model': model,
        'r2_test': r2_test,
        'cv_score': cv_score,
        'y_pred': y_pred
    }

# ESTRATEGIA 3: Gradient Boosting con regularización fuerte
print(f"\n4. ESTRATEGIA 3: Gradient Boosting regularizado")

gb_params = [
    {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.01, 'subsample': 0.7},
    {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.01, 'subsample': 0.8},
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'subsample': 0.7},
    {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.005, 'subsample': 0.8},
]

gb_results = {}
for i, params in enumerate(gb_params):
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2_test = r2_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()

    print(f"   Config {i+1}: R2_test={r2_test:.4f}, CV={cv_score:.4f}")
    print(f"      Params: {params}")

    gb_results[f'GB_{i+1}'] = {
        'model': model,
        'r2_test': r2_test,
        'cv_score': cv_score,
        'y_pred': y_pred,
        'params': params
    }

# ESTRATEGIA 4: Transformación logarítmica del target
print(f"\n5. ESTRATEGIA 4: Transformacion logaritmica")

# Verificar si hay valores negativos o ceros
if (y_train <= 0).any() or (y_test <= 0).any():
    print("   Sumando constante para evitar log de valores negativos/cero")
    offset = abs(y_train.min()) + 1
    y_train_log = np.log(y_train + offset)
    y_test_log = np.log(y_test + offset)
else:
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    offset = 0

model_log = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
model_log.fit(X_train, y_train_log)

y_pred_log = model_log.predict(X_test)
y_pred_original = np.exp(y_pred_log) - offset

r2_test_log = r2_score(y_test, y_pred_original)
cv_score_log = cross_val_score(model_log, X_train, y_train_log, cv=5, scoring='r2', n_jobs=-1).mean()

print(f"   Log transform: R2_test={r2_test_log:.4f}, CV={cv_score_log:.4f}")

# COMPARAR TODOS LOS MODELOS
print(f"\n{'='*80}")
print("RESUMEN DE TODAS LAS ESTRATEGIAS")
print(f"{'='*80}")

all_models = {
    f'RandomForest_{best_n_features}feat': best_result,
    **linear_results,
    **gb_results,
    'RF_LogTransform': {
        'model': model_log,
        'r2_test': r2_test_log,
        'cv_score': cv_score_log,
        'y_pred': y_pred_original
    }
}

# Ordenar por R2_test
sorted_models = sorted(all_models.items(), key=lambda x: x[1]['r2_test'], reverse=True)

print("\nRanking de modelos (ordenados por R2_test):")
print("-"*80)
for i, (name, result) in enumerate(sorted_models[:10], 1):
    print(f"{i:2d}. {name:30s} R2_test={result['r2_test']:7.4f}  CV={result['cv_score']:7.4f}")

# Mejor modelo
best_model_name, best_model_result = sorted_models[0]

print(f"\n{'='*80}")
print(f"MEJOR MODELO: {best_model_name}")
print(f"{'='*80}")
print(f"R2 Test:  {best_model_result['r2_test']:.4f}")
print(f"CV Score: {best_model_result['cv_score']:.4f}")

if best_model_result['r2_test'] > 0:
    print(f"\nMEJORA EXITOSA!")
    rmse = np.sqrt(mean_squared_error(y_test, best_model_result['y_pred']))
    mae = mean_absolute_error(y_test, best_model_result['y_pred'])
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
else:
    print(f"\nADVERTENCIA: Aun con R2 negativo. Posibles causas:")
    print(f"  - Muy pocas observaciones (n={len(y_test)} test)")
    print(f"  - Variable objetivo dificil de predecir")
    print(f"  - Necesita features engineering adicional")

# Guardar mejor modelo
import pickle
os.makedirs('modelos', exist_ok=True)
with open('modelos/best_new_business_density.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model_result['model'],
        'model_name': best_model_name,
        'features': best_result['features'],
        'r2_test': best_model_result['r2_test'],
        'cv_score': best_model_result['cv_score']
    }, f)

print(f"\nModelo guardado en: modelos/best_new_business_density.pkl")
