# GDP Growth Prediction - Gender Indicators Analysis

## Overview

Machine learning analysis predicting **GDP per capita growth** using gender and development indicators in 52 low and lower-middle income countries.

**Target Variable:** G_GPD_PCAP_SLOPE (GDP per capita growth trajectory)

## Key Results

- **Best Model:** Random Forest (R² = 0.76)
- **Validation:** 5-fold CV R² = 0.21
- **Bootstrap 95% CI:** [0.47, 0.91]
- **Top 3 Models:** Random Forest, Gradient Boosting, XGBoost

## Quick Start

```bash
# Run analysis
python complete_analysis.py

# View results
open resultados/dashboard_final.html
```

## Dataset

- **Countries:** 52 low & lower-middle income nations
- **Variables:** 131 gender and development indicators
- **Sources:** World Bank, UN, UNESCO
- **File:** DATA_GHAB2.xlsx

## Methodology

### Recursive Optimization

Random Forest parameters optimized sequentially via 5-fold CV:
1. n_estimators → best value selected
2. max_depth → best value selected
3. min_samples_split → best value selected
4. min_samples_leaf → best value selected

Only parameters showing improvement are updated before recursing.

### Models

- **Random Forest:** Recursive hyperparameter optimization
- **XGBoost:** Gradient boosting
- **Neural Network:** MLP (100-50-25)
- **Gradient Boosting:** Sequential ensemble

### Validation

- 5-fold cross-validation
- Bootstrap CI (100 iterations)
- Shapiro-Wilk normality test
- SHAP feature importance

## Results

```
resultados/
├── dashboard_final.html          # Main results
├── graficos_finales/
│   └── shap_G_GPD_PCAP_SLOPE.png # SHAP analysis
└── modelos/
    └── all_results.pkl           # Trained models
```

## Installation

```bash
pip install -r requirements.txt
```

## Files

- `complete_analysis.py` - Main analysis
- `create_final_dashboard.py` - Dashboard generator
- `DATA_GHAB2.xlsx` - Input data (52 countries × 131 variables)

## Performance

| Model | R² | RMSE | CV R² |
|-------|-------|------|-------|
| Random Forest | 0.760 | 12.61 | 0.215 |
| Gradient Boosting | 0.704 | - | -0.034 |
| XGBoost | 0.646 | - | 0.036 |

## Citation

Analysis 2025. For methodology, see dashboard documentation.

## License

Academic research use.
