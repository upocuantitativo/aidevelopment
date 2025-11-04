# Gender Indicators & Economic Development - Dual Target Analysis

## Overview

Advanced machine learning analysis of the relationship between gender indicators and economic development in 52 low and lower-middle income countries.

**Dual Target Variables:**
1. **New_Business_Density** - New business registrations per 1,000 people (ages 15-64)
2. **G_GPD_PCAP_SLOPE** - GDP per capita growth trajectory

## Key Results

- **G_GPD_PCAP_SLOPE**: R² = 0.76 (Random Forest)
- **New_Business_Density**: R² = -0.87 (challenging prediction due to sample size)
- **Models Tested**: Random Forest, XGBoost, Neural Networks, Gradient Boosting
- **Feature Selection**: Top 15 variables via correlation analysis
- **Validation**: 5-fold CV, Bootstrap CI, SHAP analysis

## Quick Start

```bash
# Run complete analysis
python complete_analysis.py

# View results
open resultados/dashboard_academico.html
```

## Dataset

- **Countries**: 52 low & lower-middle income nations
- **Variables**: 131 gender and development indicators
- **Sources**: World Bank, UN Population Division, UNESCO
- **File**: `DATA_GHAB2.xlsx`

## Methodology

### 1. Recursive Optimization

Random Forest parameters optimized sequentially:
- Level 1: n_estimators
- Level 2: max_depth
- Level 3: min_samples_split
- Level 4: min_samples_leaf

Only parameters showing improvement via 5-fold CV are updated.

### 2. Models

- **Random Forest**: Recursive hyperparameter optimization
- **XGBoost**: Gradient boosting
- **Neural Network**: MLP (100-50-25 neurons)
- **Gradient Boosting**: Sequential ensemble

### 3. Validation

- 5-fold cross-validation
- Bootstrap confidence intervals (100 iterations)
- Shapiro-Wilk normality test
- SHAP feature importance

## Results Structure

```
resultados/
├── dashboard_academico.html      # Main results (academic style)
├── graficos_finales/             # SHAP plots
│   ├── shap_New_Business_Density.png
│   └── shap_G_GPD_PCAP_SLOPE.png
└── modelos/
    └── all_results.pkl           # Complete results object
```

## Installation

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib plotly openpyxl scipy
```

## Files

- `complete_analysis.py` - Main analysis script
- `create_dashboard.py` - Generate academic dashboard
- `improve_business_density.py` - Model improvement strategies
- `DATA_GHAB2.xlsx` - Input dataset

## Notes

- New_Business_Density has negative R² due to:
  - Small sample size (52 countries, 13 test)
  - High variability (range: 0.02 to 41,376)
  - Complex economic dynamics
- G_GPD_PCAP_SLOPE achieves strong predictive performance
- All models use top 15 correlated variables for each target

## Citation

Analysis generated on 2025. For methodology details, see dashboard documentation.

## License

Academic and research use.
