# Store Sales Prediction — Kaggle Project

## Overview
Predict daily sales for 54 stores × 33 product families in Ecuador (Corporación Favorita).
Kaggle competition using time-series features + gradient boosting.

**Current Kaggle Score: RMSLE 0.49477** (XGBoost baseline)

## Quick Start
```bash
cd code/
python step1_eda_cleaning.py      # EDA + data cleaning → data/processed/
python step2_feature_engineering.py # Features: LR trend + Fourier + lags → data/features/
python step3_model_training.py     # XGBoost train → models/
python step4_generate_submission.py # Predict → submissions/
```

## Project Structure
```
Store sales/
├── CLAUDE.md                        ← THIS FILE (auto-loaded each session)
├── .gitignore                       ← Excludes generated files (>100MB)
├── code/
│   ├── config.py                    ← Shared: ALL paths, constants, mappings
│   ├── utils.py                     ← Shared: Timer, RMSLE, print helpers
│   ├── feature_engine.py            ← **CRITICAL**: build_features() shared by step2 & step4
│   ├── step1_eda_cleaning.py        ← Load raw → EDA text summary → clean → export
│   ├── step1_visualization.ipynb    ← 10 matplotlib/seaborn charts
│   ├── step2_feature_engineering.py ← Trend(LR) + Fourier + lags/rolling + OHE → split
│   ├── step2_feature_engineering.ipynb ← Demo: trend fit, Fourier viz, correlation
│   ├── step3_model_training.py      ← XGBoost 1500 rounds, early_stopping=50
│   ├── step3_model_training.ipynb   ← Demo: training curves, importance, residuals
│   ├── step4_generate_submission.py ← Per-group ffill → predict → submission.csv
│   └── step4_generate_submission.ipynb ← Demo: distribution comparison, validation
├── data/
│   ├── raw/          (7 original CSVs — NEVER modify)
│   ├── processed/    (5 cleaned CSVs — step1 output)
│   └── features/     (X_train.csv, X_valid.csv — step2 output, .gitignored)
├── models/           (xgboost_model.json — step3 output, .gitignored)
└── submissions/      (submission.csv — step4 output, .gitignored)
```

## Architecture Rules (DO NOT BREAK)

### 1. Feature Consistency is EVERYTHING
- `feature_engine.py:build_features()` is the SINGLE SOURCE OF TRUTH for feature engineering
- Step 2 calls it for training data, Step 4 calls it for test data
- **NEVER** write separate feature code in step2 and step4 — they MUST use the same function

### 2. NaN Handling (Critical Bug Source)
- `build_features()` does NOT call fillna() — it preserves NaN for callers to handle
- Step 2 (training): `dropna(subset=["sales_lag_7", "sales_lag_14", "sales_lag_28"])` THEN `fillna(0)`
- Step 4 (prediction): per-group ffill from training data for dynamic columns, THEN `fillna(0)`

### 3. Per-Group ffill (Not Global!)
- Test lag/rolling features MUST be filled per `(store_nbr, family)` group
- Global ffill leaks values across different store-family groups → garbage predictions
- Step 4 saves `_family` and `_store` before one-hot encoding to enable per-group operations

### 4. ID Ordering
- test.csv ID order is the submission order
- After `sort_values()` in build_features, rows are reordered
- `id` column must be preserved through build_features, then `reindex(ORIGINAL_IDS)` at output

### 5. Time-Based Split (Never Random!)
- Validation: 2017-07-26 to 2017-08-15 (last ~3 weeks of training)
- Test: 2017-08-16 to 2017-08-31 (16 days)
- Random split = data leakage = fake good validation scores

## Key Bugs Fixed (History)
1. **ID mismatch**: test_ids saved BEFORE sort_values → scrambled submission (score 3.6)
2. **OHE all-zeros**: Manual feature construction didn't set family_*/city_* columns → score 1.14
3. **NaN in training**: fillna(0) before dropping NaN lags → model trained on fake data
4. **Global ffill**: ffill leaked values across groups → predictions all ~0
5. **Large file push**: X_train.csv (3.5GB) > GitHub LFS 2GB limit → added to .gitignore

## Dependencies
```
numpy pandas scikit-learn statsmodels xgboost matplotlib seaborn
```
All installed in Python 3.14 environment.

## Git Remote
https://github.com/Eason3-0/kaggle.git (may have moved to kaggle_store_sales)
