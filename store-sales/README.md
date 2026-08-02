# Store Sales Prediction — Kaggle Competition

Predict daily sales for 54 stores × 33 product families across Ecuador, using time-series feature engineering and XGBoost gradient boosting.

**Kaggle Score: RMSLE 0.49477**

## Project Structure

```
Store sales/
├── code/
│   ├── config.py                       # Shared paths, constants, mappings
│   ├── utils.py                        # Timer, RMSLE, helpers
│   ├── feature_engine.py               # **Core**: build_features() shared by training & prediction
│   ├── step1_eda_cleaning.py           # EDA + data cleaning → data/processed/
│   ├── step1_visualization.ipynb       # 10 exploratory charts
│   ├── step2_feature_engineering.py    # Trend(LR) + Fourier + lags/rolling → data/features/
│   ├── step2_feature_engineering.ipynb # Feature engineering demo
│   ├── step3_model_training.py         # XGBoost training → models/
│   ├── step3_model_training.ipynb      # Training curves, residuals, importance
│   ├── step4_generate_submission.py    # Prediction + submission → submissions/
│   └── step4_generate_submission.ipynb # Submission validation
├── data/
│   ├── raw/                            # Original CSVs (never modified)
│   ├── processed/                      # Cleaned data (step 1 output)
│   └── features/                       # Feature matrices (step 2 output)
├── models/                             # Trained model (step 3 output)
├── submissions/                        # submission.csv (step 4 output)
├── CLAUDE.md                           # Architecture documentation
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn statsmodels xgboost matplotlib seaborn

# 2. Run the full pipeline
cd code/
python step1_eda_cleaning.py       # ~10s  → data/processed/
python step2_feature_engineering.py # ~15s  → data/features/
python step3_model_training.py     # ~110s → models/
python step4_generate_submission.py # ~25s  → submissions/

# 3. Upload submissions/submission.csv to Kaggle
```

Each step has a companion Jupyter notebook (`.ipynb`) for interactive exploration.

## Pipeline

```
Raw Data (data/raw/)
    │
    ▼
Step 1: EDA + Data Cleaning
    ├── Load 7 CSV files (train ~3M rows)
    ├── Text-based EDA summary
    ├── Oil forward-fill, holiday encoding
    └── Merge into single cleaned table
    │
    ▼
Step 2: Feature Engineering (155 features)
    ├── Trend: LinearRegression slope per (store, family)
    ├── Seasonal: Fourier harmonics (yearly 6-order + weekly 3-order)
    ├── Calendar: dayofweek, month, year, weekend, cyclic encoding
    ├── Lags: sales_lag_1/7/14/28, onpromotion_lag_1/7
    ├── Rolling: 7/14/30-day mean, std, min, max
    ├── Holiday: National + Regional(by state) + Local(by city) with proximity windows
    ├── Oil: lagged + rolling + percent change
    ├── Store: type, cluster, city, state one-hot encoded
    └── Interactions: weekend×holiday, promo×weekend, promo×holiday
    │
    ▼
Step 3: XGBoost Training
    ├── 1500 rounds, early_stopping=50
    ├── Validation RMSLE: 0.381
    ├── Top features: rolling_mean_7d, rolling_mean_14d, sales_lag_7
    └── Feature importance + residual analysis
    │
    ▼
Step 4: Submission Generation
    ├── Same feature_engine.build_features() as Step 2
    ├── Per-group ffill for test dynamic features
    └── submission.csv (28,512 rows)
```

## Feature Categories

| Category | Count | Key Features |
|----------|-------|-------------|
| Calendar | 14 | dayofweek, dow_sin/cos, is_weekend, is_month_start |
| Trend | 2 | trend_slope, trend_pred (LR per store×family) |
| Fourier | 18 | fourier_year_* (12), fourier_week_* (6) |
| Lag | 6 | sales_lag_1/7/14/28, onpromotion_lag_1/7 |
| Rolling | 14 | 7/14/30d mean, std, min, max + promo rolling |
| Holiday | 10 | has_holiday, is_national/regional/local, proximity windows |
| Oil | 10 | dcoilwtico, oil_lag_*, oil_rolling_*, oil_change_pct |
| Store | 76 | family OHE(33) + city OHE(22) + state OHE(16) + type OHE(5) |
| Interactions | 4 | weekend_holiday, promo_weekend, promo_holiday, dow_month |
| Other | 1 | transactions |
| **Total** | **155** | |

## Results

| Model | Validation RMSLE | Kaggle RMSLE |
|-------|-----------------|-------------|
| Linear Regression + Seasonality (baseline) | — | 0.51090 |
| **XGBoost (this project)** | 0.381 | **0.49477** |

### Top 10 Feature Importance

| Rank | Feature | Gain |
|------|---------|------|
| 1 | sales_rolling_mean_7d | 1.76B |
| 2 | sales_rolling_mean_14d | 379M |
| 3 | sales_lag_7 | 258M |
| 4 | sales_rolling_max_7d | 212M |
| 5 | sales_lag_14 | 92M |
| 6 | is_weekend | 88M |
| 7 | promo_weekend | 26M |
| 8 | sales_lag_1 | 24M |
| 9 | is_local | 23M |
| 10 | is_month_start | 22M |

## Key Design Decisions

- **Single feature engine**: `feature_engine.py:build_features()` is the only place features are defined. Both training (step 2) and prediction (step 4) call the same function — zero drift.
- **Time-based validation split**: 2017-07-26 to 2017-08-15. Random split would cause data leakage.
- **Per-group ffill**: Test dynamic features (lags, rolling) are forward-filled within each `(store, family)` group, not globally. Global ffill leaks values across different groups.
- **NaN lags dropped during training**: The first 28 days of each store×family combination are excluded from training because lag features are undefined.

## Dependencies

- Python 3.8+
- numpy, pandas
- scikit-learn (LinearRegression)
- statsmodels (DeterministicProcess, Fourier)
- xgboost
- matplotlib, seaborn (notebooks only)

## Data

The dataset is from the [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

| File | Rows | Description |
|------|------|-------------|
| train.csv | 3,000,888 | Daily sales (2013-01-01 to 2017-08-15) |
| test.csv | 28,512 | Prediction target (2017-08-16 to 2017-08-31) |
| stores.csv | 54 | Store metadata (city, state, type, cluster) |
| oil.csv | 1,218 | Daily oil prices |
| holidays_events.csv | 350 | Holiday calendar (National/Regional/Local) |
| transactions.csv | 83,488 | Daily transaction counts per store |
