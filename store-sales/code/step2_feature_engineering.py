# -*- coding: utf-8 -*-
"""
Step 2: Feature Engineering & Train/Validation Split
Uses shared feature_engine.py — guarantees consistency with step4.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

from config import *
from utils import *
from feature_engine import build_features

warnings.filterwarnings("ignore")
ensure_dirs()

print_section("Step 2: Feature Engineering")

# 1. Load
with Timer("Loading data"):
    df = pd.read_csv(TRAIN_CLEANED_PATH, parse_dates=["date"])
    df_stores = pd.read_csv(STORES_CLEANED_PATH)
    df_oil = pd.read_csv(OIL_CLEANED_PATH, parse_dates=["date"])
    df_holidays = pd.read_csv(HOLIDAYS_CLEANED_PATH, parse_dates=["date"])
print(f"  Train: {df.shape[0]:,} rows")

# 2. Build features
print_section("Building Features")
with Timer("Feature engineering"):
    df, feature_cols = build_features(df, df_stores, df_oil, df_holidays, verbose=True)
print(f"  Features: {len(feature_cols)}, Rows: {df.shape[0]:,}")

# 3. Drop rows with NaN lags (first 28 days per group), THEN fillna
initial = len(df)
df = df.dropna(subset=["sales_lag_7", "sales_lag_14", "sales_lag_28"])
print(f"  Dropped {initial - len(df):,} rows with NaN lags ({fmt_pct(initial - len(df), initial)})")
df = df.fillna(0)

# 4. Train/Validation split
print_section("Train/Validation Split")
valid_start = pd.Timestamp(VALID_START_DATE)
y = df.pop("sales")
X = df.drop(columns=["date", "id"], errors="ignore")
train_mask = df["date"] < valid_start

X_train, X_valid = X[train_mask], X[~train_mask]
y_train, y_valid = y[train_mask], y[~train_mask]
print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
print(f"  Valid: {X_valid.shape[0]:,} x {X_valid.shape[1]}")

# 5. Export
print_section("Exporting")
train_out = X_train.copy(); train_out["sales"] = y_train.values; train_out["date"] = df.loc[train_mask, "date"].values
valid_out = X_valid.copy(); valid_out["sales"] = y_valid.values; valid_out["date"] = df.loc[~train_mask, "date"].values
# Also drop 'id' from feature_names for model training
feature_cols_out = [c for c in X_train.columns if c != "id"]

train_out.to_csv(os.path.join(DATA_FEATURES_DIR, "X_train.csv"), index=False)
valid_out.to_csv(os.path.join(DATA_FEATURES_DIR, "X_valid.csv"), index=False)
pd.DataFrame({"feature": feature_cols_out}).to_csv(
    os.path.join(DATA_FEATURES_DIR, "feature_names.txt"), index=False)

print(f"  → {DATA_FEATURES_DIR}")
print(f"[INFO] Step 2 done. {len(feature_cols)} features.")
