# -*- coding: utf-8 -*-
"""
Step 4: Generate Submission
Uses shared feature_engine.py — 100% consistent with step2.
Per-group ffill for lag/rolling columns.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from config import *
from utils import *
from feature_engine import build_features

warnings.filterwarnings("ignore")
ensure_dirs()

print_section("Step 4: Generate Submission")

# 1. Load
with Timer("Loading"):
    df_train = pd.read_csv(TRAIN_CLEANED_PATH, parse_dates=["date"])
    df_test_raw = pd.read_csv(TEST_PATH, parse_dates=["date"])
    df_stores = pd.read_csv(STORES_CLEANED_PATH)
    df_oil = pd.read_csv(OIL_CLEANED_PATH, parse_dates=["date"])
    df_holidays = pd.read_csv(HOLIDAYS_CLEANED_PATH, parse_dates=["date"])
print(f"  Train: {df_train.shape[0]:,}, Test: {df_test_raw.shape[0]:,}")

ORIGINAL_IDS = df_test_raw["id"].values.copy()

# 2. Prepare test
df_test_base = df_test_raw.copy()
df_test_base["sales"] = np.nan
df_test_base["onpromotion"] = df_test_base["onpromotion"].fillna(0)
df_test_base = df_test_base.merge(
    df_stores[["store_nbr", "city", "state", "type", "cluster", "type_code"]],
    on="store_nbr", how="left")
for col in ["has_holiday", "has_event", "is_national", "is_regional", "is_local"]:
    if col in df_train.columns:
        hr = df_train[["date", col]].drop_duplicates("date")
        df_test_base = df_test_base.merge(hr, on="date", how="left")
        df_test_base[col] = df_test_base[col].fillna(0).astype(int)

# 3. Concatenate, save grouping cols, build features
df_all = pd.concat([df_train, df_test_base], ignore_index=True)
df_all["date"] = pd.to_datetime(df_all["date"])
df_all["_family"] = df_all["family"].copy()
df_all["_store"] = df_all["store_nbr"].copy()

print_section("Building Features (shared engine)")
with Timer("Feature engineering"):
    df_all, _ = build_features(df_all, df_stores, df_oil, df_holidays, verbose=True)

# 4. Per-group ffill for dynamic columns
print_section("Filling Test Features (per-group ffill)")

test_mask = df_all["date"] >= pd.Timestamp(TEST_START_DATE)

dynamic_cols = [c for c in df_all.columns if
                c.startswith("sales_lag_") or c.startswith("sales_rolling_") or
                c.startswith("onpromotion_lag_") or c.startswith("onpromotion_rolling_")]

for col in dynamic_cols:
    df_all.loc[test_mask, col] = np.nan
    df_all[col] = df_all.groupby(["_store", "_family"])[col].ffill().bfill()

df_all = df_all.fillna(0)
print(f"  Filled {len(dynamic_cols)} dynamic columns")

# 5. Load model & predict
print_section("Loading Model & Predicting")

booster = xgb.Booster()
booster.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))

with open(os.path.join(MODELS_DIR, "feature_list.txt")) as f:
    MODEL_COLS = [l.strip() for l in f if l.strip()]

model_feature_cols = [c for c in MODEL_COLS if c != "id"]
missing = set(model_feature_cols) - set(df_all.columns)
for c in missing: df_all[c] = 0.0
print(f"  Model: {len(model_feature_cols)} features, missing: {len(missing)}")

X_test = df_all.loc[test_mask, model_feature_cols].fillna(0)
assert len(X_test) == 28512

with Timer("Prediction"):
    y_pred = booster.predict(xgb.DMatrix(X_test))
y_pred = np.maximum(0, y_pred)
print(f"  Predictions: mean={y_pred.mean():.1f}, median={np.median(y_pred):.1f}")

# 6. Build submission
print_section("Building Submission")

test_ids = df_all.loc[test_mask, "id"].values.astype(int)
sub = pd.DataFrame({"id": test_ids, "sales": y_pred})
sub = sub.set_index("id").reindex(ORIGINAL_IDS).reset_index()
sub.columns = ["id", "sales"]
sub["sales"] = sub["sales"].fillna(0).clip(lower=0)

assert len(sub) == 28512
assert (sub["id"].values == ORIGINAL_IDS).all(), "ID MISMATCH!"

sub.to_csv(os.path.join(SUBMISSIONS_DIR, "submission.csv"), index=False)
print(f"  Mean={sub['sales'].mean():.1f}  Median={sub['sales'].median():.1f}")
print(f"  Min={sub['sales'].min():.1f}  Max={sub['sales'].max():.1f}")
print(f"  Zeros={(sub['sales']==0).sum()}  ID: OK")
print(f"\n[INFO] Done. Upload {SUBMISSIONS_DIR}/submission.csv to Kaggle.")
