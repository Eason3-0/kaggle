# -*- coding: utf-8 -*-
"""
Step 3: XGBoost Training — simple, fast, effective.
"""
import os, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from config import *
from utils import *

warnings.filterwarnings("ignore")
ensure_dirs()

print_section("Step 3: XGBoost Training")

# 1. Load
with Timer("Loading features"):
    X_train = pd.read_csv(os.path.join(DATA_FEATURES_DIR, "X_train.csv"))
    X_valid = pd.read_csv(os.path.join(DATA_FEATURES_DIR, "X_valid.csv"))

y_train = X_train.pop("sales")
y_valid = X_valid.pop("sales")
X_train.drop(columns=["date", "id"], inplace=True, errors='ignore')
X_valid.drop(columns=["date", "id"], inplace=True, errors='ignore')

# Align
common = X_train.columns.intersection(X_valid.columns).tolist()
X_train, X_valid = X_train[common], X_valid[common]
print(f"  Train: {X_train.shape}, Valid: {X_valid.shape}")

# 2. Train
params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.03, "max_depth": 10, "subsample": 0.7,
    "colsample_bytree": 0.7, "min_child_weight": 3,
    "reg_alpha": 0.5, "reg_lambda": 2.0,
    "random_state": 42, "n_jobs": -1, "verbosity": 0,
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

print("  Training (1500 rounds, early_stopping=50)...")
with Timer("Training"):
    model = xgb.train(
        params, dtrain, num_boost_round=1500,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50, verbose_eval=100,
    )

y_pred = model.predict(dvalid)
val_rmsle = rmsle(y_valid, y_pred)
print(f"\n  Best iter: {model.best_iteration}, RMSLE: {val_rmsle:.4f}")

# 3. Feature importance
imp = model.get_score(importance_type="gain")
imp_df = pd.DataFrame({"feature": list(imp.keys()), "gain": list(imp.values())}).sort_values("gain", ascending=False)
print(f"\n  Top 10 features:")
for i, (_, r) in enumerate(imp_df.head(10).iterrows()):
    print(f"    {i+1}. {r['feature']:40s} {r['gain']:>15.0f}")

# 4. Export
model.save_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
with open(os.path.join(MODELS_DIR, "feature_list.txt"), "w") as f:
    for c in common: f.write(c + "\n")

print(f"\n[INFO] Model saved. {len(common)} features. RMSLE={val_rmsle:.4f}")
