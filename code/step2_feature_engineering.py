# -*- coding: utf-8 -*-
"""
====================================================================
Step 2: Feature Engineering & Train/Validation Split
Store Sales Prediction — Kaggle Competition
====================================================================

This script will:
  1. Load cleaned data from data/processed/
  2. Build date features (year, month, day, dayofweek, sin/cos encoding)
  3. Build holiday features (National/Regional/Local matching by store city/state)
  4. Build lag & rolling features (7/14/28-day lags, rolling mean/std/min/max)
  5. Build store & economic features (type encoding, oil lags & rolling, family stats)
  6. Combine all features into a feature matrix
  7. Split by time into X_train, X_valid, y_train, y_valid
  8. Export feature matrices to data/features/

Usage:
    cd code/
    python step2_feature_engineering.py
"""

from config import *
from utils import *

# TODO: Implement Step 2

if __name__ == "__main__":
    print("Step 2: Feature Engineering — NOT YET IMPLEMENTED")
    print(f"Will read from:  {DATA_PROCESSED_DIR}")
    print(f"Will write to:   {DATA_FEATURES_DIR}")
