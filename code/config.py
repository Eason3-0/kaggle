# -*- coding: utf-8 -*-
"""
====================================================================
Shared Configuration
Store Sales Prediction — Kaggle Competition
====================================================================

Centralized paths, constants, and type mappings used across all steps.
Import this module in every step script to avoid duplicating path logic.

Usage:
    from config import *
    # or
    import config
    df = pd.read_csv(config.TRAIN_PATH, parse_dates=["date"])
"""

import os

# ============================================================
# Project Root & Directory Paths
# ============================================================

# Absolute path to the project root (parent of code/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

# Output directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")

# Code directory
CODE_DIR = os.path.join(PROJECT_ROOT, "code")


# ============================================================
# Raw Data File Paths
# ============================================================

TRAIN_PATH = os.path.join(DATA_RAW_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_RAW_DIR, "test.csv")
STORES_PATH = os.path.join(DATA_RAW_DIR, "stores.csv")
OIL_PATH = os.path.join(DATA_RAW_DIR, "oil.csv")
HOLIDAYS_PATH = os.path.join(DATA_RAW_DIR, "holidays_events.csv")
TRANSACTIONS_PATH = os.path.join(DATA_RAW_DIR, "transactions.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_RAW_DIR, "sample_submission.csv")


# ============================================================
# Processed Data File Paths
# ============================================================

TRAIN_CLEANED_PATH = os.path.join(DATA_PROCESSED_DIR, "train_cleaned.csv")
STORES_CLEANED_PATH = os.path.join(DATA_PROCESSED_DIR, "stores_cleaned.csv")
OIL_CLEANED_PATH = os.path.join(DATA_PROCESSED_DIR, "oil_cleaned.csv")
HOLIDAYS_CLEANED_PATH = os.path.join(DATA_PROCESSED_DIR, "holidays_cleaned.csv")
TRANSACTIONS_CLEANED_PATH = os.path.join(DATA_PROCESSED_DIR, "transactions_cleaned.csv")


# ============================================================
# Feature / Model / Submission Output Paths
# ============================================================

FEATURE_MATRIX_TRAIN_PATH = os.path.join(DATA_FEATURES_DIR, "X_train.csv")
FEATURE_MATRIX_VALID_PATH = os.path.join(DATA_FEATURES_DIR, "X_valid.csv")
FEATURE_MATRIX_TEST_PATH = os.path.join(DATA_FEATURES_DIR, "X_test.csv")
TARGET_TRAIN_PATH = os.path.join(DATA_FEATURES_DIR, "y_train.csv")
TARGET_VALID_PATH = os.path.join(DATA_FEATURES_DIR, "y_valid.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "model_lgb.txt")
SUBMISSION_PATH = os.path.join(SUBMISSIONS_DIR, "submission.csv")


# ============================================================
# Date Constants
# ============================================================

# Training data range
TRAIN_START_DATE = "2013-01-01"
TRAIN_END_DATE = "2017-08-15"

# Test data range (prediction target)
TEST_START_DATE = "2017-08-16"
TEST_END_DATE = "2017-08-31"

# Suggested validation split point (last ~2 weeks of training)
VALID_START_DATE = "2017-07-26"


# ============================================================
# Data Constants
# ============================================================

NUM_STORES = 54
NUM_FAMILIES = 33

# Product families
FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
    "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
    "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
    "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
    "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
    "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
]


# ============================================================
# Encoding Mappings
# ============================================================

# Store type ordinal encoding: A=1, B=2, C=3, D=4, E=5
STORE_TYPE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# Holiday/Event type encoding
EVENT_TYPE_MAP = {
    "Holiday": 1,
    "Bridge": 2,
    "Event": 3,
    "Additional": 4,
    "Transfer": 5,
    "Work Day": 6,
}

# Day-of-week names
DOW_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}


# ============================================================
# Model Constants
# ============================================================

# Default LightGBM parameters
DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

# Evaluation metric
METRIC_NAME = "RMSLE"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [DATA_PROCESSED_DIR, DATA_FEATURES_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
        os.makedirs(d, exist_ok=True)


# ============================================================
# Print config summary (useful for debugging)
# ============================================================

if __name__ == "__main__":
    print(f"PROJECT_ROOT:       {PROJECT_ROOT}")
    print(f"DATA_RAW_DIR:       {DATA_RAW_DIR}")
    print(f"DATA_PROCESSED_DIR: {DATA_PROCESSED_DIR}")
    print(f"DATA_FEATURES_DIR:  {DATA_FEATURES_DIR}")
    print(f"MODELS_DIR:         {MODELS_DIR}")
    print(f"SUBMISSIONS_DIR:    {SUBMISSIONS_DIR}")
    print(f"\nAll directory constants loaded. ensure_dirs() available.")
