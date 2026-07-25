# -*- coding: utf-8 -*-
"""
====================================================================
Shared Utilities
Store Sales Prediction — Kaggle Competition
====================================================================

Reusable helper functions and classes used across all step scripts.
Import from this module to avoid code duplication.

Usage:
    from utils import print_section, Timer, fmt_pct, load_raw_data
"""

import os
import time
from datetime import datetime

import pandas as pd


# ============================================================
# Pretty Printing
# ============================================================

def print_section(title: str):
    """Print a section divider for readability."""
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def fmt_pct(numerator, denominator) -> str:
    """Format a ratio as a percentage string."""
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.2f}%"


# ============================================================
# Timer
# ============================================================

class Timer:
    """Simple context manager / stopwatch for timing code blocks.

    Usage:
        with Timer("Loading data"):
            df = pd.read_csv(...)
        # Prints: [Timer] Loading data: 3.2s

        t = Timer()
        t.start()
        # ... do work ...
        elapsed = t.stop()
    """

    def __init__(self, label: str = "Operation"):
        self.label = label
        self._start = None

    def start(self):
        """Start (or restart) the timer."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop and return elapsed seconds."""
        if self._start is None:
            return 0.0
        elapsed = time.perf_counter() - self._start
        self._start = None
        return elapsed

    def elapsed(self) -> float:
        """Return elapsed seconds without stopping."""
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        elapsed = self.stop()
        print(f"[Timer] {self.label}: {elapsed:.1f}s")


# ============================================================
# Data Loading
# ============================================================

def load_raw_data(data_dir: str) -> dict:
    """Load all raw CSV files from the given data directory.

    Args:
        data_dir: Path to the data/raw/ directory.

    Returns:
        dict with keys: train, test, stores, oil, holidays, transactions, sample_sub
    """
    from config import (  # Deferred import to avoid circular dependency
        TRAIN_PATH, TEST_PATH, STORES_PATH, OIL_PATH,
        HOLIDAYS_PATH, TRANSACTIONS_PATH, SAMPLE_SUB_PATH,
    )

    print("Loading raw data files...")

    datasets = {
        "train": pd.read_csv(TRAIN_PATH, parse_dates=["date"]),
        "test": pd.read_csv(TEST_PATH, parse_dates=["date"]),
        "stores": pd.read_csv(STORES_PATH),
        "oil": pd.read_csv(OIL_PATH, parse_dates=["date"]),
        "holidays": pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"]),
        "transactions": pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"]),
        "sample_sub": pd.read_csv(SAMPLE_SUB_PATH),
    }

    for name, df in datasets.items():
        print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    return datasets


def load_processed_data(data_dir: str) -> dict:
    """Load all cleaned CSV files from the given processed directory.

    Args:
        data_dir: Path to the data/processed/ directory.

    Returns:
        dict with keys: train, stores, oil, holidays, transactions
    """
    print("Loading cleaned data files...")

    datasets = {
        "train": pd.read_csv(
            os.path.join(data_dir, "train_cleaned.csv"), parse_dates=["date"]
        ),
        "stores": pd.read_csv(os.path.join(data_dir, "stores_cleaned.csv")),
        "oil": pd.read_csv(
            os.path.join(data_dir, "oil_cleaned.csv"), parse_dates=["date"]
        ),
        "holidays": pd.read_csv(
            os.path.join(data_dir, "holidays_cleaned.csv"), parse_dates=["date"]
        ),
        "transactions": pd.read_csv(
            os.path.join(data_dir, "transactions_cleaned.csv"), parse_dates=["date"]
        ),
    }

    for name, df in datasets.items():
        print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    return datasets


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================
# RMSLE Metric (used in Steps 3 & 4)
# ============================================================

def rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Logarithmic Error.

    RMSLE = sqrt(1/N * sum((log(y_pred+1) - log(y_true+1))^2))

    Args:
        y_true: Array-like of true values.
        y_pred: Array-like of predicted values.

    Returns:
        RMSLE score (lower is better).
    """
    import numpy as np

    y_true = np.maximum(0, np.asarray(y_true, dtype=float))
    y_pred = np.maximum(0, np.asarray(y_pred, dtype=float))

    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
