# -*- coding: utf-8 -*-
"""
====================================================================
Step 1: EDA + Data Cleaning
Store Sales Prediction — Kaggle Competition
====================================================================

This script performs:
  1. Load all raw CSV data files via config.py paths
  2. Text-based EDA summary for each table
  3. Data cleaning (missing value imputation, anomaly tagging, format unification)
  4. Merge core tables into a single master feature table
  5. Export cleaned CSV files to data/processed/
  6. Print EDA summary report to console

Prerequisites:
    Project structure should be:
    ├── data/raw/             (original CSV files)
    ├── data/processed/       (output — created if missing)
    ├── code/config.py        (shared paths & constants)
    └── code/utils.py         (shared utilities)

Usage:
    cd code/
    python step1_eda_cleaning.py
"""

import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Import shared configuration and utilities
from config import *
from utils import (
    print_section,
    print_subsection,
    fmt_pct,
    Timer,
    ensure_dir,
    load_raw_data,
)

warnings.filterwarnings("ignore")

# Ensure output directories exist
ensure_dirs()


# ============================================================
# 1. Load Data
# ============================================================

print_section("Loading Raw Data")

print(f"[INFO] Raw data dir:       {DATA_RAW_DIR}")
print(f"[INFO] Processed data dir: {DATA_PROCESSED_DIR}")

with Timer("Loading all data"):
    datasets = load_raw_data(DATA_RAW_DIR)

df_train = datasets["train"]
df_test = datasets["test"]
df_stores = datasets["stores"]
df_oil = datasets["oil"]
df_holidays = datasets["holidays"]
df_transactions = datasets["transactions"]
df_sample_sub = datasets["sample_sub"]


# ============================================================
# 2. EDA — Text Summary
# ============================================================

print_section("2. EDA Exploration")

# ----------------------------------------------------------
# 2.1 Overview of Each Table
# ----------------------------------------------------------
print_subsection("2.1 Table Overview")

all_datasets = {
    "train": df_train,
    "test": df_test,
    "stores": df_stores,
    "oil": df_oil,
    "holidays_events": df_holidays,
    "transactions": df_transactions,
    "sample_submission": df_sample_sub,
}

for name, df in all_datasets.items():
    print(f"\n{'─' * 48}")
    print(f"【{name}.csv】")
    print(f"  Shape:        {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Columns:      {', '.join(df.columns.tolist())}")
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Memory:       {mem_mb:.1f} MB")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"  Missing:")
        for col, cnt in missing.items():
            print(f"    {col}: {cnt:,} ({fmt_pct(cnt, len(df))})")
    else:
        print(f"  Missing:      None")

    if "date" in df.columns:
        print(f"  Date Range:   {df['date'].min().date()} ~ {df['date'].max().date()}")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0 and len(df) > 0:
        print(f"  Numeric Stats:")
        desc = df[num_cols].describe()
        for col in num_cols:
            print(f"    {col}: mean={desc.loc['mean', col]:.2f}, "
                  f"std={desc.loc['std', col]:.2f}, "
                  f"min={desc.loc['min', col]:.2f}, "
                  f"max={desc.loc['max', col]:.2f}")


# ----------------------------------------------------------
# 2.2 Deep Dive: train.csv
# ----------------------------------------------------------
print_subsection("2.2 train.csv Analysis")

print(f"\n  Time Span:     {df_train['date'].min().date()} ~ {df_train['date'].max().date()}")
print(f"  Num Stores:    {df_train['store_nbr'].nunique()}")
print(f"  Num Families:  {df_train['family'].nunique()}")
print(f"  Families:      {', '.join(sorted(df_train['family'].unique()))}")

sales = df_train["sales"]
print(f"\n  Sales Distribution:")
print(f"    Total:        {sales.sum():,.0f}")
print(f"    Mean:         {sales.mean():.2f}")
print(f"    Median:       {sales.median():.2f}")
print(f"    Std:          {sales.std():.2f}")
print(f"    Min:          {sales.min():.2f}")
print(f"    Max:          {sales.max():.2f}")
print(f"    P25:          {sales.quantile(0.25):.2f}")
print(f"    P75:          {sales.quantile(0.75):.2f}")
print(f"    P95:          {sales.quantile(0.95):.2f}")
print(f"    P99:          {sales.quantile(0.99):.2f}")

zero_sales = (sales == 0).sum()
neg_sales = (sales < 0).sum()
print(f"\n  Zero Sales:     {zero_sales:,} ({fmt_pct(zero_sales, len(sales))})")
print(f"  Negative Sales: {neg_sales:,} ({fmt_pct(neg_sales, len(sales))})")

promo = df_train["onpromotion"]
print(f"\n  Promotion Distribution:")
print(f"    Active records:     {(promo > 0).sum():,} ({fmt_pct((promo > 0).sum(), len(promo))})")
print(f"    Mean (when active): {promo[promo > 0].mean():.2f}")
print(f"    Max:                {promo.max():.0f}")

promo_mask = df_train["onpromotion"] > 0
print(f"\n  Promotion Effect:")
print(f"    Avg Sales with Promo:    {sales[promo_mask].mean():.2f}")
print(f"    Avg Sales without Promo: {sales[~promo_mask].mean():.2f}")

df_train["year"] = df_train["date"].dt.year
yearly = df_train.groupby("year")["sales"].agg(["sum", "mean", "count"])
print(f"\n  Yearly Summary:")
for yr, row in yearly.iterrows():
    print(f"    {yr}: total={row['sum']:,.0f}, mean={row['mean']:.2f}, records={row['count']:,}")


# ----------------------------------------------------------
# 2.3 stores.csv Analysis
# ----------------------------------------------------------
print_subsection("2.3 stores.csv Analysis")

print(f"\n  Store Type Distribution:")
for t, cnt in df_stores["type"].value_counts().sort_index().items():
    print(f"    Type {t}: {cnt} stores")

print(f"\n  City Distribution ({df_stores['city'].nunique()} cities):")
for city, cnt in df_stores["city"].value_counts().items():
    print(f"    {city}: {cnt}")

print(f"\n  State Distribution:")
for st, cnt in df_stores["state"].value_counts().items():
    print(f"    {st}: {cnt}")

print(f"\n  Cluster Distribution ({df_stores['cluster'].nunique()} clusters):")
for cl, cnt in df_stores["cluster"].value_counts().sort_index().items():
    print(f"    Cluster {cl}: {cnt}")


# ----------------------------------------------------------
# 2.4 holidays_events.csv Analysis
# ----------------------------------------------------------
print_subsection("2.4 holidays_events.csv Analysis")

print(f"\n  Event Type Distribution:")
for t, cnt in df_holidays["type"].value_counts().items():
    print(f"    {t}: {cnt}")

print(f"\n  Locale Coverage:")
for loc, cnt in df_holidays["locale"].value_counts().items():
    print(f"    {loc}: {cnt}")

# transferred count — handle both bool and string types
if df_holidays["transferred"].dtype == bool:
    tx_count = df_holidays["transferred"].sum()
else:
    tx_count = (df_holidays["transferred"].astype(str).str.lower() == "true").sum()
print(f"  transferred=True: {tx_count}")
print(f"  Unique locales:   {df_holidays['locale_name'].nunique()}")

national = df_holidays[df_holidays["locale"] == "National"]["description"].unique()
print(f"\n  National Holidays/Events ({len(national)} total):")
for desc in sorted(national):
    print(f"    - {desc}")


# ----------------------------------------------------------
# 2.5 oil.csv Analysis
# ----------------------------------------------------------
print_subsection("2.5 oil.csv Analysis")

missing_oil = df_oil["dcoilwtico"].isnull()
print(f"\n  Missing Records: {missing_oil.sum():,} / {len(df_oil):,} ({fmt_pct(missing_oil.sum(), len(df_oil))})")

df_oil_temp = df_oil.copy()
df_oil_temp["dayofweek"] = df_oil_temp["date"].dt.dayofweek
missing_by_dow = df_oil_temp[missing_oil]["dayofweek"].value_counts().sort_index()
print(f"\n  Missing by Day of Week:")
for dow, cnt in missing_by_dow.items():
    print(f"    {DOW_NAMES[dow]}: {cnt} days")

oil_prices = df_oil["dcoilwtico"].dropna()
print(f"\n  Oil Price Statistics:")
print(f"    Mean:   {oil_prices.mean():.2f}")
print(f"    Min:    {oil_prices.min():.2f} ({df_oil.loc[oil_prices.idxmin(), 'date'].date()})")
print(f"    Max:    {oil_prices.max():.2f} ({df_oil.loc[oil_prices.idxmax(), 'date'].date()})")
print(f"    Start:  {oil_prices.iloc[0]:.2f}")
print(f"    End:    {oil_prices.iloc[-1]:.2f}")

df_oil_temp["year"] = df_oil_temp["date"].dt.year
yearly_oil = df_oil_temp.groupby("year")["dcoilwtico"].agg(["mean", "min", "max"])
print(f"\n  Yearly Oil Price:")
for yr, row in yearly_oil.iterrows():
    print(f"    {yr}: mean={row['mean']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}")


# ----------------------------------------------------------
# 2.6 transactions.csv Analysis
# ----------------------------------------------------------
print_subsection("2.6 transactions.csv Analysis")

daily_sales = df_train.groupby(["date", "store_nbr"])["sales"].sum().reset_index()
daily_sales.rename(columns={"sales": "total_sales"}, inplace=True)
merged_tx_sales = df_transactions.merge(daily_sales, on=["date", "store_nbr"], how="inner")
corr = 0.0
if len(merged_tx_sales) > 0:
    corr = merged_tx_sales["transactions"].corr(merged_tx_sales["total_sales"])
    print(f"\n  Correlation (transactions vs sales): {corr:.4f}")

tx = df_transactions["transactions"]
print(f"\n  Transaction Stats:")
print(f"    Mean:   {tx.mean():.1f}")
print(f"    Median: {tx.median():.1f}")
print(f"    Min:    {tx.min():.0f}")
print(f"    Max:    {tx.max():.0f}")

print(f"\n  test.csv Date Range:  {df_test['date'].min().date()} ~ {df_test['date'].max().date()}")
print(f"  test.csv Unique Days: {df_test['date'].nunique()}")


# ============================================================
# 3. Data Cleaning
# ============================================================

print_section("3. Data Cleaning")

# ----------------------------------------------------------
# 3.1 oil.csv — Forward fill missing values
# ----------------------------------------------------------
print_subsection("3.1 Cleaning oil.csv")

df_oil_clean = df_oil.sort_values("date").reset_index(drop=True).copy()

zero_oil = (df_oil_clean["dcoilwtico"] == 0).sum()
if zero_oil > 0:
    print(f"  [WARNING] Found {zero_oil} rows with oil price = 0 → setting to NaN")
    df_oil_clean.loc[df_oil_clean["dcoilwtico"] == 0, "dcoilwtico"] = np.nan

before_missing = df_oil_clean["dcoilwtico"].isnull().sum()
df_oil_clean["dcoilwtico"] = df_oil_clean["dcoilwtico"].ffill()
df_oil_clean["dcoilwtico"] = df_oil_clean["dcoilwtico"].bfill()
after_missing = df_oil_clean["dcoilwtico"].isnull().sum()
print(f"  Missing: {before_missing} → {after_missing} (ffill + bfill)")
assert after_missing == 0, "Oil still has missing values!"


# ----------------------------------------------------------
# 3.2 train.csv — Negative sales & outlier detection
# ----------------------------------------------------------
print_subsection("3.2 Cleaning train.csv")

df_train_clean = df_train.copy()

neg_mask = df_train_clean["sales"] < 0
if neg_mask.sum() > 0:
    print(f"  [WARNING] {neg_mask.sum()} negative sales → set to 0")
    df_train_clean.loc[neg_mask, "sales"] = 0.0
else:
    print(f"  Negative sales: 0 — clean.")

Q1 = df_train_clean["sales"].quantile(0.25)
Q3 = df_train_clean["sales"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR
outliers = (df_train_clean["sales"] > upper_bound).sum()
print(f"  IQR Upper Bound (3×IQR): {upper_bound:.0f}")
print(f"  Records above bound:    {outliers:,} ({fmt_pct(outliers, len(df_train_clean))})")
print(f"  Cleaned rows: {len(df_train_clean):,}")


# ----------------------------------------------------------
# 3.3 holidays_events.csv — transferred + encoding
# ----------------------------------------------------------
print_subsection("3.3 Cleaning holidays_events.csv")

df_holidays_clean = df_holidays.copy()

if df_holidays_clean["transferred"].dtype != bool:
    df_holidays_clean["transferred"] = df_holidays_clean["transferred"].map(
        {"True": True, "False": False}
    )

df_holidays_clean["is_national"] = (df_holidays_clean["locale"] == "National").astype(int)
df_holidays_clean["is_regional"] = (df_holidays_clean["locale"] == "Regional").astype(int)
df_holidays_clean["is_local"] = (df_holidays_clean["locale"] == "Local").astype(int)
df_holidays_clean["type_code"] = df_holidays_clean["type"].map(EVENT_TYPE_MAP)

holiday_dates = df_holidays_clean.groupby("date").agg(
    has_holiday=("type_code", lambda x: 1 if (x == 1).any() else 0),
    has_event=("type_code", lambda x: 1 if (x.isin([2, 3, 4, 5]).any()) else 0),
    is_national=("is_national", "max"),
    is_regional=("is_regional", "max"),
    is_local=("is_local", "max"),
).reset_index()

print(f"  Unique holiday dates: {len(holiday_dates)}")
print(f"    Has Holiday: {holiday_dates['has_holiday'].sum()}")
print(f"    Has Event:   {holiday_dates['has_event'].sum()}")


# ----------------------------------------------------------
# 3.4 transactions.csv
# ----------------------------------------------------------
print_subsection("3.4 Cleaning transactions.csv")

df_transactions_clean = df_transactions.copy()
df_transactions_clean["date"] = pd.to_datetime(df_transactions_clean["date"])

neg_tx = (df_transactions_clean["transactions"] < 0).sum()
if neg_tx > 0:
    print(f"  [WARNING] {neg_tx} negative transactions → set to 0")
    df_transactions_clean.loc[df_transactions_clean["transactions"] < 0, "transactions"] = 0

print(f"  Rows: {len(df_transactions_clean):,}")
print(f"  Dates: {df_transactions_clean['date'].nunique()}")
print(f"  Stores: {df_transactions_clean['store_nbr'].nunique()}")


# ----------------------------------------------------------
# 3.5 stores.csv — Ordinal encoding
# ----------------------------------------------------------
print_subsection("3.5 Cleaning stores.csv")

df_stores_clean = df_stores.copy()
df_stores_clean["type_code"] = df_stores_clean["type"].map(STORE_TYPE_MAP)
print(f"  Rows: {len(df_stores_clean)}")
print(f"  type_code mapping: {STORE_TYPE_MAP}")


# ============================================================
# 4. Merge Core Tables
# ============================================================

print_section("4. Merging Core Tables")

print("Merging: train ← stores ← oil ← holidays ← transactions ...")

df_merged = df_train_clean.copy()

# 4.1 stores
df_merged = df_merged.merge(
    df_stores_clean[["store_nbr", "city", "state", "type", "cluster", "type_code"]],
    on="store_nbr", how="left",
)

# 4.2 oil
df_merged = df_merged.merge(
    df_oil_clean[["date", "dcoilwtico"]], on="date", how="left",
)

# 4.3 holidays
df_merged = df_merged.merge(holiday_dates, on="date", how="left")
for col in ["has_holiday", "has_event", "is_national", "is_regional", "is_local"]:
    df_merged[col] = df_merged[col].fillna(0).astype(int)

# 4.4 transactions
df_merged = df_merged.merge(
    df_transactions_clean[["date", "store_nbr", "transactions"]],
    on=["date", "store_nbr"], how="left",
)

print(f"  Merged: {df_merged.shape[0]:,} rows × {df_merged.shape[1]} cols")

new_cols = [c for c in df_merged.columns if c not in df_train_clean.columns]
print(f"  New columns: {new_cols}")

print(f"\n  Missing After Merge:")
for col in df_merged.columns:
    m = df_merged[col].isnull().sum()
    if m > 0:
        print(f"    {col}: {m:,} ({fmt_pct(m, len(df_merged))})")


# ============================================================
# 5. Export Cleaned CSV Files
# ============================================================

print_section("5. Exporting Cleaned CSV Files")

# Drop temporary columns
if "year" in df_merged.columns:
    df_merged.drop(columns=["year"], inplace=True)

outputs = {
    "train_cleaned.csv": df_merged,
    "stores_cleaned.csv": df_stores_clean,
    "oil_cleaned.csv": df_oil_clean,
    "holidays_cleaned.csv": df_holidays_clean,
    "transactions_cleaned.csv": df_transactions_clean,
}

for filename, df_out in outputs.items():
    out_path = os.path.join(DATA_PROCESSED_DIR, filename)
    df_out.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"  {filename:30s} → {df_out.shape[0]:>10,} rows × {df_out.shape[1]:>3} cols ({size_mb:.1f} MB)")

print(f"\n[INFO] Exported to: {DATA_PROCESSED_DIR}")


# ============================================================
# 6. Final Summary
# ============================================================

print_section("6. Key Findings Summary")

promo_pct = (df_train_clean["onpromotion"] > 0).sum() / len(df_train_clean) * 100

print(f"""
  {'─' * 58}
                        KEY FINDINGS
  {'─' * 58}
  1. Scale:         train ~{len(df_train_clean):,} rows, test ~{len(df_test):,} rows
  2. Time Range:    {df_train_clean['date'].min().date()} ~ {df_train_clean['date'].max().date()} (train)
  3. Target:        {TEST_START_DATE} to {TEST_END_DATE}, {df_stores_clean['store_nbr'].nunique()} stores × {df_train_clean['family'].nunique()} families
  4. Stores:        {df_stores_clean['store_nbr'].nunique()} stores, {df_stores_clean['city'].nunique()} cities, types A-E
  5. Families:      {df_train_clean['family'].nunique()} product families
  6. Promotions:    {promo_pct:.1f}% of records have active promotions
  7. Oil:           weekends/holidays missing → forward filled
  8. Holidays:      National/Regional/Local levels, with transfer & bridge days
  9. Transactions:  Strongly correlated with sales (r ≈ {corr:.2f})
  10. Special:      2016 Manabi earthquake, 2014 World Cup events
  {'─' * 58}
""")

print(f"[INFO] Step 1 complete. Cleaned data → {DATA_PROCESSED_DIR}")
