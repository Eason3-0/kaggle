# -*- coding: utf-8 -*-
"""
Shared Feature Engineering Module — used by BOTH step2 and step4.
Guarantees 100% identical features between training and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import Fourier
from utils import Timer


def build_features(df, df_stores, df_oil, df_holidays, verbose=True):
    """
    Build all features on the given DataFrame (must have 'sales' column).

    Args:
        df: DataFrame with columns from train_cleaned.csv (date, store_nbr, family, sales, ...)
        df_stores: stores DataFrame
        df_oil: oil DataFrame (cleaned, no missing)
        df_holidays: holidays DataFrame (cleaned)

    Returns:
        df: DataFrame with all features + 'sales' + 'date' columns
        feature_cols: list of feature column names (excludes 'sales', 'date')
    """
    log = print if verbose else lambda *a, **k: None

    # ---- Calendar ----
    log("  Calendar features ...")
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = (df["day"] <= 3).astype(int)
    df["is_month_end"] = (df["day"] >= 28).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---- Trend ----
    log("  Trend features ...")
    trend_data = []
    for (store, family), group in df.groupby(["store_nbr", "family"]):
        # Only use non-NaN sales for fitting (test data has NaN sales)
        valid_mask = ~group["sales"].isna()
        X_all = group["days_since_start"].values.reshape(-1, 1)
        y_all = group["sales"].values
        X_valid = group.loc[valid_mask, "days_since_start"].values.reshape(-1, 1)
        y_valid = group.loc[valid_mask, "sales"].values

        if len(y_valid) < 30:
            trend_slope = 0.0
            trend_pred = np.full(len(group), np.nanmean(y_valid) if len(y_valid) > 0 else 0.0)
        else:
            lr = LinearRegression(); lr.fit(X_valid, y_valid)
            trend_slope = lr.coef_[0]
            trend_pred = lr.predict(X_all)
        trend_data.append({"store_nbr": store, "family": family,
                           "date": group["date"].values,
                           "trend_slope": trend_slope, "trend_pred": trend_pred})
    df_trend = pd.concat([
        pd.DataFrame({"date": r["date"], "store_nbr": r["store_nbr"],
                      "family": r["family"], "trend_slope": r["trend_slope"],
                      "trend_pred": r["trend_pred"]})
        for r in trend_data
    ], ignore_index=True)
    df = df.merge(df_trend, on=["date", "store_nbr", "family"], how="left")

    # ---- Fourier ----
    log("  Fourier features ...")
    ud = df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    ud["days_idx"] = range(len(ud))
    idx_s = ud["days_idx"]
    fy = Fourier(period=365.25, order=6).in_sample(idx_s)
    fw = Fourier(period=7, order=3).in_sample(idx_s)
    fy.columns = [f"fourier_year_{c}" for c in fy.columns]
    fw.columns = [f"fourier_week_{c}" for c in fw.columns]
    fourier_df = pd.concat([ud[["date"]].reset_index(drop=True),
                             fy.reset_index(drop=True), fw.reset_index(drop=True)], axis=1)
    df = df.merge(fourier_df, on="date", how="left")

    # ---- Lag ----
    log("  Lag features ...")
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)
    for lag in [1, 7]:
        df[f"onpromotion_lag_{lag}"] = df.groupby(["store_nbr", "family"])["onpromotion"].shift(lag)

    # ---- Rolling ----
    log("  Rolling features ...")
    for w in [7, 14, 30]:
        roll = df.groupby(["store_nbr", "family"])["sales"].rolling(w, min_periods=1)
        for stat in ["mean", "std", "min", "max"]:
            col = f"sales_rolling_{stat}_{w}d"
            df[col] = getattr(roll, stat)().reset_index(level=[0, 1], drop=True)
    for w in [7, 14]:
        rp = df.groupby(["store_nbr", "family"])["onpromotion"].rolling(w, min_periods=1)
        df[f"onpromotion_rolling_mean_{w}d"] = rp.mean().reset_index(level=[0, 1], drop=True)

    # ---- Holiday ----
    log("  Holiday features ...")
    holiday_dates_set = set(
        df_holidays[(df_holidays["is_national"] == 1) & (df_holidays["type_code"] == 1)]["date"].unique()
    )
    # Local/Regional matching
    store_city = df_stores.set_index("store_nbr")["city"].to_dict()
    store_state = df_stores.set_index("store_nbr")["state"].to_dict()

    city_holiday_dates = {}
    state_holiday_dates = {}
    for _, row in df_holidays.iterrows():
        if row["type_code"] != 1: continue
        if row["is_local"]:
            city_holiday_dates.setdefault(row["locale_name"], set()).add(row["date"])
        elif row["is_regional"]:
            state_holiday_dates.setdefault(row["locale_name"], set()).add(row["date"])

    # Build date-city and date-state DataFrames for merge
    local_h = df_holidays[(df_holidays["is_local"] == 1) & (df_holidays["type_code"] == 1)][["date", "locale_name"]]
    local_h.columns = ["date", "city"]; local_h["is_local_holiday"] = 1
    df = df.merge(local_h, on=["date", "city"], how="left")
    df["store_local_holiday"] = df["is_local_holiday"].fillna(0).astype(int)
    df.drop(columns=["is_local_holiday"], inplace=True)

    regional_h = df_holidays[(df_holidays["is_regional"] == 1) & (df_holidays["type_code"] == 1)][["date", "locale_name"]]
    regional_h.columns = ["date", "state"]; regional_h["is_regional_holiday_store"] = 1
    df = df.merge(regional_h, on=["date", "state"], how="left")
    df["store_regional_holiday"] = df["is_regional_holiday_store"].fillna(0).astype(int)
    df.drop(columns=["is_regional_holiday_store"], inplace=True)

    df["is_store_holiday"] = ((df["has_holiday"] == 1) | df["store_local_holiday"] | df["store_regional_holiday"]).astype(int)

    # Holiday proximity
    holiday_all = holiday_dates_set.copy()
    for s in city_holiday_dates.values(): holiday_all.update(s)
    for s in state_holiday_dates.values(): holiday_all.update(s)

    ud["days_to_any_holiday"] = ud["date"].apply(
        lambda d: min([abs((d - hd).days) for hd in holiday_all]) if holiday_all else 99
    )
    df = df.merge(ud[["date", "days_to_any_holiday"]], on="date", how="left")
    df["holiday_near_3d"] = (df["days_to_any_holiday"] <= 3).astype(int)
    df["holiday_near_7d"] = (df["days_to_any_holiday"] <= 7).astype(int)

    # ---- Oil ----
    log("  Oil features ...")
    df_oil_s = df_oil.sort_values("date").reset_index(drop=True)
    for lag in [1, 7, 14, 30]:
        df_oil_s[f"oil_lag_{lag}"] = df_oil_s["dcoilwtico"].shift(lag)
    for w in [7, 14, 30]:
        df_oil_s[f"oil_rolling_mean_{w}d"] = df_oil_s["dcoilwtico"].rolling(w, min_periods=1).mean()
    df_oil_s["oil_change_pct"] = df_oil_s["dcoilwtico"].pct_change().fillna(0)
    oil_cols = ["date"] + [c for c in df_oil_s.columns if c.startswith("oil_") or c == "dcoilwtico"]
    df = df.merge(df_oil_s[oil_cols], on="date", how="left", suffixes=("", "_oil"))
    for col in [c for c in df.columns if c.startswith("oil_") or c == "dcoilwtico"]:
        df[col] = df[col].ffill().bfill()

    # ---- OHE ----
    log("  One-hot encoding ...")
    df = pd.get_dummies(df, columns=["family", "city", "state", "type"], drop_first=False)

    # ---- Interactions ----
    log("  Interaction features ...")
    df["weekend_holiday"] = df["is_weekend"] * df["is_store_holiday"]
    df["promo_weekend"] = (df["onpromotion"] > 0).astype(int) * df["is_weekend"]
    df["promo_holiday"] = (df["onpromotion"] > 0).astype(int) * df["is_store_holiday"]
    df["dow_month"] = df["dayofweek"] * 12 + df["month"]

    # ---- Cleanup ----
    # Drop non-feature columns (but KEEP 'id' — caller may need it for ordering)
    drop_cols = ["trend_intercept", "store_local_holiday", "store_regional_holiday",
                 "days_to_any_holiday", "days_since_start", "quarter"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # NOTE: Do NOT call fillna(0) here!
    # NaN in lag columns must be handled by the caller:
    #   - Step 2 (training): drop NaN lag rows first, THEN fillna(0)
    #   - Step 4 (prediction): ffill from training data for test rows

    feature_cols = [c for c in df.columns if c not in ["sales", "date"]]

    return df, feature_cols
