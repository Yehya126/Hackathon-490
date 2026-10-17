"""Leak-safe features for next-day forecasting: every value uses data up to the previous day."""

from __future__ import annotations

import pandas as pd

from panel import KEYS, TRAIN_END

CATEGORICAL = ["Branch", "Product", "Category", "shelf_class"]
NUMERIC = [
    "lag_1", "lag_7", "lag_14", "lag_28", "same_weekday_mean_4",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "branch_roll_mean_7", "branch_roll_mean_28",
    "weekday", "day_of_month", "trend", "price_ratio", "unit_margin",
]
FEATURES = CATEGORICAL + NUMERIC


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(KEYS + ["Date"]).copy()
    units = df.groupby(KEYS, sort=False)["units"]
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = units.shift(lag)
    df["lag_21"] = units.shift(21)
    df["same_weekday_mean_4"] = df[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1, skipna=False)
    previous = units.shift(1)
    grouped_prev = previous.groupby([df[k] for k in KEYS], sort=False)
    for window in (7, 14, 28):
        df[f"roll_mean_{window}"] = grouped_prev.transform(lambda s: s.rolling(window, min_periods=window).mean())
    df["roll_std_7"] = grouped_prev.transform(lambda s: s.rolling(7, min_periods=7).std())

    branch = df.groupby(["Branch", "Date"])["units"].sum().rename("branch_units").reset_index()
    branch_prev = branch.groupby("Branch")["branch_units"].shift(1)
    for window in (7, 28):
        branch[f"branch_roll_mean_{window}"] = branch_prev.groupby(branch["Branch"]).transform(
            lambda s: s.rolling(window, min_periods=window).mean())
    df = df.merge(branch.drop(columns="branch_units"), on=["Branch", "Date"], how="left")

    df["weekday"] = df["Date"].dt.dayofweek
    df["day_of_month"] = df["Date"].dt.day
    df["trend"] = (df["Date"] - df["Date"].min()).dt.days
    reference = df[df["Date"] <= TRAIN_END].groupby("Product")["price"].median()
    df["price_ratio"] = df["price"] / df["Product"].map(reference)
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df.drop(columns="lag_21")
