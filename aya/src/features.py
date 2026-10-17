"""Leak-safe features for next-day forecasting: every value uses data up to the previous day."""

from __future__ import annotations

import numpy as np
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
# Extra inputs for the baseline-offset model: everything relative to the baseline.
RELATIVE = ["baseline", "wd_index", "branch_growth", "same_weekday_rel", "roll_mean_7_rel", "lag_1_rel"]
OFFSET_FEATURES = FEATURES + RELATIVE
GROWTH_CENTER_DAYS = 14.5  # a 28-day mean ending yesterday is centred 14.5 days before today


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
    df = _add_baseline(df)
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df.drop(columns="lag_21")


def _add_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline = recent level x branch weekday index x branch growth since that level was measured.

    The weekday index is fitted on training days only; level and growth use data up to yesterday.
    """
    train = df[df["Date"] <= TRAIN_END]
    by_day = train.groupby(["Branch", "Date"])["units"].sum().reset_index()
    by_day["weekday"] = by_day["Date"].dt.dayofweek
    index = by_day.groupby(["Branch", "weekday"])["units"].mean()
    index = index / index.groupby(level="Branch").transform("mean")
    lookup = lambda branch, weekday: pd.Series(  # noqa: E731
        index.reindex(pd.MultiIndex.from_arrays([branch, weekday])).to_numpy(), index=branch.index)
    df["wd_index"] = lookup(df["Branch"], df["weekday"])
    yesterday_index = lookup(df["Branch"], (df["weekday"] - 1) % 7)

    branch = df.drop_duplicates(["Branch", "Date"])[["Branch", "Date", "branch_roll_mean_28"]].copy()
    earlier = branch.groupby("Branch")["branch_roll_mean_28"].shift(28)
    branch["branch_growth"] = (np.log(branch["branch_roll_mean_28"] / earlier) / 28).clip(-0.02, 0.02)
    df = df.merge(branch[["Branch", "Date", "branch_growth"]], on=["Branch", "Date"], how="left")

    level = df["roll_mean_28"].clip(lower=0.1)
    df["baseline"] = level * df["wd_index"] * np.exp(df["branch_growth"].fillna(0) * GROWTH_CENTER_DAYS)
    df["same_weekday_rel"] = df["same_weekday_mean_4"] / df["baseline"]
    df["roll_mean_7_rel"] = df["roll_mean_7"] / df["baseline"]
    # Yesterday judged against what yesterday's weekday normally sells (stops a Tuesday peak leaking into Wednesday).
    df["lag_1_rel"] = df["lag_1"] / (level * yesterday_index)
    return df
