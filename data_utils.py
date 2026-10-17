"""Shared loading, splitting, and basket-preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("Transaction ID", "Date", "Branch", "Product", "Quantity", "Line Gross Profit (USD)")


def load_data(path: str | Path = "stories_transactions.csv") -> pd.DataFrame:
    """Load the raw transaction lines and create a sortable timestamp."""
    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data["Transaction Timestamp"] = pd.to_datetime(
        data["Date"].dt.strftime("%Y-%m-%d") + " " + data["Time"].astype(str), errors="raise"
    )
    return data


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_data(data: pd.DataFrame) -> DataSplits:
    """Chronologically split whole transactions 70%/15%/15%."""
    times = data.groupby("Transaction ID", sort=False)["Transaction Timestamp"].agg(["min", "nunique"])
    times = times.sort_values("min", kind="stable")
    if (times["nunique"] != 1).any():
        raise ValueError("Every transaction must have exactly one timestamp")

    count = len(times)
    train_end = times.iloc[int(count * 0.70) - 1]["min"]
    train_ids = times.index[times["min"] <= train_end]
    remaining = times.loc[~times.index.isin(train_ids)]
    validation_end = remaining.iloc[int(count * 0.15) - 1]["min"]
    validation_ids = remaining.index[remaining["min"] <= validation_end]
    test_ids = remaining.index[~remaining.index.isin(validation_ids)]

    def rows(ids: pd.Index) -> pd.DataFrame:
        return data.loc[data["Transaction ID"].isin(ids)].copy()

    return DataSplits(rows(train_ids), rows(validation_ids), rows(test_ids))


def make_baskets(train: pd.DataFrame, *other_splits: pd.DataFrame) -> list[pd.DataFrame]:
    """Create boolean baskets using only the product vocabulary in training."""
    products = pd.Index(sorted(train["Product"].unique()), name="Product")

    def encode(data: pd.DataFrame) -> pd.DataFrame:
        basket = pd.crosstab(data["Transaction ID"], data["Product"]).gt(0)
        return basket.reindex(columns=products, fill_value=False).astype(bool)

    return [encode(data) for data in (train, *other_splits)]


def typical_unit_profit(train: pd.DataFrame) -> pd.Series:
    """Estimate median product profit using training rows only."""
    if (train["Quantity"] <= 0).any():
        raise ValueError("Quantity must be positive")
    unit_profit = train["Line Gross Profit (USD)"] / train["Quantity"]
    return train.assign(_unit_profit=unit_profit).groupby("Product")["_unit_profit"].median()

