"""Load and validate the raw transaction-line dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "Transaction ID",
    "Date",
    "Branch",
    "Product",
    "Quantity",
    "Line Gross Profit (USD)",
)


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load transaction lines without changing or dropping raw observations."""
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if "Time" in frame:
        timestamps = pd.to_datetime(
            frame["Date"].dt.strftime("%Y-%m-%d") + " " + frame["Time"].astype(str),
            errors="raise",
        )
    else:
        timestamps = frame["Date"]
    frame["Transaction Timestamp"] = timestamps

    if frame["Transaction ID"].isna().any():
        raise ValueError("Transaction ID contains missing values")
    if frame["Product"].isna().any():
        raise ValueError("Product contains missing values")
    return frame

