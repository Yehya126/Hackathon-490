"""Leakage-safe transaction-level chronological splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransactionSplits:
    """Line-level frames whose transaction memberships never overlap."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def transaction_ids(self) -> dict[str, pd.Index]:
        return {
            name: pd.Index(part["Transaction ID"].drop_duplicates())
            for name, part in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        }


def chronological_transaction_split(
    frame: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> TransactionSplits:
    """Split whole transactions in timestamp order using transaction counts.

    Transactions sharing the boundary timestamp are kept in the earlier split.
    This may make proportions differ slightly, but prevents simultaneous orders
    from being arbitrarily separated and keeps all rows of an order together.
    """
    if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
        raise ValueError("Split proportions must sum to 1")
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("Split proportions must all be positive")
    required = {"Transaction ID", "Transaction Timestamp"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot split an empty dataset")

    tx = (
        frame.groupby("Transaction ID", sort=False, observed=True)
        .agg(
            timestamp=("Transaction Timestamp", "min"),
            timestamp_count=("Transaction Timestamp", "nunique"),
        )
        .sort_values(["timestamp"], kind="stable")
    )
    if (tx["timestamp_count"] != 1).any():
        raise ValueError("Every transaction must have exactly one timestamp")

    count = len(tx)
    if count < 3:
        raise ValueError("At least three transactions are required")
    train_target = max(1, int(count * train_size))
    validation_target = max(1, int(count * validation_size))

    train_boundary = tx.iloc[min(train_target - 1, count - 3)]["timestamp"]
    train_ids = tx.index[tx["timestamp"] <= train_boundary]
    remaining = tx.loc[~tx.index.isin(train_ids)]
    validation_position = min(validation_target - 1, len(remaining) - 2)
    validation_boundary = remaining.iloc[validation_position]["timestamp"]
    validation_ids = remaining.index[remaining["timestamp"] <= validation_boundary]
    test_ids = remaining.index[~remaining.index.isin(validation_ids)]

    if not len(test_ids):
        raise ValueError("Split boundaries left no test transactions")

    def select(ids: pd.Index) -> pd.DataFrame:
        # Preserve original indexes so source rows and future predictions align.
        return frame.loc[frame["Transaction ID"].isin(ids)].copy()

    return TransactionSplits(
        train=select(train_ids),
        validation=select(validation_ids),
        test=select(test_ids),
    )

