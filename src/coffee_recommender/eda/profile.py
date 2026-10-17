"""Compact, serializable EDA summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_transactions(frame: pd.DataFrame) -> dict[str, Any]:
    """Create a data-quality profile; call on training data for detailed EDA."""
    transaction_sizes = frame.groupby("Transaction ID")["Product"].nunique()
    numeric = frame.select_dtypes(include="number")
    return {
        "rows": int(len(frame)),
        "transactions": int(frame["Transaction ID"].nunique()),
        "columns": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_transaction_products": int(frame.duplicated(["Transaction ID", "Product"]).sum()),
        "date_range": {
            "start": frame["Transaction Timestamp"].min().isoformat(),
            "end": frame["Transaction Timestamp"].max().isoformat(),
        },
        "branch_line_counts": {str(k): int(v) for k, v in frame["Branch"].value_counts().items()},
        "product_count": int(frame["Product"].nunique()),
        "basket_size": transaction_sizes.describe().to_dict(),
        "numeric_summary": numeric.describe().to_dict(),
    }


def detailed_training_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Profile model-relevant patterns using training observations only.

    This is diagnostic output, not a production preprocessing step. Financial
    columns are summarized here but remain excluded from the basket matrix.
    """
    required = {
        "Transaction ID",
        "Branch",
        "Product",
        "Category",
        "Quantity",
        "Line Gross Profit (USD)",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing EDA columns: {sorted(missing)}")

    transaction_count = frame["Transaction ID"].nunique()
    product_transactions = (
        frame[["Transaction ID", "Product"]]
        .drop_duplicates()
        .groupby("Product")["Transaction ID"]
        .nunique()
        .sort_values(ascending=False)
    )
    product_support = (product_transactions / transaction_count).round(6)

    unique_items = frame[["Transaction ID", "Product"]].drop_duplicates()
    pairs = unique_items.merge(unique_items, on="Transaction ID")
    pairs = pairs.loc[pairs["Product_x"] < pairs["Product_y"]]
    pair_counts = pairs.value_counts(["Product_x", "Product_y"]).head(20)

    numeric = frame.select_dtypes(include="number")
    outliers: dict[str, int] = {}
    for column in numeric.columns:
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[column] = int(((numeric[column] < q1 - 1.5 * iqr) | (numeric[column] > q3 + 1.5 * iqr)).sum())

    unit_profit = frame["Line Gross Profit (USD)"] / frame["Quantity"]
    product_profit = (
        frame.assign(_unit_profit=unit_profit)
        .groupby("Product")["_unit_profit"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("median", ascending=False)
        .round(4)
    )

    return {
        "product_transaction_counts": {str(k): int(v) for k, v in product_transactions.items()},
        "product_support": {str(k): float(v) for k, v in product_support.items()},
        "category_line_counts": {str(k): int(v) for k, v in frame["Category"].value_counts().items()},
        "branch_transaction_counts": {
            str(k): int(v)
            for k, v in frame.groupby("Branch")["Transaction ID"].nunique().sort_values(ascending=False).items()
        },
        "top_product_pairs": {
            f"{left} + {right}": int(value) for (left, right), value in pair_counts.items()
        },
        "iqr_outlier_counts": outliers,
        "numeric_correlations": numeric.corr().round(4).to_dict(),
        "product_unit_profit": product_profit.to_dict(orient="index"),
    }

