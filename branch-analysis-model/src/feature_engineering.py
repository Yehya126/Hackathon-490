"""Quantity-weighted analytical features and matched-product panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


PERIOD_CODES = {"daily": "D", "weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y"}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def aggregate_features(df: pd.DataFrame, period: str = "monthly") -> pd.DataFrame:
    if period not in PERIOD_CODES:
        raise ValueError(f"Unsupported reporting period {period!r}; choose from {sorted(PERIOD_CODES)}")
    work = df.copy()
    work["period"] = work["Date"].dt.to_period(PERIOD_CODES[period]).astype(str)
    keys = ["period", "Branch", "Product", "Category"]
    grouped = work.groupby(keys, observed=True, dropna=False)
    out = grouped.agg(
        total_units=("Quantity", "sum"), total_revenue=("Line Revenue (USD)", "sum"),
        total_cogs=("calculated_cogs", "sum"), total_gross_profit=("Line Gross Profit (USD)", "sum"),
        listed_revenue=("listed_revenue", "sum"), discount_amount=("discount_amount", "sum"),
        transaction_count=("Transaction ID", "nunique"), observation_count=("Transaction ID", "size"),
    ).reset_index()
    out["gross_margin"] = _safe_divide(out["total_gross_profit"], out["total_revenue"])
    out["listed_unit_price"] = _safe_divide(out["listed_revenue"], out["total_units"])
    out["unit_cost"] = _safe_divide(out["total_cogs"], out["total_units"])
    out["realized_revenue_per_unit"] = _safe_divide(out["total_revenue"], out["total_units"])
    out["effective_discount_rate"] = _safe_divide(out["discount_amount"], out["listed_revenue"])
    period_values = work["Date"].dt.to_period(PERIOD_CODES[period])
    completeness = {
        str(p): bool(p.start_time.normalize() >= work["Date"].min().normalize()
                     and p.end_time.normalize() <= work["Date"].max().normalize())
        for p in period_values.unique()
    }
    out["period_complete"] = out["period"].map(completeness).fillna(False)
    branch_period = out.groupby(["period", "Branch"], observed=True)
    for value, name in (
        ("total_units", "share_branch_sales_volume"),
        ("total_revenue", "share_branch_revenue"),
        ("total_gross_profit", "share_branch_gross_profit"),
    ):
        totals = branch_period[value].transform("sum")
        out[name] = _safe_divide(out[value], totals)
    return out.sort_values(keys).reset_index(drop=True)


def branch_product_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["Branch", "Product", "Category"], observed=True)
    out = grouped.agg(
        total_units=("Quantity", "sum"), total_revenue=("Line Revenue (USD)", "sum"),
        total_cogs=("calculated_cogs", "sum"), total_gross_profit=("Line Gross Profit (USD)", "sum"),
        listed_revenue=("listed_revenue", "sum"), discount_amount=("discount_amount", "sum"),
        periods=("Date", lambda x: x.dt.to_period("M").nunique()),
    ).reset_index()
    out["listed_unit_price"] = _safe_divide(out["listed_revenue"], out["total_units"])
    out["unit_cost"] = _safe_divide(out["total_cogs"], out["total_units"])
    out["realized_revenue_per_unit"] = _safe_divide(out["total_revenue"], out["total_units"])
    out["effective_discount_rate"] = _safe_divide(out["discount_amount"], out["listed_revenue"])
    out["gross_margin"] = _safe_divide(out["total_gross_profit"], out["total_revenue"])
    counts = out.groupby("Product")["Branch"].transform("nunique")
    out["matched_branch_count"] = counts
    return out.sort_values(["Product", "Branch"]).reset_index(drop=True)


def matched_product_table(branch_product: pd.DataFrame, minimum_branches: int = 2) -> pd.DataFrame:
    """Long-form comparison panel; no repeated line prices/costs enter the models."""
    return branch_product.loc[branch_product["matched_branch_count"] >= minimum_branches].copy()


def structural_fingerprint(branch_product: pd.DataFrame) -> str:
    """Hash only structural price/cost inputs, allowing quantity-only refreshes to skip refits."""
    import hashlib
    cols = ["Branch", "Product", "unit_cost", "listed_unit_price"]
    stable = branch_product[cols].sort_values(["Branch", "Product"]).to_csv(index=False, float_format="%.10g")
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()
