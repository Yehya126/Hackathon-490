"""Prep-quantity recommendation (newsvendor on the NB demand distribution) and profit simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast import nb_quantile

# Cost of one unprepared-but-unsold unit, as a share of unit cost.
OVERAGE_SHARE = {
    "fresh": 1.0,          # prepared ahead and binned at close
    "made_to_order": 0.1,  # only ingredients are stocked; small spoilage
    "packaged": 0.05,      # holding cost; stock carries over
}


def critical_ratio(df: pd.DataFrame) -> pd.Series:
    overage = df["unit_cost"] * df["shelf_class"].astype(str).map(OVERAGE_SHARE)
    return df["unit_margin"] / (df["unit_margin"] + overage)


def recommend(df: pd.DataFrame, mu_col: str, alpha: pd.Series) -> pd.Series:
    """Units to prepare: the NB quantile at each product's critical ratio."""
    return pd.Series(nb_quantile(df[mu_col], alpha, critical_ratio(df)), index=df.index)


def simulate(df: pd.DataFrame, prep_col: str) -> pd.DataFrame:
    """Treat actual sales as demand; return per-row sold, waste, missed and profit."""
    prep = np.maximum(np.round(df[prep_col].fillna(0)), 0)
    sold = np.minimum(prep, df["units"])
    leftover = prep - sold
    overage_cost = df["unit_cost"] * df["shelf_class"].astype(str).map(OVERAGE_SHARE)
    return pd.DataFrame({
        "Branch": df["Branch"].astype(str), "shelf_class": df["shelf_class"].astype(str),
        "prepared": prep, "sold": sold, "missed": df["units"] - sold,
        "leftover": leftover, "waste_cost": leftover * overage_cost,
        "missed_profit": (df["units"] - sold) * df["unit_margin"],
        "profit": sold * df["unit_margin"] - leftover * overage_cost,
    })


def summarize(sim: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = ["prepared", "sold", "missed", "leftover", "waste_cost", "missed_profit", "profit"]
    out = sim.groupby(by)[cols].sum()
    out["service_level"] = out["sold"] / (out["sold"] + out["missed"])
    return out.round(1)
