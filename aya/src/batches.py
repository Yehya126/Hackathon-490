"""Two-batch prep for fresh products: a morning batch, then an afternoon top-up decided from morning sales.

Demand model: the day's demand is negative binomial = Poisson with a gamma-distributed rate
multiplier (mean 1, variance alpha). Morning sales reveal that multiplier, so afternoon demand
given morning sales is again negative binomial with
    n    = 1/alpha + morning_sold
    mean = afternoon_expected * (1/alpha + morning_sold) / (1/alpha + morning_expected)
A strong morning raises the afternoon expectation; a weak morning lowers it.
Fresh items are binned at close (nothing carries to the next day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from inventory import critical_ratio
from panel import KEYS, TRAIN_END

SHRINK = 50  # pseudo-units pulling a product's morning share toward its category's share


def morning_units(data: pd.DataFrame, cutoff_hour: int) -> pd.Series:
    """Units sold before the cutoff hour, per branch, product and day."""
    am = data[data["Hour"] < cutoff_hour]
    return am.groupby(KEYS + ["Date"])["Quantity"].sum().rename("units_am")


def morning_share(data: pd.DataFrame, cutoff_hour: int) -> pd.Series:
    """Share of a product's daily demand that falls before the cutoff, fitted on training days only."""
    train = data[data["Date"] <= TRAIN_END].assign(am=lambda d: d["Hour"] < cutoff_hour)
    prod = train.groupby(KEYS + ["Category"]).apply(
        lambda g: pd.Series({"am": g.loc[g["am"], "Quantity"].sum(), "total": g["Quantity"].sum()}),
        include_groups=False).reset_index()
    cat = prod.groupby(["Branch", "Category"])[["am", "total"]].sum()
    cat_share = (cat["am"] / cat["total"]).rename("cat_share").reset_index()
    prod = prod.merge(cat_share, on=["Branch", "Category"])
    prod["share"] = (prod["am"] + SHRINK * prod["cat_share"]) / (prod["total"] + SHRINK)
    return prod.set_index(KEYS)["share"]


def attach(df: pd.DataFrame, data: pd.DataFrame, cutoff_hour: int) -> pd.DataFrame:
    out = df.copy()
    keys = pd.MultiIndex.from_frame(out[KEYS + ["Date"]].astype({"Branch": str, "Product": str}))
    out["units_am"] = morning_units(data, cutoff_hour).reindex(keys).fillna(0).to_numpy()
    share = morning_share(data, cutoff_hour)
    out["am_share"] = share.reindex(pd.MultiIndex.from_frame(out[KEYS].astype(str))).to_numpy()
    return out


def _nb_ppf(q, n, mean):
    n = np.asarray(n, dtype=float)
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-6)
    return stats.nbinom(n, n / (n + mean)).ppf(q)


def two_batch(df: pd.DataFrame, mu_col: str, alpha: pd.Series, morning_level: float | None = None) -> pd.DataFrame:
    """Simulate the two-batch policy on actual sales (treated as demand). Fresh rows only.

    morning_level: quantile of morning demand to bake in the first batch (None = the product's
    critical ratio). Unsold morning stock stays on the shelf for the afternoon.
    """
    a = np.asarray(alpha, dtype=float)
    mu = df[mu_col].to_numpy()
    share = df["am_share"].to_numpy()
    cr = critical_ratio(df).to_numpy()
    mu_am, mu_pm = mu * share, mu * (1 - share)

    q1 = _nb_ppf(cr if morning_level is None else morning_level, 1 / a, mu_am)
    d_am = df["units_am"].to_numpy()
    d_pm = df["units"].to_numpy() - d_am
    sold_am = np.minimum(q1, d_am)
    left = q1 - sold_am

    # Afternoon decision uses what was sold in the morning (if the morning sold out, this understates demand).
    n_post = 1 / a + sold_am
    mean_pm = mu_pm * n_post / (1 / a + mu_am)
    target = _nb_ppf(cr, n_post, mean_pm)
    q2 = np.maximum(target - left, 0)
    stock_pm = left + q2
    sold_pm = np.minimum(stock_pm, d_pm)
    waste = stock_pm - sold_pm

    cost, margin = df["unit_cost"].to_numpy(), df["unit_margin"].to_numpy()
    sold = sold_am + sold_pm
    return pd.DataFrame({
        "Branch": df["Branch"].astype(str).to_numpy(), "morning_batch": q1, "afternoon_batch": q2,
        "prepared": q1 + q2, "sold": sold, "missed": df["units"].to_numpy() - sold, "leftover": waste,
        "waste_cost": waste * cost, "missed_profit": (df["units"].to_numpy() - sold) * margin,
        "profit": sold * margin - waste * cost,
    }, index=df.index)


def morning_plan(df: pd.DataFrame, mu_col: str, alpha: pd.Series, morning_level: float | None) -> pd.Series:
    """Tomorrow's first batch (the afternoon top-up is decided live from morning sales)."""
    a = np.asarray(alpha, dtype=float)
    cr = critical_ratio(df).to_numpy()
    return pd.Series(_nb_ppf(cr if morning_level is None else morning_level, 1 / a,
                             df[mu_col].to_numpy() * df["am_share"].to_numpy()), index=df.index)


def afternoon_topup(expected_day: float, am_share: float, alpha: float, morning_sold: float,
                    on_shelf: float, critical: float) -> float:
    """Live rule for staff at the cutoff hour: how many more to make now."""
    n_post = 1 / alpha + morning_sold
    mean_pm = expected_day * (1 - am_share) * n_post / (1 / alpha + expected_day * am_share)
    return float(max(_nb_ppf(critical, n_post, mean_pm) - on_shelf, 0))
