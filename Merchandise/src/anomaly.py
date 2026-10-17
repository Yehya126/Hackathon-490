"""Within-branch anomaly detection: actual sales vs the series' own NB demand distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast import nb_cdf, nb_sf

P_THRESHOLD = 0.01   # tail probability below which a day is unusual
MIN_GAP = 3          # minimum absolute unit gap for product-level alerts
P_ZERO = 0.02        # a zero-sales day is a possible stockout when P(0) is below this


def score_products(df: pd.DataFrame, mu_col: str, alpha: pd.Series) -> pd.DataFrame:
    out = df[["Branch", "Product", "Category", "shelf_class", "Date", "units", "unit_margin"]].copy()
    out["expected"] = df[mu_col]
    out["p_low"] = nb_cdf(df["units"], df[mu_col], alpha)
    out["p_high"] = nb_sf(df["units"], df[mu_col], alpha)
    gap = out["units"] - out["expected"]
    zero = (out["units"] == 0) & (out["p_low"] < P_ZERO)
    drop = (out["p_low"] < P_THRESHOLD) & (gap <= -MIN_GAP)
    spike = (out["p_high"] < P_THRESHOLD) & (gap >= MIN_GAP)
    out["flag"] = np.select([zero, drop, spike], ["possible_stockout", "demand_drop", "demand_spike"], "")
    out["level"] = "product"
    return _finish(out)


def score_categories(df: pd.DataFrame, mu_col: str, alpha: pd.Series) -> pd.DataFrame:
    """Aggregate to branch x category x day; larger counts make smaller shifts detectable."""
    frame = df.assign(var=df[mu_col] + alpha * df[mu_col] ** 2,
                      margin_units=df["unit_margin"] * df[mu_col])
    agg = frame.groupby(["Branch", "Category", "Date"], observed=True).agg(
        units=("units", "sum"), expected=(mu_col, "sum"), var=("var", "sum"),
        margin_units=("margin_units", "sum")).reset_index()
    agg_alpha = ((agg["var"] - agg["expected"]) / agg["expected"] ** 2).clip(lower=1e-3)
    agg["unit_margin"] = agg["margin_units"] / agg["expected"]
    agg["p_low"] = nb_cdf(agg["units"], agg["expected"], agg_alpha)
    agg["p_high"] = nb_sf(agg["units"], agg["expected"], agg_alpha)
    agg["flag"] = np.select([agg["p_low"] < P_THRESHOLD, agg["p_high"] < P_THRESHOLD],
                            ["demand_drop", "demand_spike"], "")
    agg["Product"] = "(all " + agg["Category"].astype(str) + ")"
    agg["level"] = "category"
    return _finish(agg.drop(columns=["var", "margin_units"]))


def score_sustained(df: pd.DataFrame, mu_col: str, alpha: pd.Series, window: int = 7) -> pd.DataFrame:
    """Product-level rolling window: catches drops that single low-volume days cannot show.

    Only the first day of each run of flagged windows is kept, so one episode is one alert.
    """
    frame = df.assign(var=df[mu_col] + alpha * df[mu_col] ** 2,
                      margin_units=df["unit_margin"] * df[mu_col]).sort_values(["Branch", "Product", "Date"])
    g = frame.groupby(["Branch", "Product"], observed=True)
    roll = lambda col: g[col].transform(lambda s: s.rolling(window, min_periods=window).sum())  # noqa: E731
    out = frame[["Branch", "Product", "Category", "shelf_class", "Date"]].copy()
    out["units"], out["expected"], var = roll("units"), roll(mu_col), roll("var")
    out["unit_margin"] = roll("margin_units") / out["expected"]
    out = out.dropna(subset=["expected"])
    var = var.loc[out.index]
    w_alpha = ((var - out["expected"]) / out["expected"] ** 2).clip(lower=1e-3)
    out["p_low"] = nb_cdf(out["units"], out["expected"], w_alpha)
    out["p_high"] = nb_sf(out["units"], out["expected"], w_alpha)
    gap = out["units"] - out["expected"]
    flag = np.select([(out["p_low"] < P_THRESHOLD) & (gap <= -MIN_GAP),
                      (out["p_high"] < P_THRESHOLD) & (gap >= MIN_GAP)],
                     ["sustained_drop", "sustained_rise"], "")
    previous = pd.Series(flag, index=out.index).groupby([out["Branch"], out["Product"]], observed=True).shift(1)
    out["flag"] = np.where(flag != previous.fillna(""), flag, "")
    out["Product"] = out["Product"].astype(str) + f" ({window}-day)"
    out["level"] = "product_window"
    return _finish(out)


def _finish(out: pd.DataFrame) -> pd.DataFrame:
    gap = out["units"] - out["expected"]
    out["profit_impact"] = gap * out["unit_margin"]   # negative = profit lost vs expectation
    out["p_value"] = np.minimum(out["p_low"], out["p_high"])
    out["severity"] = -np.log10(np.maximum(out["p_value"], 1e-12)) * out["profit_impact"].abs()
    return out


def explain(row) -> str:
    verb = {"possible_stockout": "sold nothing", "demand_drop": "sold far less",
            "demand_spike": "sold far more", "sustained_drop": "has been selling far less",
            "sustained_rise": "has been selling far more"}[row["flag"]]
    impact = row["profit_impact"]
    money = f"est. lost profit ${-impact:.0f}" if impact < 0 else f"extra profit ${impact:.0f}; check tomorrow's stock"
    return (f"{row['Branch']} | {row['Product']} {verb}: {row['units']:.0f} vs expected "
            f"{row['expected']:.1f} (p={row['p_value']:.3f}) | {money}")


def alerts(*scored: pd.DataFrame) -> pd.DataFrame:
    flagged = pd.concat([s[s["flag"] != ""] for s in scored], ignore_index=True)
    flagged["message"] = flagged.apply(explain, axis=1)
    flagged["rank_in_branch_day"] = flagged.groupby(["Branch", "Date"])["severity"].rank(
        ascending=False, method="first").astype(int)
    return flagged.sort_values(["Date", "Branch", "rank_in_branch_day"], ignore_index=True)


def inject_sustained(df: pd.DataFrame, n_episodes: int = 40, length: int = 7, seed: int = 1) -> pd.DataFrame:
    """Plant multi-day drops (sales cut to 30%) in random series to measure window detection."""
    rng = np.random.default_rng(seed)
    out = df.sort_values(["Branch", "Product", "Date"]).copy()
    out["injected"] = ""
    dates = sorted(out["Date"].unique())
    series = out[["Branch", "Product"]].drop_duplicates().to_numpy()
    for i in rng.choice(len(series), n_episodes, replace=False):
        start = dates[rng.integers(0, len(dates) - length)]
        mask = ((out["Branch"] == series[i][0]) & (out["Product"] == series[i][1])
                & (out["Date"] >= start) & (out["Date"] < start + pd.Timedelta(days=length)))
        out.loc[mask, "units"] = np.floor(out.loc[mask, "units"] * 0.3)
        out.loc[mask, "injected"] = f"episode_{i}"
    return out


def inject(df: pd.DataFrame, mu_col: str, share: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Plant synthetic anomalies (zeroed days, halved+ drops, spikes) to measure detection."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["injected"] = ""
    idx = rng.choice(out.index, int(len(out) * share), replace=False)
    kinds = rng.choice(["zero", "drop", "spike"], len(idx))
    mu = out.loc[idx, mu_col].to_numpy()
    new = np.where(kinds == "zero", 0,
                   np.where(kinds == "drop", np.floor(out.loc[idx, "units"] * 0.3),
                            out.loc[idx, "units"] + np.ceil(2 * mu + 3)))
    out.loc[idx, "units"] = new
    out.loc[idx, "injected"] = kinds
    return out
