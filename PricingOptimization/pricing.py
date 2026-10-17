"""Price optimisation per branch x product.

Costs are fixed; we pick a new list price for each product at each branch.

1. Baseline: expected units/day at today's price (recent average, or the Merchandise model's forecast).
2. Price response: units = baseline * (new_price / old_price) ** elasticity.
   The data has no price changes, so elasticities are assumptions per category,
   with low / mid / high scenarios.
3. Search: try prices within +/-10% in $0.05 steps, keep the one with the best
   expected profit that does not lose money even in the most price-sensitive scenario.

Run from the repo root:  py PricingOptimization/pricing.py [--baseline merchandise]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
# Forecast outputs of the Merchandise model (folder was called aya/ before the rename).
MERCH_OUT = next((p for p in (HERE.parent / "Merchandise" / "outputs", HERE.parent / "aya" / "outputs") if p.exists()),
                 HERE.parent / "Merchandise" / "outputs")

TRAIN_END = pd.Timestamp("2026-05-15")       # whole days; boundary days go to the later set
VALIDATION_END = pd.Timestamp("2026-06-08")
LAST_DAY = pd.Timestamp("2026-06-30")

MAX_CHANGE = 0.10   # max price move per product
STEP = 0.05         # price grid in dollars
WINDOWS = (14, 28, 56)

# Assumed own-price elasticity per category (units % change per 1% price change).
# Net of customers switching to another item in the same shop.
ELASTICITY = pd.DataFrame(
    [
        ("Hot Coffee",   -0.4, -0.8, -1.4),
        ("Cold Coffee",  -0.6, -1.0, -1.6),
        ("Matcha & Tea", -0.6, -1.1, -1.8),
        ("Cold Drinks",  -0.8, -1.3, -2.0),
        ("Bakery",       -0.7, -1.2, -1.9),
        ("Desserts",     -0.8, -1.4, -2.2),
        ("Breakfast",    -0.6, -1.1, -1.8),
        ("Sandwiches",   -0.6, -1.1, -1.8),
        ("Meals",        -0.5, -1.0, -1.6),
    ],
    columns=["Category", "low", "mid", "high"],
).set_index("Category")
SCENARIOS = {"low": 0.25, "mid": 0.50, "high": 0.25}


# ---------- data ----------

def load_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily units per branch x product, zero-filled, from the three split files."""
    lines = pd.concat([pd.read_csv(DATA / f"{s}.csv", parse_dates=["Date"])
                       for s in ("train", "validation", "test")])
    keys = ["Branch", "Product"]
    info = lines.groupby(keys).agg(
        Category=("Category", "first"),
        price=("Unit Price (USD)", "first"),
        cost=("Unit Cost (USD)", "first"),
        n_prices=("Unit Price (USD)", "nunique"),
    )
    if (info["n_prices"] > 1).any():
        raise ValueError("Expected one list price per branch x product")
    daily = lines.groupby(keys + ["Date"])["Quantity"].sum()
    grid = pd.MultiIndex.from_product(
        [*info.index.levels, pd.date_range(lines["Date"].min(), lines["Date"].max())],
        names=keys + ["Date"])
    daily = daily.reindex(grid, fill_value=0).rename("units").reset_index()
    # Average discount per series, so unit revenue matches what was really earned.
    disc = lines.groupby(keys)["Discount Rate"].mean().rename("discount")
    return daily, info.drop(columns="n_prices").join(disc).reset_index()


def mean_units(daily: pd.DataFrame, end: pd.Timestamp, days: int) -> pd.Series:
    window = daily[(daily["Date"] > end - pd.Timedelta(days=days)) & (daily["Date"] <= end)]
    return window.groupby(["Branch", "Product"])["units"].mean()


def period_mean(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    period = daily[(daily["Date"] >= start) & (daily["Date"] <= end)]
    return period.groupby(["Branch", "Product"])["units"].mean()


def error(pred: pd.Series, actual: pd.Series) -> float:
    """Total absolute error / total units (on per-series daily means)."""
    return float((pred - actual).abs().sum() / actual.sum())


def merchandise_baseline() -> pd.Series | None:
    """Mean expected units/day from the Merchandise forecast files, or None if unusable."""
    for name in ("forecast_next_14_days.csv", "forecast_2026-07-01.csv"):
        path = MERCH_OUT / name
        if not path.exists():
            continue
        try:
            f = pd.read_csv(path)
            s = f.groupby(["Branch", "Product"])["expected_units"].mean()
        except (KeyError, ValueError, pd.errors.ParserError):
            continue
        if len(s) == 135 and s.notna().all():
            return s
    return None


# ---------- pricing ----------

def units_at(baseline, old_price, new_price, elasticity):
    return baseline * (new_price / old_price) ** elasticity


def best_price(price: float, cost: float, discount: float, elasticities: dict[str, float],
               max_change: float = MAX_CHANGE, step: float = STEP) -> float:
    """Best list price for one product (baseline of 1 unit/day; the choice does not depend on volume)."""
    grid = np.arange(np.ceil(price * (1 - max_change) / step), np.floor(price * (1 + max_change) / step) + 1) * step
    grid = np.unique(np.append(np.round(grid, 2), price))
    margin = grid * (1 - discount) - cost
    profit = {k: margin * units_at(1.0, price, grid, e) for k, e in elasticities.items()}
    current = price * (1 - discount) - cost
    expected = sum(SCENARIOS[k] * profit[k] for k in profit)
    # Must not lose money even if customers are as sensitive as the "high" scenario.
    ok = (profit["high"] >= current - 1e-12) & (margin > 0)
    expected = np.where(ok, expected, -np.inf)
    best = np.flatnonzero(expected >= expected.max() - 1e-12)
    return float(grid[best[np.argmin(np.abs(grid[best] - price))]])  # tie -> closest to today's price


def recommend(info: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    rec = info.copy()
    rec["baseline_units_per_day"] = baseline.reindex(pd.MultiIndex.from_frame(rec[["Branch", "Product"]])).to_numpy()
    rec["new_price"] = [
        best_price(r.price, r.cost, r.discount, ELASTICITY.loc[r.Category].to_dict())
        for r in rec.itertuples()]
    rec["change_pct"] = (rec["new_price"] / rec["price"] - 1) * 100
    rec["action"] = np.select([rec["change_pct"] > 0, rec["change_pct"] < 0], ["raise", "cut"], "keep")
    old_margin = rec["price"] * (1 - rec["discount"]) - rec["cost"]
    new_margin = rec["new_price"] * (1 - rec["discount"]) - rec["cost"]
    rec["profit_per_day_now"] = old_margin * rec["baseline_units_per_day"]
    for k in SCENARIOS:
        e = rec["Category"].map(ELASTICITY[k])
        units = units_at(rec["baseline_units_per_day"], rec["price"], rec["new_price"], e)
        rec[f"units_per_day_{k}"] = units
        rec[f"profit_change_per_day_{k}"] = new_margin * units - rec["profit_per_day_now"]
    # Sensitivity at which this change stops paying off (ignores products with no change).
    r0, r1 = rec["price"] * (1 - rec["discount"]), rec["new_price"] * (1 - rec["discount"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rec["break_even_elasticity"] = np.where(
            rec["action"] == "keep", np.nan,
            np.log((r0 - rec["cost"]) / (r1 - rec["cost"])) / np.log(rec["new_price"] / rec["price"]))
    return rec.round(4)


def totals(rec: pd.DataFrame) -> dict:
    out = {}
    for branch, g in [("ALL", rec), *rec.groupby("Branch")]:
        now = g["profit_per_day_now"].sum()
        out[branch] = {"profit_per_day_now": round(now, 2)} | {
            f"change_{k}": f"{g[f'profit_change_per_day_{k}'].sum():+.2f} $/day "
                           f"({g[f'profit_change_per_day_{k}'].sum() / now:+.1%})"
            for k in SCENARIOS}
    return out


def simulate_test(daily: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """Apply prices chosen from data up to Jun 8 to the real test-period sales (Jun 9-30)."""
    actual = period_mean(daily, VALIDATION_END + pd.Timedelta(days=1), LAST_DAY)
    sim = rec.drop(columns="baseline_units_per_day")
    sim["baseline_units_per_day"] = actual.reindex(pd.MultiIndex.from_frame(sim[["Branch", "Product"]])).to_numpy()
    old_margin = sim["price"] * (1 - sim["discount"]) - sim["cost"]
    new_margin = sim["new_price"] * (1 - sim["discount"]) - sim["cost"]
    now = (old_margin * sim["baseline_units_per_day"]).sum()
    out = {"actual_profit_per_day": round(now, 2)}
    for k in SCENARIOS:
        units = units_at(sim["baseline_units_per_day"], sim["price"], sim["new_price"],
                         sim["Category"].map(ELASTICITY[k]))
        out[f"change_{k}"] = f"{(new_margin * units).sum() / now - 1:+.1%}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["history", "merchandise"], default="history")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    daily, info = load_daily()

    # Validation: which averaging window best predicts the next period's daily sales?
    val_actual = period_mean(daily, TRAIN_END + pd.Timedelta(days=1), VALIDATION_END)
    val_error = {w: error(mean_units(daily, TRAIN_END, w), val_actual) for w in WINDOWS}
    window = min(val_error, key=val_error.get)
    test_actual = period_mean(daily, VALIDATION_END + pd.Timedelta(days=1), LAST_DAY)
    test_error = error(mean_units(daily, VALIDATION_END, window), test_actual)

    # Test: prices chosen from data up to Jun 8, scored on Jun 9-30 sales.
    test_rec = recommend(info, mean_units(daily, VALIDATION_END, window))

    # Final recommendation for July, using all data (or the Merchandise forecast).
    source = f"history (last {window} days)"
    baseline = mean_units(daily, LAST_DAY, window)
    if args.baseline == "merchandise":
        merch = merchandise_baseline()
        if merch is None:
            source += " - Merchandise forecast files missing or incomplete, used history"
        else:
            baseline, source = merch, "Merchandise forecast (mean expected units)"
    rec = recommend(info, baseline)
    rec.sort_values(["Branch", "Category", "Product"]).to_csv(OUT / "price_recommendations.csv", index=False)

    summary = {
        "baseline": source,
        "baseline_error_validation_by_window": {w: round(v, 3) for w, v in val_error.items()},
        "baseline_error_test": round(test_error, 3),
        "elasticity_assumptions": ELASTICITY.to_dict("index"),
        "scenario_weights": SCENARIOS,
        "actions": rec.groupby("Branch")["action"].value_counts().unstack(fill_value=0).to_dict("index"),
        "expected_profit_change": totals(rec),
        "test_simulation": simulate_test(daily, test_rec),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
