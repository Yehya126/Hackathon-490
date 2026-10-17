"""End-to-end run: tune on validation, evaluate once on test, forecast tomorrow, write outputs + report.

    py aya/run_pipeline.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from anomaly import alerts, score_categories, score_products, score_sustained  # noqa: E402
from features import FEATURES, add_features  # noqa: E402
from forecast import (accuracy, fit_dispersion, fit_lgbm, nb_quantile, seasonal_naive,  # noqa: E402
                      weekday_average)
from inventory import critical_ratio, recommend, simulate, summarize  # noqa: E402
from panel import KEYS, build_daily_panel, load_transactions, split_by_date  # noqa: E402
from report import write_report  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def with_tomorrow(panel: pd.DataFrame) -> pd.DataFrame:
    """Append one future day per series (units unknown) so features can be built for it."""
    last = panel[panel["Date"] == panel["Date"].max()].copy()
    last["Date"] += pd.Timedelta(days=1)
    for col in ["units", "revenue", "gross_profit", "discount", "last_sale_hour"]:
        last[col] = float("nan")
    return pd.concat([panel, last], ignore_index=True)


def rounded(frame: pd.DataFrame, digits: int) -> pd.DataFrame:
    numeric = frame.select_dtypes("number").columns
    return frame.assign(**{c: frame[c].round(digits) for c in numeric})


def branch_alpha(df, pred_col, alpha):
    return df["Branch"].astype(str).map(alpha)


panel = build_daily_panel(load_transactions())
tomorrow = panel["Date"].max() + pd.Timedelta(days=1)
df = add_features(with_tomorrow(panel))
future = df[df["Date"] == tomorrow].copy()
train, valid, test = split_by_date(df[df["Date"] < tomorrow])
valid, test = valid.copy(), test.copy()

# 1) Tune on validation: rounds via early stopping, NB dispersion from out-of-sample residuals.
model, rounds = fit_lgbm(train, valid)
valid["forecast"] = model.predict(valid[FEATURES], num_iteration=rounds)
alpha = fit_dispersion(valid["units"], valid["forecast"], valid["Branch"])
print(f"Rounds {rounds}; NB alpha per branch: {alpha.round(3).to_dict()}")

# 2) Refit on train + validation, evaluate once on test.
final, _ = fit_lgbm(pd.concat([train, valid]), rounds=rounds)
test["forecast"] = final.predict(test[FEATURES])
test["naive"] = seasonal_naive(test)
test["weekday_avg"] = weekday_average(test)
a_test = branch_alpha(test, "forecast", alpha)
test["p10"] = nb_quantile(test["forecast"], a_test, 0.10)
test["p90"] = nb_quantile(test["forecast"], a_test, 0.90)
test["critical_ratio"] = critical_ratio(test)
test["prep_recommended"] = recommend(test, "forecast", a_test)

metrics = {name: accuracy(test, col) for name, col in
           [("Same weekday last week", "naive"), ("Weekday average", "weekday_avg"), ("LightGBM + NB", "forecast")]}
coverage = ((test["units"] >= test["p10"]) & (test["units"] <= test["p90"])).groupby(test["Branch"], observed=True).mean()

policies = {"Same weekday last week": "naive", "Forecast mean": "forecast", "Recommended (NB quantile)": "prep_recommended"}
sims = {name: simulate(test, col) for name, col in policies.items()}
inventory = {name: summarize(s, ["Branch", "shelf_class"]) for name, s in sims.items()}

prod = score_products(test, "forecast", a_test)
cat = score_categories(test, "forecast", a_test)
sus = score_sustained(test, "forecast", a_test)
test_alerts = alerts(prod, cat, sus)

# 3) Production model on all history -> tomorrow's prep plan.
production, _ = fit_lgbm(df[df["Date"] < tomorrow], rounds=rounds)
future["forecast"] = production.predict(future[FEATURES])
a_future = branch_alpha(future, "forecast", alpha)
future["p10"] = nb_quantile(future["forecast"], a_future, 0.10)
future["p90"] = nb_quantile(future["forecast"], a_future, 0.90)
future["critical_ratio"] = critical_ratio(future)
future["prep_recommended"] = recommend(future, "forecast", a_future)
future["vs_recent_same_weekday"] = future["forecast"] - future["same_weekday_mean_4"]

# Outputs
cols = KEYS + ["Category", "shelf_class", "Date"]
rounded(test[cols + ["units", "forecast", "p10", "p90", "naive", "prep_recommended"]], 2).to_csv(
    OUT / "test_forecasts.csv", index=False)
rounded(future[cols + ["forecast", "p10", "p90", "critical_ratio", "prep_recommended",
                       "same_weekday_mean_4", "vs_recent_same_weekday"]], 2).to_csv(OUT / f"prep_plan_{tomorrow:%Y-%m-%d}.csv", index=False)
rounded(test_alerts[["Date", "Branch", "level", "Product", "flag", "units", "expected", "p_value",
                     "profit_impact", "severity", "rank_in_branch_day", "message"]], 3).to_csv(OUT / "alerts_test.csv", index=False)

totals = {name: s[["profit", "waste_cost", "missed_profit", "sold", "missed"]].sum().round(1).to_dict()
          for name, s in sims.items()}
summary = {
    "test_period": [str(test["Date"].min().date()), str(test["Date"].max().date())],
    "forecast_for": str(tomorrow.date()),
    "boosting_rounds": rounds,
    "nb_alpha": alpha.round(4).to_dict(),
    "accuracy": {k: v.to_dict(orient="index") for k, v in metrics.items()},
    "p10_p90_coverage": coverage.round(3).to_dict(),
    "inventory_totals": totals,
    "alerts": {"count": len(test_alerts), "by_flag": test_alerts["flag"].value_counts().to_dict()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

for name, table in metrics.items():
    print(f"\n{name} (test)\n{table.to_string()}")
print("\nP10-P90 coverage:", coverage.round(3).to_dict())
print("\nInventory totals (test):")
print(pd.DataFrame(totals).T.to_string())
print(f"\nAlerts on test: {len(test_alerts)}", test_alerts["flag"].value_counts().to_dict())

write_report(OUT / "report.html", test=test, future=future, alerts=test_alerts, metrics=metrics,
             coverage=coverage, sims=sims, tomorrow=tomorrow)
print("\nWrote", ", ".join(p.name for p in sorted(OUT.glob("*.*"))))
