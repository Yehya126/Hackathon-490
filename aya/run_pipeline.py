"""End-to-end run: tune on validation, evaluate on test, forecast tomorrow's expected units sold, write outputs + report.

    py aya/run_pipeline.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from anomaly import alerts, score_categories, score_products, score_sustained  # noqa: E402
from features import add_features  # noqa: E402
from forecast import (accuracy, fit_dispersion, fit_lgbm, nb_quantile, seasonal_naive,  # noqa: E402
                      weekday_average)
from panel import KEYS, build_daily_panel, load_transactions, split_by_date  # noqa: E402
from report import write_report  # noqa: E402

MODE = "offset"  # chosen on validation in compare_models.py
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


def category_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Expected units per branch x category, plus the branch total (Category = "ALL")."""
    f = frame.assign(Category=frame["Category"].astype(str), Branch=frame["Branch"].astype(str))
    cat = f.groupby(["Branch", "Category"])["forecast"].sum().reset_index()
    total = f.groupby("Branch")["forecast"].sum().reset_index().assign(Category="ALL")
    return pd.concat([cat, total], ignore_index=True).sort_values(
        ["Branch", "forecast"], ascending=[True, False], ignore_index=True)


data = load_transactions()
panel = build_daily_panel(data)
tomorrow = panel["Date"].max() + pd.Timedelta(days=1)
df = add_features(with_tomorrow(panel))
future = df[df["Date"] == tomorrow].copy()
train, valid, test = split_by_date(df[df["Date"] < tomorrow])
valid, test = valid.copy(), test.copy()

# 1) Tune on validation: rounds via early stopping. The NB dispersion (spread around the expected value)
#    is used only by the alerts, to judge whether a day's sales are unusually far from expected.
model, rounds = fit_lgbm(train, valid, mode=MODE)
valid["forecast"] = model.predict(valid)
alpha = fit_dispersion(valid["units"], valid["forecast"], valid["Branch"])
print(f"Rounds {rounds}; NB alpha per branch: {alpha.round(3).to_dict()}")

# 2) Refit on train + validation, evaluate once on test.
final, _ = fit_lgbm(pd.concat([train, valid]), rounds=rounds, mode=MODE)
test["forecast"] = final.predict(test)
previous, _ = fit_lgbm(pd.concat([train, valid]), rounds=fit_lgbm(train, valid, mode="raw")[1], mode="raw")
test["previous_model"] = previous.predict(test)  # first version, for comparison only
test["naive"] = seasonal_naive(test)
test["weekday_avg"] = weekday_average(test)
a_test = branch_alpha(test, "forecast", alpha)
metrics = {name: accuracy(test, col) for name, col in
           [("Same weekday last week", "naive"), ("Weekday average", "weekday_avg"),
            ("Previous model (raw LightGBM)", "previous_model"), ("LightGBM + NB", "forecast")]}
daily = test.groupby(["Branch", "Date"], observed=True)[["units", "forecast"]].sum()
branch_day_wape = ((daily["forecast"] - daily["units"]).abs().groupby(level="Branch").sum()
                   / daily["units"].groupby(level="Branch").sum())
# Check of the spread the alerts rely on: share of days inside the 10th-90th percentile range.
inside = ((test["units"] >= nb_quantile(test["forecast"], a_test, 0.10))
          & (test["units"] <= nb_quantile(test["forecast"], a_test, 0.90)))
alert_calibration = inside.groupby(test["Branch"], observed=True).mean()

prod = score_products(test, "forecast", a_test)
cat = score_categories(test, "forecast", a_test)
sus = score_sustained(test, "forecast", a_test)
test_alerts = alerts(prod, cat, sus)

# 3) Production model on all history -> tomorrow's expected units sold.
production, _ = fit_lgbm(df[df["Date"] < tomorrow], rounds=rounds, mode=MODE)
future["forecast"] = production.predict(future)
future["vs_recent_same_weekday"] = future["forecast"] - future["same_weekday_mean_4"]
totals_tomorrow = category_totals(future)

# Outputs
cols = KEYS + ["Category", "Date"]
rounded(test[cols + ["units", "forecast", "naive"]].rename(columns={
    "units": "actual_units", "forecast": "expected_units", "naive": "same_weekday_last_week"}), 2).to_csv(
    OUT / "test_forecasts.csv", index=False)
rounded(future[cols + ["forecast", "same_weekday_mean_4", "vs_recent_same_weekday"]].rename(columns={
    "forecast": "expected_units", "same_weekday_mean_4": "usual_last_4_same_weekdays",
    "vs_recent_same_weekday": "change_vs_usual"}), 2).to_csv(OUT / f"forecast_{tomorrow:%Y-%m-%d}.csv", index=False)
rounded(totals_tomorrow.rename(columns={"forecast": "expected_units"}), 1).to_csv(
    OUT / f"category_forecast_{tomorrow:%Y-%m-%d}.csv", index=False)
rounded(test_alerts[["Date", "Branch", "level", "Product", "flag", "units", "expected", "p_value",
                     "profit_impact", "severity", "rank_in_branch_day", "message"]], 3).to_csv(OUT / "alerts_test.csv", index=False)

summary = {
    "test_period": [str(test["Date"].min().date()), str(test["Date"].max().date())],
    "forecast_for": str(tomorrow.date()),
    "model": f"LightGBM ({MODE} mode) + negative binomial",
    "boosting_rounds": rounds,
    "nb_alpha": alpha.round(4).to_dict(),
    "accuracy": {k: v.to_dict(orient="index") for k, v in metrics.items()},
    "branch_daily_total_wape": branch_day_wape.round(3).to_dict(),
    "alert_spread_check_p10_p90_coverage": alert_calibration.round(3).to_dict(),
    "alerts": {"count": len(test_alerts), "by_flag": test_alerts["flag"].value_counts().to_dict()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

for name, table in metrics.items():
    print(f"\n{name} (test)\n{table.to_string()}")
print("\nError on branch daily totals:", branch_day_wape.round(3).to_dict())
print(f"\nAlerts on test: {len(test_alerts)}", test_alerts["flag"].value_counts().to_dict())

write_report(OUT / "report.html", test=test, future=future, alerts=test_alerts, metrics=metrics,
             tomorrow=tomorrow, category_plan=totals_tomorrow)
print("\nWrote", ", ".join(p.name for p in sorted(OUT.glob("*.*"))))
