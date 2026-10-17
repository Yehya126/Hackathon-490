"""End-to-end run: tune on validation, evaluate on test, forecast expected units sold for the next 14 days,
write outputs + report.

    py Merchandise/run_pipeline.py
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
from horizon import MAX_HORIZON, HorizonModel, future_frame, stacked  # noqa: E402
from panel import KEYS, TRAIN_END, VALIDATION_END, build_daily_panel, load_transactions, split_by_date  # noqa: E402
from report import write_report  # noqa: E402

MODE = "offset"  # chosen on validation in compare_models.py
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def with_future(panel: pd.DataFrame, days: int) -> pd.DataFrame:
    """Append future days per series (units unknown) so forecast rows can be built for them."""
    last = panel[panel["Date"] == panel["Date"].max()]
    extra = []
    for d in range(1, days + 1):
        day = last.copy()
        day["Date"] += pd.Timedelta(days=d)
        for col in ["units", "revenue", "gross_profit", "discount", "last_sale_hour"]:
            day[col] = float("nan")
        extra.append(day)
    return pd.concat([panel] + extra, ignore_index=True)


def rounded(frame: pd.DataFrame, digits: int) -> pd.DataFrame:
    numeric = frame.select_dtypes("number").columns
    return frame.assign(**{c: frame[c].round(digits) for c in numeric})


def branch_alpha(df, pred_col, alpha):
    return df["Branch"].astype(str).map(alpha)


def category_totals(frame: pd.DataFrame, by_date: bool = False) -> pd.DataFrame:
    """Expected units per branch x category, plus the branch total (Category = "ALL")."""
    f = frame.assign(Category=frame["Category"].astype(str), Branch=frame["Branch"].astype(str))
    keys = ["Branch"] + (["Date"] if by_date else [])
    cat = f.groupby(keys + ["Category"])["forecast"].sum().reset_index()
    total = f.groupby(keys)["forecast"].sum().reset_index().assign(Category="ALL")
    return pd.concat([cat, total], ignore_index=True).sort_values(
        keys + ["forecast"], ascending=[True] * len(keys) + [False], ignore_index=True)


def horizon_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    """Product-level WAPE, branch daily-total WAPE and bias for each number of days ahead."""
    out = []
    for h, g in frame.groupby("horizon"):
        b = g.groupby(["Branch", "Date"], observed=True)[["units", "forecast"]].sum()
        out.append({"days_ahead": h,
                    "product_wape": (g["forecast"] - g["units"]).abs().sum() / g["units"].sum(),
                    "branch_day_wape": (b["forecast"] - b["units"]).abs().sum() / b["units"].sum(),
                    "bias": g["forecast"].sum() / g["units"].sum() - 1})
    return pd.DataFrame(out).set_index("days_ahead")


data = load_transactions()
panel = build_daily_panel(data)
last_date = panel["Date"].max()
tomorrow = last_date + pd.Timedelta(days=1)
df = add_features(with_future(panel, MAX_HORIZON))
known = df[df["Date"] <= last_date]
train, valid, test = split_by_date(known)
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

# 3) Expected units 1-14 days ahead: one model across horizons (chosen on validation in compare_horizons.py).
#    Rounds by early stopping on validation targets, refit on train + validation, scored on test targets,
#    then refit on every known day to forecast the next 14 days.
rows = stacked(known)
h_train = rows[rows["Date"] <= TRAIN_END]
h_valid = rows[(rows["Date"] > TRAIN_END) & (rows["Date"] <= VALIDATION_END)]
h_test = rows[rows["Date"] > VALIDATION_END].dropna(subset=["roll_mean_28"]).copy()
h_rounds = HorizonModel().fit(h_train, h_valid).rounds
h_test["forecast"] = HorizonModel().fit(pd.concat([h_train, h_valid]), rounds=h_rounds).predict(h_test)
by_horizon = horizon_accuracy(h_test)
# Test-period view of the model behind the forward forecasts: 1 day ahead from the multi-day model.
day1 = h_test[h_test["horizon"] == 1]
day1 = pd.Series(day1["forecast"].to_numpy(), index=pd.MultiIndex.from_frame(day1[KEYS + ["Date"]].astype(str)))
test_view = test.copy()
test_view["forecast"] = day1.reindex(pd.MultiIndex.from_frame(test[KEYS + ["Date"]].astype(str))).to_numpy()
by_horizon_branch = pd.concat({b: horizon_accuracy(g) for b, g in h_test.groupby(h_test["Branch"].astype(str))},
                              names=["Branch"])

fortnight = future_frame(df, last_date)
fortnight["forecast"] = HorizonModel().fit(rows, rounds=h_rounds).predict(fortnight)
future = fortnight[fortnight["horizon"] == 1].copy()
future["vs_recent_same_weekday"] = future["forecast"] - future["same_weekday_mean_4"]
totals_tomorrow = category_totals(future)
totals_fortnight = category_totals(fortnight, by_date=True)

# Outputs
cols = KEYS + ["Category", "Date"]
rounded(test_view[cols + ["units", "forecast", "naive"]].rename(columns={
    "units": "actual_units", "forecast": "expected_units", "naive": "same_weekday_last_week"}), 2).to_csv(
    OUT / "test_forecasts.csv", index=False)
rounded(future[cols + ["forecast", "same_weekday_mean_4", "vs_recent_same_weekday"]].rename(columns={
    "forecast": "expected_units", "same_weekday_mean_4": "usual_last_4_same_weekdays",
    "vs_recent_same_weekday": "change_vs_usual"}), 2).to_csv(OUT / f"forecast_{tomorrow:%Y-%m-%d}.csv", index=False)
rounded(totals_tomorrow.rename(columns={"forecast": "expected_units"}), 1).to_csv(
    OUT / f"category_forecast_{tomorrow:%Y-%m-%d}.csv", index=False)
rounded(fortnight[cols + ["horizon", "forecast"]].rename(columns={
    "horizon": "days_ahead", "forecast": "expected_units"}).sort_values(KEYS + ["Date"]), 2).to_csv(
    OUT / "forecast_next_14_days.csv", index=False)
rounded(totals_fortnight.rename(columns={"forecast": "expected_units"}), 1).to_csv(
    OUT / "category_forecast_next_14_days.csv", index=False)
rounded(pd.concat({"ALL": by_horizon}, names=["Branch"]).pipe(lambda t: pd.concat([t, by_horizon_branch])), 4).to_csv(
    OUT / "test_accuracy_by_days_ahead.csv")
rounded(test_alerts[["Date", "Branch", "level", "Product", "flag", "units", "expected", "p_value",
                     "profit_impact", "severity", "rank_in_branch_day", "message"]], 3).to_csv(OUT / "alerts_test.csv", index=False)

summary = {
    "test_period": [str(test["Date"].min().date()), str(test["Date"].max().date())],
    "forecast_for": [str(tomorrow.date()), str((last_date + pd.Timedelta(days=MAX_HORIZON)).date())],
    "model": f"LightGBM ({MODE} mode) + negative binomial",
    "boosting_rounds": rounds,
    "nb_alpha": alpha.round(4).to_dict(),
    "accuracy": {k: v.to_dict(orient="index") for k, v in metrics.items()},
    "branch_daily_total_wape": branch_day_wape.round(3).to_dict(),
    "alert_spread_check_p10_p90_coverage": alert_calibration.round(3).to_dict(),
    "multi_day_model_rounds": h_rounds,
    "test_accuracy_by_days_ahead": by_horizon.round(4).to_dict(orient="index"),
    "alerts": {"count": len(test_alerts), "by_flag": test_alerts["flag"].value_counts().to_dict()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

for name, table in metrics.items():
    print(f"\n{name} (test)\n{table.to_string()}")
print("\nError on branch daily totals:", branch_day_wape.round(3).to_dict())
print("\n1-14 day model on test, by days ahead (%):\n", (by_horizon * 100).round(1).to_string())
print(f"\nAlerts on test: {len(test_alerts)}", test_alerts["flag"].value_counts().to_dict())

write_report(OUT / "report.html", test=test_view, future=future, alerts=test_alerts, metrics=metrics,
             tomorrow=tomorrow, category_plan=totals_tomorrow, fortnight=fortnight,
             by_horizon=by_horizon_branch)
print("\nWrote", ", ".join(p.name for p in sorted(OUT.glob("*.*"))))
