"""End-to-end run: tune on validation, evaluate once on test, forecast tomorrow, write outputs + report.

    py aya/run_pipeline.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from batches import attach, morning_plan, two_batch  # noqa: E402
from anomaly import alerts, score_categories, score_products, score_sustained  # noqa: E402
from features import add_features  # noqa: E402
from forecast import (accuracy, fit_dispersion, fit_lgbm, nb_quantile, seasonal_naive,  # noqa: E402
                      weekday_average)
from inventory import critical_ratio, recommend, simulate, summarize  # noqa: E402
from panel import KEYS, build_daily_panel, load_transactions, split_by_date  # noqa: E402
from report import write_report  # noqa: E402

MODE = "offset"      # chosen on validation in compare_models.py
CUTOFF_HOUR = 15     # afternoon top-up time, chosen on validation in choose_batches.py
MORNING_LEVEL = 0.9  # morning batch = this quantile of morning demand (choose_batches.py)
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


def category_plan(frame: pd.DataFrame, alpha: pd.Series) -> pd.DataFrame:
    """Branch x category (and branch total) expected units with a P10-P90 range, for staffing and ingredients."""
    a = branch_alpha(frame, "forecast", alpha)
    f = frame.assign(var=frame["forecast"] + a * frame["forecast"] ** 2, Category=frame["Category"].astype(str),
                     Branch=frame["Branch"].astype(str))
    cat = f.groupby(["Branch", "Category"])[["forecast", "var"]].sum().reset_index()
    total = f.groupby("Branch")[["forecast", "var"]].sum().reset_index().assign(Category="ALL")
    plan = pd.concat([cat, total], ignore_index=True)
    plan_alpha = ((plan["var"] - plan["forecast"]) / plan["forecast"] ** 2).clip(lower=1e-3)
    plan["p10"] = nb_quantile(plan["forecast"], plan_alpha, 0.10)
    plan["p90"] = nb_quantile(plan["forecast"], plan_alpha, 0.90)
    return plan.drop(columns="var").sort_values(["Branch", "forecast"], ascending=[True, False], ignore_index=True)


data = load_transactions()
panel = build_daily_panel(data)
tomorrow = panel["Date"].max() + pd.Timedelta(days=1)
df = add_features(with_tomorrow(panel))
future = df[df["Date"] == tomorrow].copy()
train, valid, test = split_by_date(df[df["Date"] < tomorrow])
valid, test = valid.copy(), test.copy()

# 1) Tune on validation: rounds via early stopping, NB dispersion from out-of-sample residuals.
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
test["p10"] = nb_quantile(test["forecast"], a_test, 0.10)
test["p90"] = nb_quantile(test["forecast"], a_test, 0.90)
test["critical_ratio"] = critical_ratio(test)
test["prep_recommended"] = recommend(test, "forecast", a_test)

metrics = {name: accuracy(test, col) for name, col in
           [("Same weekday last week", "naive"), ("Weekday average", "weekday_avg"),
            ("Previous model (raw LightGBM)", "previous_model"), ("LightGBM + NB", "forecast")]}
coverage = ((test["units"] >= test["p10"]) & (test["units"] <= test["p90"])).groupby(test["Branch"], observed=True).mean()

policies = {"Same weekday last week": "naive", "Forecast mean": "forecast", "Recommended (NB quantile)": "prep_recommended"}
sims = {name: simulate(test, col) for name, col in policies.items()}
# Two batches: fresh products get a morning batch + 3 pm top-up; other products keep the single recommendation.
is_fresh = test["shelf_class"].astype(str) == "fresh"
fresh_test = attach(test[is_fresh], data, CUTOFF_HOUR)
two = two_batch(fresh_test, "forecast", a_test[is_fresh], MORNING_LEVEL).assign(shelf_class="fresh")
rest = simulate(test[~is_fresh], "prep_recommended")
sims["Recommended + fresh two batches"] = pd.concat([two[rest.columns], rest]).loc[test.index]
inventory = {name: summarize(s, ["Branch", "shelf_class"]) for name, s in sims.items()}

prod = score_products(test, "forecast", a_test)
cat = score_categories(test, "forecast", a_test)
sus = score_sustained(test, "forecast", a_test)
test_alerts = alerts(prod, cat, sus)

# 3) Production model on all history -> tomorrow's prep plan.
production, _ = fit_lgbm(df[df["Date"] < tomorrow], rounds=rounds, mode=MODE)
future["forecast"] = production.predict(future)
a_future = branch_alpha(future, "forecast", alpha)
future["p10"] = nb_quantile(future["forecast"], a_future, 0.10)
future["p90"] = nb_quantile(future["forecast"], a_future, 0.90)
future["critical_ratio"] = critical_ratio(future)
future["prep_recommended"] = recommend(future, "forecast", a_future)
future["vs_recent_same_weekday"] = future["forecast"] - future["same_weekday_mean_4"]
fresh_future = future["shelf_class"].astype(str) == "fresh"
future_fresh = attach(future[fresh_future], data, CUTOFF_HOUR)
future["am_share"] = future_fresh["am_share"]
future["morning_batch"] = morning_plan(future_fresh, "forecast", a_future[fresh_future], MORNING_LEVEL)
plan_tomorrow = category_plan(future, alpha)

# Outputs
cols = KEYS + ["Category", "shelf_class", "Date"]
rounded(test[cols + ["units", "forecast", "p10", "p90", "naive", "prep_recommended"]], 2).to_csv(
    OUT / "test_forecasts.csv", index=False)
rounded(future[cols + ["forecast", "p10", "p90", "critical_ratio", "prep_recommended", "morning_batch",
                       "am_share", "same_weekday_mean_4", "vs_recent_same_weekday"]], 2).to_csv(OUT / f"prep_plan_{tomorrow:%Y-%m-%d}.csv", index=False)
rounded(plan_tomorrow, 1).to_csv(OUT / f"category_plan_{tomorrow:%Y-%m-%d}.csv", index=False)
rounded(test_alerts[["Date", "Branch", "level", "Product", "flag", "units", "expected", "p_value",
                     "profit_impact", "severity", "rank_in_branch_day", "message"]], 3).to_csv(OUT / "alerts_test.csv", index=False)

totals = {name: s[["profit", "waste_cost", "missed_profit", "sold", "missed"]].sum().round(1).to_dict()
          for name, s in sims.items()}
summary = {
    "test_period": [str(test["Date"].min().date()), str(test["Date"].max().date())],
    "forecast_for": str(tomorrow.date()),
    "model": f"LightGBM ({MODE} mode) + negative binomial",
    "boosting_rounds": rounds,
    "nb_alpha": alpha.round(4).to_dict(),
    "accuracy": {k: v.to_dict(orient="index") for k, v in metrics.items()},
    "p10_p90_coverage": coverage.round(3).to_dict(),
    "inventory_totals": totals,
    "two_batch": {"cutoff_hour": CUTOFF_HOUR, "morning_level": MORNING_LEVEL},
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
             coverage=coverage, sims=sims, tomorrow=tomorrow, category_plan=plan_tomorrow,
             cutoff_hour=CUTOFF_HOUR)
print("\nWrote", ", ".join(p.name for p in sorted(OUT.glob("*.*"))))
