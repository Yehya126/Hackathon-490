"""Step 3: compare baselines and LightGBM on validation; check the NB distribution."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import add_features  # noqa: E402
from forecast import (accuracy, fit_dispersion, fit_lgbm, nb_quantile, poisson_vs_nb_loglik,  # noqa: E402
                      seasonal_naive, weekday_average)
from panel import build_daily_panel, load_transactions, split_by_date  # noqa: E402

df = add_features(build_daily_panel(load_transactions()))
train, valid, _ = split_by_date(df)
valid = valid.copy()

model, best = fit_lgbm(train, valid)
print("Best rounds:", best)
valid["naive"] = seasonal_naive(valid)
valid["wd_avg"] = weekday_average(valid)
valid["lgbm"] = model.predict(valid[[c for c in model.feature_name()]], num_iteration=best)

for col in ["naive", "wd_avg", "lgbm"]:
    print(f"\n== {col} (validation) ==\n", accuracy(valid, col).to_string())

# Distribution: fit alpha on train in-sample residuals, check coverage on validation.
train_fit = train.dropna(subset=["lag_28"]).copy()
train_fit["lgbm"] = model.predict(train_fit[model.feature_name()], num_iteration=best)
alpha = fit_dispersion(train_fit["units"], train_fit["lgbm"], train_fit["Branch"])
alpha_valid = fit_dispersion(valid["units"], valid["lgbm"], valid["Branch"])
print("\nNB alpha per branch (train / validation):\n", pd.concat([alpha, alpha_valid], axis=1).round(3))
a = valid["Branch"].astype(str).map(alpha_valid)
ll_p, ll_nb = poisson_vs_nb_loglik(valid["units"], valid["lgbm"], a)
print(f"Mean log-likelihood on validation: Poisson {ll_p:.4f}  NB {ll_nb:.4f}")

for name, al in [("train alpha", valid["Branch"].astype(str).map(alpha)), ("valid alpha", a)]:
    lo, hi = nb_quantile(valid["lgbm"], al, 0.10), nb_quantile(valid["lgbm"], al, 0.90)
    inside = ((valid["units"] >= lo) & (valid["units"] <= hi)).mean()
    below = (valid["units"] <= nb_quantile(valid["lgbm"], al, 0.5)).mean()
    print(f"{name}: P10-P90 coverage {inside:.3f}, share <= P50 {below:.3f}")

imp = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values(ascending=False)
print("\nTop features (gain share):\n", (imp / imp.sum()).head(10).round(3).to_string())

# Step 4: prep policies on validation (alpha from validation residuals; final run uses it on test).
from inventory import critical_ratio, recommend, simulate, summarize  # noqa: E402

valid["prep_naive"] = valid["naive"]
valid["prep_mean"] = valid["lgbm"].round()
valid["prep_nb"] = recommend(valid, "lgbm", a)
print("\nCritical ratio by class:\n", critical_ratio(valid).groupby(valid["shelf_class"], observed=True).describe()[["min", "50%", "max"]].round(2))
rows = []
for policy in ["prep_naive", "prep_mean", "prep_nb"]:
    s = summarize(simulate(valid, policy), ["shelf_class"])
    s.loc["ALL"] = s.sum(numeric_only=True)
    s.loc["ALL", "service_level"] = s.loc["ALL", "sold"] / (s.loc["ALL", "sold"] + s.loc["ALL", "missed"])
    s["policy"] = policy
    rows.append(s.reset_index())
print(pd.concat(rows).set_index(["policy", "shelf_class"]).round(2).to_string())

# Step 5: anomaly detection on validation, plus synthetic injection test.
from anomaly import alerts, inject, inject_sustained, score_categories, score_products, score_sustained  # noqa: E402

prod = score_products(valid, "lgbm", a)
cat = score_categories(valid, "lgbm", a)
sus = score_sustained(valid, "lgbm", a)
al = alerts(prod, cat, sus)
days = valid["Date"].nunique()
print(f"\nValidation alerts: {len(al)} over {days} days")
print(al.groupby(["Branch", "level", "flag"]).size().unstack(fill_value=0).to_string())
print("Alerts per branch-day:", round(len(al) / (days * 3), 2))
print("\nTop 8 by severity:")
for m in al.nlargest(8, "severity")["message"]:
    print(" ", m)

inj = inject(valid, "lgbm")
sp = score_products(inj, "lgbm", a)
real_gap = (inj["units"] - valid["units"]).abs()
for kind in ["zero", "drop", "spike"]:
    mask = inj["injected"] == kind
    effective = mask & (real_gap >= MIN_GAP_CHECK) if (MIN_GAP_CHECK := 3) else mask
    print(f"Injected {kind}: {mask.sum()} rows, recall {(sp.loc[mask, 'flag'] != '').mean():.2f}, "
          f"recall where change >= 3 units {(sp.loc[effective, 'flag'] != '').mean():.2f}")
clean = inj["injected"] == ""
print(f"False-alert rate on untouched rows: {(sp.loc[clean, 'flag'] != '').mean():.4f}")

inj = inject_sustained(valid)
ss = score_sustained(inj, "lgbm", a.loc[inj.index])
flagged = ss[ss["flag"] == "sustained_drop"]
planted = inj[inj["injected"] != ""]
episodes = planted[["Branch", "Product", "injected"]].drop_duplicates("injected")


def episode_in_window(row):
    # The 7-day window ending on the alert date overlaps a planted episode in the same series.
    m = planted[(planted["Branch"] == row["Branch"]) & (planted["Product"] == row["Product"].split(" (")[0])
                & (planted["Date"] > row["Date"] - pd.Timedelta(days=7)) & (planted["Date"] <= row["Date"])]
    return m["injected"].iloc[0] if len(m) else ""


hit = flagged.apply(episode_in_window, axis=1) if len(flagged) else pd.Series(dtype=str)
detected = set(hit[hit != ""])
print(f"\nSustained drops injected: {len(episodes)}, detected: {len(detected)} ({len(detected)/len(episodes):.0%})")
print(f"Sustained-drop alerts not caused by a planted episode: {(hit == '').sum()}")
ep = episodes.assign(found=episodes["injected"].isin(detected))
mean = valid.groupby(["Branch", "Product"], observed=True)["units"].mean().rename("mean_units").reset_index()
ep = ep.merge(mean, on=["Branch", "Product"])
print(ep.groupby(pd.cut(ep["mean_units"], [0, 1.5, 3, 100]), observed=True)["found"].agg(["size", "mean"]).round(2))
