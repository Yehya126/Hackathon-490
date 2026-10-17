"""Model selection on validation only (the test period is not loaded).

Each variant picks its boosting rounds by early stopping on an inner split of the training
period (Apr 18 - May 15), is refit on the full training period with that round count, and is
then scored on validation (May 16 - Jun 8), which none of the choices have seen.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import add_features  # noqa: E402
from forecast import accuracy, fit_dispersion, fit_lgbm, poisson_vs_nb_loglik, weekday_average  # noqa: E402
from panel import TRAIN_END, build_daily_panel, load_transactions, split_by_date  # noqa: E402

INNER_END = TRAIN_END - pd.Timedelta(days=28)

df = add_features(build_daily_panel(load_transactions()))
train, valid, _ = split_by_date(df)
valid = valid.copy()
inner_train, inner_valid = train[train["Date"] <= INNER_END], train[train["Date"] > INNER_END]

valid["weekday_avg"] = weekday_average(valid)
valid["baseline_only"] = valid["baseline"]
variants = ["weekday_avg", "baseline_only"]
for mode in ["raw", "linear", "offset"]:
    _, rounds = fit_lgbm(inner_train, inner_valid, mode=mode)
    model, _ = fit_lgbm(train, rounds=rounds, mode=mode)
    valid[mode] = model.predict(valid)
    variants.append(mode)
    print(f"{mode}: {rounds} rounds")

rows = []
for v in variants:
    acc = accuracy(valid, v)
    alpha = fit_dispersion(valid["units"], valid[v], valid["Branch"])
    _, ll = poisson_vs_nb_loglik(valid["units"], valid[v], valid["Branch"].astype(str).map(alpha))
    rows.append({"model": v, **{f"WAPE {b}": acc.loc[b, "WAPE"] for b in acc.index},
                 "bias": acc.loc["ALL", "bias"], "NB loglik": round(ll, 4)})
print("\nValidation (May 16 - Jun 8):\n", pd.DataFrame(rows).set_index("model").to_string())

wd = valid.assign(day=valid["Date"].dt.day_name()).groupby("day")[["units"] + variants].sum()
bias = (wd[variants].div(wd["units"], axis=0) - 1).mul(100).round(1)
print("\nBias by weekday, % (validation):\n", bias.to_string())

totals = valid.groupby("Date")[["units"] + variants].sum()
peak = totals[totals.index.dayofweek == 1]
print("\nTuesday totals (validation):\n", peak.round(0).to_string())
print("\nMax forecast vs max training value per series exceeded on",
      int((valid.groupby(["Branch", "Product"], observed=True)["offset"].max()
           > train.groupby(["Branch", "Product"], observed=True)["units"].max()).sum()), "series (offset)")
