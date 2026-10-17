"""Validation only (test not used): 1-14 day forecasts, multi-horizon model vs baseline formula.

Rounds are picked by early stopping on an inner split of training (targets Apr 18 - May 15),
the model is refit on all training targets, then scored on validation targets (May 16 - Jun 8).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import add_features  # noqa: E402
from forecast import fit_lgbm  # noqa: E402
from horizon import HorizonModel, stacked  # noqa: E402
from panel import TRAIN_END, VALIDATION_END, build_daily_panel, load_transactions, split_by_date  # noqa: E402

INNER_END = TRAIN_END - pd.Timedelta(days=28)

df = add_features(build_daily_panel(load_transactions()))
rows = stacked(df)
train = rows[rows["Date"] <= TRAIN_END]
valid = rows[(rows["Date"] > TRAIN_END) & (rows["Date"] <= VALIDATION_END)].dropna(subset=["roll_mean_28"]).copy()

inner = HorizonModel().fit(train[train["Date"] <= INNER_END], train[train["Date"] > INNER_END])
print("Rounds (inner early stopping):", inner.rounds)
model = HorizonModel().fit(train, rounds=inner.rounds)
valid["model"] = model.predict(valid)

# Next-day model for reference at horizon 1
tr, va, _ = split_by_date(df)
nd, rounds = fit_lgbm(tr[tr["Date"] <= INNER_END], tr[tr["Date"] > INNER_END], mode="offset")
nd, _ = fit_lgbm(tr, rounds=rounds, mode="offset")
next_day = pd.Series(nd.predict(va), index=pd.MultiIndex.from_frame(va[["Branch", "Product", "Date"]].astype(str)))


def wape(pred, actual):
    return (pred - actual).abs().sum() / actual.sum()


out = []
for h, g in valid.groupby("horizon"):
    b = g.groupby(["Branch", "Date"], observed=True)[["units", "model", "baseline"]].sum()
    out.append({"days ahead": h,
                "product WAPE model": wape(g["model"], g["units"]),
                "product WAPE baseline": wape(g["baseline"], g["units"]),
                "branch-day WAPE model": wape(b["model"], b["units"]),
                "bias model": g["model"].sum() / g["units"].sum() - 1,
                "bias baseline": g["baseline"].sum() / g["units"].sum() - 1})
table = pd.DataFrame(out).set_index("days ahead")
print("\nValidation by horizon:\n", (table * 100).round(1).to_string())

h1 = valid[valid["horizon"] == 1]
key = pd.MultiIndex.from_frame(h1[["Branch", "Product", "Date"]].astype(str))
print(f"\nHorizon 1: multi-horizon model WAPE {wape(h1['model'], h1['units']):.1%}, "
      f"next-day model WAPE {wape(pd.Series(next_day.reindex(key).to_numpy(), index=h1.index), h1['units']):.1%}")
print("Average over 1-14 days: model WAPE {:.1%}, baseline WAPE {:.1%}".format(
    table["product WAPE model"].mean(), table["product WAPE baseline"].mean()))
