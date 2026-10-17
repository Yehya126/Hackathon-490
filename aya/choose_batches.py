"""Choose the two-batch cutoff hour and morning batch size on validation only (test not loaded)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from batches import attach, two_batch  # noqa: E402
from features import add_features  # noqa: E402
from forecast import fit_dispersion, fit_lgbm  # noqa: E402
from inventory import recommend, simulate  # noqa: E402
from panel import build_daily_panel, load_transactions, split_by_date  # noqa: E402

data = load_transactions()
df = add_features(build_daily_panel(data))
train, valid, _ = split_by_date(df)
model, _ = fit_lgbm(train, valid, mode="offset")
valid = valid.copy()
valid["forecast"] = model.predict(valid)
alpha = fit_dispersion(valid["units"], valid["forecast"], valid["Branch"])
fresh = valid[valid["shelf_class"] == "fresh"].copy()
a = fresh["Branch"].astype(str).map(alpha)

hours = data[data["shelf_class"] == "fresh"] if "shelf_class" in data else data.merge(
    df[["Product", "shelf_class"]].drop_duplicates(), on="Product")
fresh_hours = hours[hours["shelf_class"] == "fresh"]
cum = fresh_hours.groupby("Hour")["Quantity"].sum().cumsum()
print("Cumulative share of fresh units by hour:\n", (cum / cum.iloc[-1]).round(2).to_string())

fresh["prep"] = recommend(fresh, "forecast", a)
one = simulate(fresh, "prep")
print(f"\nOne batch: profit {one.profit.sum():,.0f}  waste {one.waste_cost.sum():,.0f}  "
      f"missed {one.missed_profit.sum():,.0f}  service {one.sold.sum() / (one.sold.sum() + one.missed.sum()):.1%}")
rows = []
for hour in [10, 11, 12, 13, 14, 15, 16]:
    f = attach(fresh, data, hour)
    for level in [None, 0.5, 0.7, 0.8, 0.9]:
        s = two_batch(f, "forecast", a, level)
        rows.append({"cutoff": hour, "morning_level": "critical ratio" if level is None else level,
                     "profit": s.profit.sum(), "waste_cost": s.waste_cost.sum(), "missed_profit": s.missed_profit.sum(),
                     "service": s.sold.sum() / (s.sold.sum() + s.missed.sum())})
res = pd.DataFrame(rows).sort_values("profit", ascending=False)
print("\nTwo batches (validation, fresh products):\n", res.head(12).round(3).to_string(index=False))
