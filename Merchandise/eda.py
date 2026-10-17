"""Forecast-oriented EDA: prints findings and saves figures to Merchandise/outputs/eda/."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "src"))
from panel import ROOT, build_daily_panel, load_transactions, split_by_date  # noqa: E402

OUT = Path(__file__).parent / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(ROOT / "stories_transactions.csv")
data = load_transactions()
print(f"Raw rows {len(raw)}, cleaned rows {len(data)}, removed {len(raw) - len(data)}")
print("Raw columns not in cleaned:", sorted(set(raw.columns) - set(data.columns)))
removed = raw[~raw["Transaction ID"].isin(data["Transaction ID"])]
print("Removed rows sample:\n", removed.head(5).to_string())

panel = build_daily_panel(data)
train, _, _ = split_by_date(panel)

# Branch totals over time
totals = panel.groupby(["Date", "Branch"])["units"].sum().unstack()
ax = totals.rolling(7).mean().plot(figsize=(10, 4), title="Units per day (7-day mean)")
ax.figure.savefig(OUT / "branch_trend.png", bbox_inches="tight")
plt.close("all")

# Weekday profile per branch
panel["weekday"] = panel["Date"].dt.dayofweek
wd = panel.groupby(["Branch", "weekday"])["units"].sum().unstack(0)
wd = wd / wd.mean()
print("\nWeekday index per branch (Mon=0):\n", wd.round(2).to_string())
ax = wd.plot(kind="bar", figsize=(8, 4), title="Weekday index (1 = average)")
ax.figure.savefig(OUT / "weekday_profile.png", bbox_inches="tight")
plt.close("all")

# Trend
monthly = panel.groupby([panel["Date"].dt.month, "Branch"])["units"].sum().unstack()
print("\nMonthly units:\n", monthly.to_string())

# Sparsity and dispersion per series (train only)
series = train.groupby(["Branch", "Product"])["units"].agg(["mean", "var", lambda s: (s == 0).mean()])
series.columns = ["mean", "var", "zero_share"]
series["var_to_mean"] = series["var"] / series["mean"]
print("\nSeries stats (train):\n", series.describe().round(2).to_string())

# Lifecycle: series that start late or stop early
life = data.groupby(["Branch", "Product"])["Date"].agg(["min", "max"])
late = life[(life["min"] > pd.Timestamp("2026-01-14")) | (life["max"] < pd.Timestamp("2026-06-16"))]
print(f"\nSeries starting after Jan 14 or ending before Jun 16: {len(late)}")
print(late.to_string())

# Price changes
prices = data.groupby(["Product"])["Unit Price (USD)"].nunique()
print("\nProducts with more than one list price:", (prices > 1).sum())
pc = data.groupby(["Product", data["Date"].dt.month])["Unit Price (USD)"].median().unstack()
print(pc[pc.nunique(axis=1) > 1].to_string())

# Outlier days: branch total vs same-weekday rolling median
bt = panel.groupby(["Branch", "Date"])["units"].sum().reset_index()
bt["weekday"] = bt["Date"].dt.dayofweek
bt["expected"] = bt.groupby(["Branch", "weekday"])["units"].transform(
    lambda s: s.shift(1).rolling(4, min_periods=2).median())
bt["ratio"] = bt["units"] / bt["expected"]
print("\nMost unusual branch-days (ratio to same-weekday median of previous 4 weeks):")
print(bt.dropna().sort_values("ratio").iloc[list(range(5)) + list(range(-5, 0))].round(2).to_string())

# Series-level extreme days
p = panel.copy()
p["expected"] = p.groupby(["Branch", "Product", "weekday"])["units"].transform(
    lambda s: s.shift(1).rolling(4, min_periods=2).mean())
p["gap"] = p["units"] - p["expected"]
print("\nLargest series-level spikes:\n",
      p.nlargest(8, "gap")[["Branch", "Product", "Date", "units", "expected"]].round(1).to_string())
long_zero = p.assign(z=p["units"].eq(0)).groupby(["Branch", "Product"])["z"].apply(
    lambda s: s.groupby((~s).cumsum()).sum().max())
print("\nLongest zero-sales streak per series (top 8):\n", long_zero.nlargest(8).to_string())

# Hour profile
hours = data.groupby(["Branch", "Hour"])["Quantity"].sum().unstack(0)
ax = (hours / hours.sum()).plot(figsize=(8, 4), title="Share of units by hour")
ax.figure.savefig(OUT / "hour_profile.png", bbox_inches="tight")
print("\nTrading hours:", data["Hour"].min(), "-", data["Hour"].max())
print("Order type share:\n", data["Order Type"].value_counts(normalize=True).round(2).to_string())
