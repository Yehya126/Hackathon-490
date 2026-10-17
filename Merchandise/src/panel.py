"""Turn transaction lines into a daily branch x product panel and split it by date."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "cleaned_stories_transactions.csv"
CLASSES_PATH = Path(__file__).with_name("product_classes.csv")

TRAIN_END = pd.Timestamp("2026-05-15")
VALIDATION_END = pd.Timestamp("2026-06-08")

KEYS = ["Branch", "Product"]


def load_transactions(path: str | Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"])
    data["Hour"] = pd.to_timedelta(data["Time"]).dt.components.hours
    return data


def build_daily_panel(data: pd.DataFrame) -> pd.DataFrame:
    """One row per branch, product and calendar day; days without sales get 0 units."""
    daily = data.groupby(KEYS + ["Date"]).agg(
        units=("Quantity", "sum"),
        revenue=("Line Revenue (USD)", "sum"),
        gross_profit=("Line Gross Profit (USD)", "sum"),
        price=("Unit Price (USD)", "mean"),
        unit_cost=("Unit Cost (USD)", "mean"),
        discount=("Discount Rate", "mean"),
        last_sale_hour=("Hour", "max"),
    )
    grid = pd.MultiIndex.from_product(
        [sorted(data["Branch"].unique()), sorted(data["Product"].unique()),
         pd.date_range(data["Date"].min(), data["Date"].max())],
        names=KEYS + ["Date"],
    )
    panel = daily.reindex(grid).reset_index()
    for col in ["units", "revenue", "gross_profit", "discount"]:
        panel[col] = panel[col].fillna(0)
    # Price and cost on zero-sale days: carry the nearest known value within the series.
    for col in ["price", "unit_cost"]:
        panel[col] = panel.groupby(KEYS)[col].transform(lambda s: s.ffill().bfill())

    category = data.drop_duplicates("Product").set_index("Product")["Category"]
    panel["Category"] = panel["Product"].map(category)
    classes = pd.read_csv(CLASSES_PATH).set_index("Product")["shelf_class"]
    panel["shelf_class"] = panel["Product"].map(classes)
    if panel["shelf_class"].isna().any():
        missing = sorted(panel.loc[panel["shelf_class"].isna(), "Product"].unique())
        raise ValueError(f"Products missing a shelf class: {missing}")
    panel["unit_margin"] = panel["price"] - panel["unit_cost"]
    return panel.sort_values(KEYS + ["Date"], ignore_index=True)


def split_by_date(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel[panel["Date"] <= TRAIN_END]
    validation = panel[(panel["Date"] > TRAIN_END) & (panel["Date"] <= VALIDATION_END)]
    test = panel[panel["Date"] > VALIDATION_END]
    return train, validation, test
