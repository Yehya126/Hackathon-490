"""Training-only fitted basket and profitability transformations."""

from __future__ import annotations

import pandas as pd


class BasketEncoder:
    """Multi-hot encode product presence with a training-fitted vocabulary."""

    def __init__(self) -> None:
        self.products_: pd.Index | None = None

    def fit(self, frame: pd.DataFrame) -> "BasketEncoder":
        self._validate(frame)
        self.products_ = pd.Index(sorted(frame["Product"].dropna().unique()), name="Product")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._validate(frame)
        if self.products_ is None:
            raise RuntimeError("BasketEncoder must be fitted before transform")
        encoded = pd.crosstab(frame["Transaction ID"], frame["Product"]).gt(0)
        return encoded.reindex(columns=self.products_, fill_value=False).astype(bool)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    @staticmethod
    def _validate(frame: pd.DataFrame) -> None:
        missing = {"Transaction ID", "Product"}.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing basket columns: {sorted(missing)}")


class ProductProfitEstimator:
    """Estimate typical per-unit product profit from training lines only."""

    def __init__(self, statistic: str = "median") -> None:
        if statistic not in {"median", "mean"}:
            raise ValueError("statistic must be 'median' or 'mean'")
        self.statistic = statistic
        self.product_profit_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "ProductProfitEstimator":
        needed = {"Product", "Quantity", "Line Gross Profit (USD)"}
        if missing := needed.difference(frame.columns):
            raise ValueError(f"Missing profit columns: {sorted(missing)}")
        if (frame["Quantity"] <= 0).any():
            raise ValueError("Quantity must be positive to estimate unit profit")
        unit_profit = frame["Line Gross Profit (USD)"] / frame["Quantity"]
        values = frame.assign(_unit_profit=unit_profit).groupby("Product")["_unit_profit"]
        self.product_profit_ = getattr(values, self.statistic)().sort_index()
        return self

    def transform(self, products: pd.Series | pd.Index) -> pd.Series:
        if self.product_profit_ is None:
            raise RuntimeError("ProductProfitEstimator must be fitted before transform")
        index = pd.Index(products)
        return pd.Series(index.map(self.product_profit_), index=index, name="typical_unit_profit")


def split_by_branch(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return independent branch frames for future branch-specific rule fitting."""
    if "Branch" not in frame:
        raise ValueError("Missing Branch column")
    return {str(branch): part.copy() for branch, part in frame.groupby("Branch", observed=True)}

