"""Input discovery, validation and safe incremental ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Transaction ID", "Date", "Time", "Branch", "Customer ID", "Order Type",
    "Product", "Category", "Quantity", "Unit Price (USD)", "Discount Rate",
    "Unit Cost (USD)", "Line Revenue (USD)", "Line Gross Profit (USD)",
]
NUMERIC_COLUMNS = [
    "Quantity", "Unit Price (USD)", "Discount Rate", "Unit Cost (USD)",
    "Line Revenue (USD)", "Line Gross Profit (USD)",
]


class DataValidationError(ValueError):
    """Raised for data that cannot safely enter the analytical pipeline."""


@dataclass
class ValidationReport:
    rows: int
    transactions: int
    branches: int
    products: int
    date_min: str
    date_max: str
    exact_duplicate_lines: int = 0
    duplicate_source_row_ids: int = 0
    revenue_mismatches: int = 0
    gross_profit_mismatches: int = 0
    missing_optional_customer_ids: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def discover_input_files(path: str | Path, splits: Iterable[str] | None = None) -> list[Path]:
    """Accept one CSV or the repository's train/validation/test directory layout."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    requested = list(splits or ("train", "validation", "test"))
    split_files = [path / f"{name}.csv" for name in requested]
    existing = [p for p in split_files if p.exists()]
    if existing:
        if len(existing) != len(split_files):
            missing = [str(p.name) for p in split_files if not p.exists()]
            raise FileNotFoundError(f"Missing requested split files: {missing}")
        return existing
    conventional = path / "stories_transactions.csv"
    if conventional.exists():
        return [conventional]
    csvs = sorted(path.glob("*.csv"))
    if len(csvs) == 1:
        return csvs
    raise FileNotFoundError(
        f"Expected train/validation/test.csv or stories_transactions.csv under {path}"
    )


def load_transaction_data(
    path: str | Path,
    splits: Iterable[str] | None = None,
    *,
    strict: bool = True,
    rounding_tolerance: float = 0.02,
) -> tuple[pd.DataFrame, ValidationReport]:
    frames = []
    for file in discover_input_files(path, splits):
        frame = pd.read_csv(file)
        frame["source_split"] = file.stem if file.stem in {"train", "validation", "test"} else "all"
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    return validate_transactions(raw, strict=strict, rounding_tolerance=rounding_tolerance)


def validate_transactions(
    raw: pd.DataFrame, *, strict: bool = True, rounding_tolerance: float = 0.02
) -> tuple[pd.DataFrame, ValidationReport]:
    """Return a typed copy and an accounting/data-quality report."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    df = raw.copy()
    failures: list[str] = []
    for column in NUMERIC_COLUMNS:
        original_missing = df[column].isna()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        introduced = df[column].isna() & ~original_missing
        if introduced.any():
            failures.append(f"{column}: {int(introduced.sum())} non-numeric values")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        failures.append(f"Date: {int(df['Date'].isna().sum())} invalid values")
    timestamp_text = df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str)
    df["timestamp"] = pd.to_datetime(timestamp_text, errors="coerce")
    if df["timestamp"].isna().any():
        failures.append(f"Time: {int(df['timestamp'].isna().sum())} invalid values")

    required_text = ["Transaction ID", "Branch", "Product", "Category", "Order Type"]
    for column in required_text:
        empty = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if empty.any():
            failures.append(f"{column}: {int(empty.sum())} missing values")
    checks = {
        "Quantity must be a positive integer": (df["Quantity"] <= 0) | (df["Quantity"] % 1 != 0),
        "Unit Price (USD) must be nonnegative": df["Unit Price (USD)"] < 0,
        "Unit Cost (USD) must be nonnegative": df["Unit Cost (USD)"] < 0,
        "Discount Rate must be between 0 and 1": ~df["Discount Rate"].between(0, 1),
    }
    for label, mask in checks.items():
        if mask.fillna(True).any():
            failures.append(f"{label}: {int(mask.fillna(True).sum())} rows")

    expected_revenue = df["Quantity"] * df["Unit Price (USD)"] * (1 - df["Discount Rate"])
    expected_gp = df["Line Revenue (USD)"] - df["Quantity"] * df["Unit Cost (USD)"]
    revenue_bad = (df["Line Revenue (USD)"] - expected_revenue).abs() > rounding_tolerance
    gp_bad = (df["Line Gross Profit (USD)"] - expected_gp).abs() > rounding_tolerance
    duplicate_subset = [c for c in REQUIRED_COLUMNS if c != "Customer ID"]
    exact_duplicates = int(df.duplicated(subset=duplicate_subset, keep=False).sum())
    duplicate_ids = int(df["Source Row ID"].duplicated(keep=False).sum()) if "Source Row ID" in df else 0
    warnings = []
    if exact_duplicates:
        warnings.append(f"Found {exact_duplicates} rows participating in exact duplicate transaction lines; retained for review.")
    if revenue_bad.any():
        failures.append(f"Revenue arithmetic mismatch beyond ${rounding_tolerance:.2f}: {int(revenue_bad.sum())} rows")
    if gp_bad.any():
        failures.append(f"Gross-profit arithmetic mismatch beyond ${rounding_tolerance:.2f}: {int(gp_bad.sum())} rows")

    if failures and strict:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(failures))
    warnings.extend(failures)
    report = ValidationReport(
        rows=len(df), transactions=df["Transaction ID"].nunique(),
        branches=df["Branch"].nunique(), products=df["Product"].nunique(),
        date_min=str(df["Date"].min().date()), date_max=str(df["Date"].max().date()),
        exact_duplicate_lines=exact_duplicates, duplicate_source_row_ids=duplicate_ids,
        revenue_mismatches=int(revenue_bad.sum()), gross_profit_mismatches=int(gp_bad.sum()),
        missing_optional_customer_ids=int(df["Customer ID"].isna().sum()), warnings=warnings,
    )
    df["Quantity"] = df["Quantity"].astype("int64")
    df["calculated_cogs"] = df["Quantity"] * df["Unit Cost (USD)"]
    df["listed_revenue"] = df["Quantity"] * df["Unit Price (USD)"]
    df["discount_amount"] = df["listed_revenue"] - df["Line Revenue (USD)"]
    return df, report


def append_new_transactions(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Validate and append unseen source rows without collapsing legitimate transaction IDs."""
    new, _ = validate_transactions(incoming)
    if "Source Row ID" in existing and "Source Row ID" in new:
        overlap = set(existing["Source Row ID"]).intersection(new["Source Row ID"])
        if overlap:
            raise DataValidationError(f"Incoming data contains {len(overlap)} already-ingested Source Row IDs")
    return pd.concat([existing, new], ignore_index=True)

