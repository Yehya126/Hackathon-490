from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import DataValidationError, append_new_transactions, load_transaction_data, validate_transactions
from src.feature_engineering import aggregate_features, branch_product_features, matched_product_table
from src.fixed_effects import compare_models, evaluate_fixed_effects, fit_fixed_effects
from src.pipeline import PREDICTOR_SCHEMA_VERSION, _predictor_records


def transaction_frame() -> pd.DataFrame:
    rows = []
    source = 0
    for branch, cost_factor, price_factor in (("A", 1.0, 1.0), ("B", 1.2, 1.05)):
        for product, category, cost, price in (("Coffee", "Drinks", 2.0, 5.0), ("Cake", "Food", 3.0, 7.0)):
            for month in (1, 2, 3):
                quantity = 20
                unit_cost, unit_price, discount = cost * cost_factor, price * price_factor, 0.1
                revenue = round(quantity * unit_price * (1 - discount), 2)
                rows.append({
                    "Source Row ID": source, "Transaction ID": f"T-{source}", "Date": f"2026-{month:02d}-01",
                    "Time": "12:00:00", "Branch": branch, "Customer ID": None, "Order Type": "Dine-in",
                    "Product": product, "Category": category, "Quantity": quantity,
                    "Unit Price (USD)": unit_price, "Discount Rate": discount, "Unit Cost (USD)": unit_cost,
                    "Line Revenue (USD)": revenue, "Line Gross Profit (USD)": round(revenue - quantity * unit_cost, 2),
                })
                source += 1
    return pd.DataFrame(rows)


def typed_frame() -> pd.DataFrame:
    return validate_transactions(transaction_frame())[0]


def test_accounting_validation_and_legitimate_transaction_repeats():
    raw = transaction_frame()
    raw.loc[1, "Transaction ID"] = raw.loc[0, "Transaction ID"]
    typed, report = validate_transactions(raw)
    assert len(typed) == len(raw)
    assert report.transactions == len(raw) - 1
    broken = raw.copy()
    broken.loc[0, "Line Revenue (USD)"] += 5
    with pytest.raises(DataValidationError, match="Revenue arithmetic"):
        validate_transactions(broken)


def test_split_loader_preserves_provenance(tmp_path):
    raw = transaction_frame()
    raw.iloc[:4].to_csv(tmp_path / "train.csv", index=False)
    raw.iloc[4:8].to_csv(tmp_path / "validation.csv", index=False)
    raw.iloc[8:].to_csv(tmp_path / "test.csv", index=False)
    loaded, report = load_transaction_data(tmp_path)
    assert len(loaded) == len(raw)
    assert set(loaded["source_split"]) == {"train", "validation", "test"}
    assert report.revenue_mismatches == 0


def test_weighted_features_matching_and_incomplete_period_flag():
    typed = typed_frame()
    monthly = aggregate_features(typed, "monthly")
    assert monthly["total_units"].sum() == typed["Quantity"].sum()
    panel = branch_product_features(typed)
    assert len(matched_product_table(panel)) == 4
    assert not monthly.loc[monthly["period"] == "2026-03", "period_complete"].any()


def test_incremental_ingestion_rejects_seen_source_rows():
    with pytest.raises(DataValidationError, match="already-ingested"):
        append_new_transactions(typed_frame(), transaction_frame().iloc[:1])


def test_fixed_effects_recover_known_branch_multipliers():
    panel = branch_product_features(typed_frame())
    cost = fit_fixed_effects(panel, "unit_cost", "huber", reference_branch="A", reference_product="Cake")
    price = fit_fixed_effects(panel, "listed_unit_price", "ols", reference_branch="A", reference_product="Cake")
    assert cost.observations == 4
    assert cost.branch_multipliers["B"] == pytest.approx(1.2, rel=1e-6)
    assert price.branch_multipliers["B"] == pytest.approx(1.05, rel=1e-6)


def test_huber_and_ols_comparison_has_benchmark_stability():
    panel = branch_product_features(typed_frame())
    comparison = compare_models(panel, "unit_cost")
    assert set(comparison) == {"huber", "ols", "benchmark_stability_log_effect_range"}
    assert max(comparison["benchmark_stability_log_effect_range"].values()) < 1e-10


def test_holdout_scoring_never_refits_and_flags_unseen_levels():
    panel = branch_product_features(typed_frame())
    model = fit_fixed_effects(panel, "unit_cost", "huber", reference_branch="A", reference_product="Cake")
    metrics, predictions = evaluate_fixed_effects(model, panel)
    assert metrics["coverage"] == 1
    assert metrics["mape_pct"] == pytest.approx(0, abs=1e-8)
    unseen = panel.iloc[[0]].copy()
    unseen["Branch"] = "NEW"
    unseen_metrics, unseen_predictions = evaluate_fixed_effects(model, unseen)
    assert unseen_metrics["coverage"] == 0
    assert unseen_predictions["scorable"].eq(False).all()


def test_predictor_contract_is_decision_neutral():
    panel = branch_product_features(typed_frame())
    cost_model = fit_fixed_effects(panel, "unit_cost", "huber", reference_branch="A", reference_product="Cake")
    price_model = fit_fixed_effects(panel, "listed_unit_price", "huber", reference_branch="A", reference_product="Cake")
    _, cost = evaluate_fixed_effects(cost_model, panel)
    _, price = evaluate_fixed_effects(price_model, panel)
    output = _predictor_records(panel, cost, price)
    assert output["schema_version"].eq(PREDICTOR_SCHEMA_VERSION).all()
    assert output["prediction_complete"].all()
    assert {"predicted_unit_cost", "predicted_listed_unit_price", "cost_residual_pct", "price_residual_pct"}.issubset(output)
    forbidden = {"recommendation", "priority_score", "action", "status"}
    assert forbidden.isdisjoint(output.columns)
