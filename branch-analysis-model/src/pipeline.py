"""Leakage-safe training, validation and test orchestration for predictor outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data_processing import load_transaction_data
from .feature_engineering import branch_product_features, matched_product_table, structural_fingerprint
from .fixed_effects import compare_models, evaluate_fixed_effects, fit_fixed_effects
from .visualization import generate_predictor_report


PREDICTOR_SCHEMA_VERSION = "1.0.0"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _label_predictions(frame: pd.DataFrame, split: str, target_model: str, method: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "target_model", target_model)
    result.insert(0, "method", method)
    result.insert(0, "split", split)
    return result


def _predictor_records(
    test_features: pd.DataFrame, cost_predictions: pd.DataFrame, price_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the stable, decision-neutral record contract for downstream models."""
    keys = ["Branch", "Product"]
    cost = cost_predictions[keys + [
        "predicted", "log_residual", "absolute_error", "absolute_percentage_error", "scorable",
    ]].rename(columns={
        "predicted": "predicted_unit_cost", "log_residual": "cost_log_residual",
        "absolute_error": "cost_absolute_error", "absolute_percentage_error": "cost_absolute_percentage_error",
        "scorable": "cost_scorable",
    })
    price = price_predictions[keys + [
        "predicted", "log_residual", "absolute_error", "absolute_percentage_error", "scorable",
    ]].rename(columns={
        "predicted": "predicted_listed_unit_price", "log_residual": "price_log_residual",
        "absolute_error": "price_absolute_error", "absolute_percentage_error": "price_absolute_percentage_error",
        "scorable": "price_scorable",
    })
    columns = [
        "Branch", "Product", "Category", "total_units", "total_revenue", "total_cogs",
        "total_gross_profit", "gross_margin", "listed_unit_price", "unit_cost",
        "realized_revenue_per_unit", "effective_discount_rate", "periods",
    ]
    output = test_features[columns].merge(cost, on=keys, how="left").merge(price, on=keys, how="left")
    output["cost_residual_pct"] = 100 * (output["unit_cost"] / output["predicted_unit_cost"] - 1)
    output["price_residual_pct"] = 100 * (output["listed_unit_price"] / output["predicted_listed_unit_price"] - 1)
    output["prediction_complete"] = output["cost_scorable"].fillna(False) & output["price_scorable"].fillna(False)
    output.insert(0, "schema_version", PREDICTOR_SCHEMA_VERSION)
    output.insert(1, "data_split", "test")
    return output.sort_values(keys).reset_index(drop=True)


class PredictorEngine:
    """Train on train.csv, diagnose on validation.csv, and score test.csv once."""

    def __init__(self, output_dir: str | Path = "outputs"):
        self.output_dir = Path(output_dir)

    def run(self, data_dir: str | Path, *, strict: bool = True) -> dict[str, Any]:
        data_dir = Path(data_dir)
        required = [data_dir / f"{split}.csv" for split in ("train", "validation", "test")]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Predictor evaluation requires chronological split files; missing: {missing}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Development phase: the test file is not loaded until the model protocol is frozen.
        train, train_report = load_transaction_data(data_dir, ["train"], strict=strict)
        validation, validation_report = load_transaction_data(data_dir, ["validation"], strict=strict)
        train_features = branch_product_features(train)
        validation_features = branch_product_features(validation)
        train_panel = matched_product_table(train_features)
        validation_panel = matched_product_table(validation_features)
        fingerprint = structural_fingerprint(train_panel)

        models = {
            "cost": compare_models(train_panel, "unit_cost"),
            "price": compare_models(train_panel, "listed_unit_price"),
        }
        validation_metrics: dict[str, Any] = {}
        prediction_frames: list[pd.DataFrame] = []
        for model_name in ("cost", "price"):
            validation_metrics[model_name] = {}
            for method in ("huber", "ols"):
                metrics, predictions = evaluate_fixed_effects(models[model_name][method], validation_panel)
                validation_metrics[model_name][method] = metrics
                prediction_frames.append(_label_predictions(predictions, "validation", model_name, method))

        validation_effect_stability: dict[str, Any] = {}
        for model_name, target in (("cost", "unit_cost"), ("price", "listed_unit_price")):
            diagnostic = fit_fixed_effects(
                validation_panel, target, "huber",
                reference_branch=models[model_name]["huber"]["reference_branch"],
                reference_product=models[model_name]["huber"]["reference_product"],
            )
            trained = models[model_name]["huber"]["branch_effects"]
            validation_effect_stability[model_name] = {
                branch: {
                    "train_log_effect": float(trained[branch]),
                    "validation_log_effect": float(diagnostic.branch_effects[branch]),
                    "absolute_difference": abs(float(trained[branch]) - float(diagnostic.branch_effects[branch])),
                }
                for branch in sorted(trained)
            }

        protocol = {
            "predictor_schema_version": PREDICTOR_SCHEMA_VERSION,
            "primary_method": "huber", "structural_model_fingerprint": fingerprint,
            "training_only_model_fit": True,
            "validation_use": "Held-out method comparison and stability diagnostics; no test-driven tuning.",
            "decision_policy_included": False,
        }
        protocol_hash = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        protocol["protocol_sha256"] = protocol_hash
        _write_json(self.output_dir / "frozen_evaluation_protocol.json", protocol)

        # Final evaluation phase: load test only after the predictor protocol is fixed.
        test, test_report = load_transaction_data(data_dir, ["test"], strict=strict)
        test_features = branch_product_features(test)
        test_panel = matched_product_table(test_features)
        test_metrics: dict[str, Any] = {}
        primary_predictions: dict[str, pd.DataFrame] = {}
        for model_name in ("cost", "price"):
            test_metrics[model_name] = {}
            for method in ("huber", "ols"):
                metrics, predictions = evaluate_fixed_effects(models[model_name][method], test_panel)
                test_metrics[model_name][method] = metrics
                prediction_frames.append(_label_predictions(predictions, "test", model_name, method))
                if method == "huber":
                    primary_predictions[model_name] = predictions

        train_ids, validation_ids, test_ids = (set(frame["Transaction ID"]) for frame in (train, validation, test))
        leakage = {
            "train_validation_transaction_overlap": len(train_ids & validation_ids),
            "train_test_transaction_overlap": len(train_ids & test_ids),
            "validation_test_transaction_overlap": len(validation_ids & test_ids),
        }
        if "Source Row ID" in train and "Source Row ID" in validation and "Source Row ID" in test:
            train_rows, validation_rows, test_rows = (set(frame["Source Row ID"]) for frame in (train, validation, test))
            leakage.update({
                "train_validation_source_row_overlap": len(train_rows & validation_rows),
                "train_test_source_row_overlap": len(train_rows & test_rows),
                "validation_test_source_row_overlap": len(validation_rows & test_rows),
            })
        if any(leakage.values()):
            raise ValueError(f"Split leakage detected: {leakage}")

        predictor_frame = _predictor_records(
            test_features, primary_predictions["cost"], primary_predictions["price"]
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        evaluation = {
            "evaluation_design": {
                "type": "chronological retrospective holdout", "model_fit_split": "train only",
                "validation_split_use": protocol["validation_use"], "final_evaluation_split": "test",
                "primary_method": "huber", "protocol_sha256": protocol_hash,
                "important_disclosure": "The test files were used by an earlier combined descriptive run before split isolation was implemented; this is retrospective rather than a pristine blind test.",
            },
            "split_validation": {
                "train": train_report.to_dict(), "validation": validation_report.to_dict(),
                "test": test_report.to_dict(), "leakage_checks": leakage,
            },
            "train_fit": {
                model_name: {
                    method: {k: models[model_name][method][k] for k in ("observations", "rmse", "mae", "r_squared")}
                    for method in ("huber", "ols")
                } for model_name in ("cost", "price")
            },
            "validation_metrics": validation_metrics, "test_metrics": test_metrics,
            "validation_branch_effect_stability": validation_effect_stability,
        }
        model_artifact = {
            "schema_version": PREDICTOR_SCHEMA_VERSION, "generated_at_utc": generated_at,
            "training_split": "train", "primary_method": "huber",
            "structural_fingerprint": fingerprint, "models": models,
        }
        predictor_payload = {
            "schema_version": PREDICTOR_SCHEMA_VERSION, "generated_at_utc": generated_at,
            "data_split": "test", "model_method": "huber",
            "decision_policy_included": False,
            "description": "Decision-neutral branch-product cost and listed-price predictions for a downstream decision model.",
            "field_semantics": {
                "predicted_unit_cost": "Training-fitted expected unit cost for this branch-product pair.",
                "predicted_listed_unit_price": "Training-fitted expected listed unit price for this branch-product pair.",
                "cost_residual_pct": "100 * (observed unit cost / predicted unit cost - 1).",
                "price_residual_pct": "100 * (observed listed price / predicted listed price - 1).",
                "prediction_complete": "True when both branch and product levels were seen during training.",
            },
            "records": predictor_frame.to_dict("records"),
        }

        pd.concat(prediction_frames, ignore_index=True).to_csv(
            self.output_dir / "holdout_predictions.csv", index=False
        )
        predictor_frame.to_csv(self.output_dir / "predictor_output.csv", index=False)
        _write_json(self.output_dir / "predictor_output.json", predictor_payload)
        _write_json(self.output_dir / "trained_models.json", model_artifact)
        _write_json(self.output_dir / "holdout_evaluation.json", evaluation)
        report_path = generate_predictor_report(
            predictor_frame, models, evaluation, self.output_dir / "predictor_report.html"
        )
        return {
            "evaluation_mode": "chronological_retrospective_holdout",
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
            "prediction_records": len(predictor_frame),
            "complete_prediction_records": int(predictor_frame["prediction_complete"].sum()),
            "decision_policy_included": False,
            "output_dir": str(self.output_dir.resolve()),
            "predictor_output": str((self.output_dir / "predictor_output.json").resolve()),
            "report": str(report_path.resolve()),
        }
