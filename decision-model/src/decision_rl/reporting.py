"""Product, category, branch and company paired forecast reports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .registry import DecisionRegistry
from .reward import PortfolioComparison
from .schemas import ForecastRow


_METRICS = (
    "predicted_units",
    "predicted_list_price_usd",
    "predicted_effective_price_usd",
    "predicted_unit_cost_usd",
    "predicted_revenue_usd",
    "predicted_cogs_usd",
    "predicted_gross_profit_usd",
)


def _financials(row: ForecastRow) -> dict[str, float]:
    return {name: float(getattr(row, name)) for name in _METRICS}


def _aggregate(rows: Iterable[ForecastRow], key_function) -> list[dict[str, Any]]:
    groups: dict[Any, list[ForecastRow]] = defaultdict(list)
    for row in rows:
        groups[key_function(row)].append(row)
    result: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        key_values = key if isinstance(key, tuple) else (key,)
        result.append(
            {
                "key": key_values,
                "predicted_units": sum(item.predicted_units for item in group),
                "predicted_revenue_usd": sum(item.predicted_revenue_usd for item in group),
                "predicted_cogs_usd": sum(item.predicted_cogs_usd for item in group),
                "predicted_gross_profit_usd": sum(item.predicted_gross_profit_usd for item in group),
            }
        )
    return result


def _paired_aggregate(
    baseline_rows: Iterable[ForecastRow],
    policy_rows: Iterable[ForecastRow],
    key_function,
) -> list[dict[str, Any]]:
    baseline = {tuple(item["key"]): item for item in _aggregate(baseline_rows, key_function)}
    policy = {tuple(item["key"]): item for item in _aggregate(policy_rows, key_function)}
    if baseline.keys() != policy.keys():
        raise ValueError("Baseline and policy aggregate groups do not align")
    metrics = (
        "predicted_units",
        "predicted_revenue_usd",
        "predicted_cogs_usd",
        "predicted_gross_profit_usd",
    )
    return [
        {
            "key": key,
            "baseline": {name: baseline[key][name] for name in metrics},
            "with_actions": {name: policy[key][name] for name in metrics},
            "delta": {name: policy[key][name] - baseline[key][name] for name in metrics},
        }
        for key in sorted(baseline, key=str)
    ]


def build_portfolio_report(
    comparison: PortfolioComparison,
    registry: DecisionRegistry,
    *,
    split_id: str,
    simulator_version: str,
    policy_checkpoint: str | None = None,
    limitations: Iterable[str] = (),
) -> dict[str, Any]:
    baseline = comparison.baseline.keyed_rows()
    policy = comparison.policy.keyed_rows()
    if baseline.keys() != policy.keys():
        raise ValueError("Baseline and policy forecast rows do not align")
    product_rows = []
    for key in sorted(baseline):
        base = baseline[key]
        changed = policy[key]
        product_rows.append(
            {
                "branch": base.branch_id,
                "category": base.category_id,
                "product": base.product_id,
                "baseline": _financials(base),
                "with_actions": _financials(changed),
                "delta": {
                    name: float(getattr(changed, name) - getattr(base, name)) for name in _METRICS
                },
            }
        )
    decisions = []
    for action in comparison.new_actions:
        definition = registry.get(action.decision_id)
        decision = action.as_dict()
        decision.update(
            {
                "label": definition.label,
                "decision_kind": definition.metadata.get("decision_kind", "intervention"),
                "simulator_capability": definition.simulator_capability,
                "guardrails": list(definition.metadata.get("guardrails", ())),
            }
        )
        decisions.append(decision)
    return {
        "week_start": comparison.baseline.week_start.isoformat(),
        "decisions": decisions,
        "product_forecasts": product_rows,
        "baseline_aggregates": {
            "category": _aggregate(comparison.baseline.rows, lambda row: (row.branch_id, row.category_id)),
            "branch": _aggregate(comparison.baseline.rows, lambda row: row.branch_id),
        },
        "policy_aggregates": {
            "category": _aggregate(comparison.policy.rows, lambda row: (row.branch_id, row.category_id)),
            "branch": _aggregate(comparison.policy.rows, lambda row: row.branch_id),
        },
        "paired_aggregates": {
            "category": _paired_aggregate(
                comparison.baseline.rows,
                comparison.policy.rows,
                lambda row: (row.branch_id, row.category_id),
            ),
            "branch": _paired_aggregate(
                comparison.baseline.rows,
                comparison.policy.rows,
                lambda row: row.branch_id,
            ),
        },
        "company_totals": {
            "baseline_predicted_revenue_usd": comparison.baseline.total_revenue,
            "policy_predicted_revenue_usd": comparison.policy.total_revenue,
            "delta_expected_revenue_usd": comparison.expected_revenue_change_usd,
            "baseline_predicted_gross_profit_usd": comparison.baseline.total_gross_profit,
            "policy_predicted_gross_profit_usd": comparison.policy.total_gross_profit,
            "delta_expected_gross_profit_usd": comparison.expected_gross_profit_change_usd,
            "action_implementation_costs_usd": comparison.implementation_costs_usd,
            "net_incremental_profit_usd": comparison.net_incremental_profit_usd,
            "risk": comparison.risk,
            "reward": comparison.reward,
        },
        "simulator_support": comparison.policy.simulator_support,
        "assumptions": list(comparison.policy.assumptions),
        "validation_status": (
            "synthetic_unverified_intervention_effects"
            if "synthetic" in comparison.policy.simulator_support
            else "action_conditioned_simulator"
        ),
        "provenance": {
            "decision_registry_path": str(registry.source_path),
            "decision_registry_sha256": registry.sha256,
            "policy_checkpoint": policy_checkpoint,
            "data_split_id": split_id,
            "simulator_version": simulator_version,
            "shared_scenario_id": comparison.shared_scenario_id,
            "shared_seed": comparison.shared_seed,
        },
        "limitations": list(limitations),
    }


def write_report_json(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def write_product_forecast_csv(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["branch", "category", "product"]
    fieldnames.extend(f"baseline_{name}" for name in _METRICS)
    fieldnames.extend(f"policy_{name}" for name in _METRICS)
    fieldnames.extend(f"delta_{name}" for name in _METRICS)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["product_forecasts"]:
            output = {name: row[name] for name in ("branch", "category", "product")}
            for prefix, source in (
                ("baseline", row["baseline"]),
                ("policy", row["with_actions"]),
                ("delta", row["delta"]),
            ):
                output.update({f"{prefix}_{name}": value for name, value in source.items()})
            writer.writerow(output)
    return destination
