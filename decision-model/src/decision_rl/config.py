"""General configuration loader kept separate from the decision registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionModelConfig:
    frequency: str
    planning_horizon_weeks: int
    max_new_actions_per_week: int
    minimum_actions_before_stop: int
    candidate_proposals_per_step: int
    discount_factor_gamma: float
    reward_objective: str
    risk_aversion_lambda: float
    registry_file: Path
    require_human_approval: bool
    require_positive_marginal_net_profit: bool
    minimum_marginal_net_profit_usd: float
    raw: Mapping[str, Any]


def load_config(path: str | Path) -> DecisionModelConfig:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Cannot load configuration {source}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError("General configuration must be a JSON object")
    required = {
        "frequency",
        "planning_horizon_weeks",
        "max_new_actions_per_week",
        "candidate_proposals_per_step",
        "discount_factor_gamma",
        "reward_objective",
        "risk_aversion_lambda",
        "registry_file",
        "constraints",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ConfigurationError(f"Configuration is missing fields: {missing}")
    for name in ("planning_horizon_weeks", "max_new_actions_per_week", "candidate_proposals_per_step"):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int) or raw[name] < 1:
            raise ConfigurationError(f"{name} must be a positive integer")
    minimum_actions = raw.get("minimum_actions_before_stop", 0)
    if isinstance(minimum_actions, bool) or not isinstance(minimum_actions, int) or minimum_actions < 0:
        raise ConfigurationError("minimum_actions_before_stop must be a nonnegative integer")
    if minimum_actions > raw["max_new_actions_per_week"]:
        raise ConfigurationError(
            "minimum_actions_before_stop cannot exceed max_new_actions_per_week"
        )
    constraints = raw["constraints"]
    if not isinstance(constraints, dict) or not isinstance(constraints.get("require_human_approval"), bool):
        raise ConfigurationError("constraints.require_human_approval must be boolean")
    require_positive_profit = constraints.get("require_positive_marginal_net_profit", False)
    if not isinstance(require_positive_profit, bool):
        raise ConfigurationError(
            "constraints.require_positive_marginal_net_profit must be boolean"
        )
    minimum_marginal_profit = constraints.get("minimum_marginal_net_profit_usd", 0.0)
    if (
        isinstance(minimum_marginal_profit, bool)
        or not isinstance(minimum_marginal_profit, (int, float))
        or minimum_marginal_profit < 0
    ):
        raise ConfigurationError(
            "constraints.minimum_marginal_net_profit_usd must be nonnegative"
        )
    registry_path = Path(raw["registry_file"])
    if not registry_path.is_absolute():
        registry_path = source.parent.parent / registry_path
    return DecisionModelConfig(
        frequency=str(raw["frequency"]),
        planning_horizon_weeks=raw["planning_horizon_weeks"],
        max_new_actions_per_week=raw["max_new_actions_per_week"],
        minimum_actions_before_stop=minimum_actions,
        candidate_proposals_per_step=raw["candidate_proposals_per_step"],
        discount_factor_gamma=float(raw["discount_factor_gamma"]),
        reward_objective=str(raw["reward_objective"]),
        risk_aversion_lambda=float(raw["risk_aversion_lambda"]),
        registry_file=registry_path.resolve(),
        require_human_approval=constraints["require_human_approval"],
        require_positive_marginal_net_profit=require_positive_profit,
        minimum_marginal_net_profit_usd=float(minimum_marginal_profit),
        raw=raw,
    )
