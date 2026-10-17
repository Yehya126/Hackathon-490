"""Paired no-new-action comparison and weekly reward calculation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .schemas import BusinessState, CandidateAction, NextWeekForecast
from .simulator_contract import NextWeekSimulator


@dataclass(frozen=True)
class RewardConfig:
    scale: float
    risk_aversion_lambda: float = 0.0
    evidence_utility_weight: float = 0.0
    action_count_penalty: float = 0.0
    negative_net_profit_penalty_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Reward scale must be a positive fixed TRAIN-derived value")
        if not isfinite(self.risk_aversion_lambda) or self.risk_aversion_lambda < 0:
            raise ValueError("risk_aversion_lambda must be finite and nonnegative")
        if not isfinite(self.evidence_utility_weight) or self.evidence_utility_weight < 0:
            raise ValueError("evidence_utility_weight must be finite and nonnegative")
        if not isfinite(self.action_count_penalty) or self.action_count_penalty < 0:
            raise ValueError("action_count_penalty must be finite and nonnegative")
        if (
            not isfinite(self.negative_net_profit_penalty_multiplier)
            or self.negative_net_profit_penalty_multiplier < 1
        ):
            raise ValueError(
                "negative_net_profit_penalty_multiplier must be finite and at least 1"
            )


@dataclass(frozen=True)
class PortfolioComparison:
    baseline: NextWeekForecast
    policy: NextWeekForecast
    new_actions: tuple[CandidateAction, ...]
    implementation_costs_usd: float
    expected_revenue_change_usd: float
    expected_gross_profit_change_usd: float
    net_incremental_profit_usd: float
    risk: float
    evidence_utility: float
    reward: float
    shared_seed: int
    shared_scenario_id: str | None


def compare_portfolio(
    simulator: NextWeekSimulator,
    state: BusinessState,
    new_actions: tuple[CandidateAction, ...],
    exogenous_scenario: Any,
    seed: int,
    reward_config: RewardConfig,
    risk: float = 0.0,
) -> PortfolioComparison:
    if not isfinite(risk) or risk < 0:
        raise ValueError("Risk must be a calculable, finite, nonnegative value")
    existing = list(state.active_actions)
    baseline = simulator.simulate_next_week(state, existing, exogenous_scenario, seed)
    policy = simulator.simulate_next_week(state, existing + list(new_actions), exogenous_scenario, seed)
    if baseline.scenario_id != policy.scenario_id:
        raise ValueError("Paired forecasts must use the same exogenous scenario")
    implementation_costs = sum(action.implementation_cost_usd for action in new_actions)
    revenue_change = policy.total_revenue - baseline.total_revenue
    profit_change = policy.total_gross_profit - baseline.total_gross_profit
    net_profit = profit_change - implementation_costs
    evidence_utility = sum(
        float(action.evidence.get("learning_value_score", 0.0)) for action in new_actions
    )
    financial_reward = net_profit / reward_config.scale
    if net_profit < 0:
        financial_reward *= reward_config.negative_net_profit_penalty_multiplier
    reward = (
        financial_reward
        - reward_config.risk_aversion_lambda * risk
        + reward_config.evidence_utility_weight * evidence_utility
        - reward_config.action_count_penalty * len(new_actions)
    )
    return PortfolioComparison(
        baseline=baseline,
        policy=policy,
        new_actions=new_actions,
        implementation_costs_usd=implementation_costs,
        expected_revenue_change_usd=revenue_change,
        expected_gross_profit_change_usd=profit_change,
        net_incremental_profit_usd=net_profit,
        risk=risk,
        evidence_utility=evidence_utility,
        reward=reward,
        shared_seed=seed,
        shared_scenario_id=baseline.scenario_id,
    )
