"""Clearly labeled synthetic action-response sandbox.

This module is for tests and demos only. Its response coefficients are explicit
assumptions, not estimates from Stories transactions.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from math import isfinite
from typing import Any, Callable, Iterable, Mapping

from .schemas import BusinessState, CandidateAction, ForecastRow, NextWeekForecast, ProductState


@dataclass(frozen=True)
class SyntheticResponseProfile:
    price_elasticity: float = -1.2
    promotion_units_lift_per_pct: float = 0.005
    bundle_uptake_rate: float = 0.06
    bundle_substitution_rate: float = 0.02
    cross_sell_acceptance_rate: float = 0.04
    cross_sell_cannibalization_rate: float = 0.01
    weekly_noise_stddev: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (
            self.price_elasticity,
            self.promotion_units_lift_per_pct,
            self.bundle_uptake_rate,
            self.bundle_substitution_rate,
            self.cross_sell_acceptance_rate,
            self.cross_sell_cannibalization_rate,
            self.weekly_noise_stddev,
        )):
            raise ValueError("Synthetic coefficients must be finite")
        if self.weekly_noise_stddev < 0:
            raise ValueError("weekly_noise_stddev must be nonnegative")
        for name in (
            "bundle_uptake_rate",
            "bundle_substitution_rate",
            "cross_sell_acceptance_rate",
            "cross_sell_cannibalization_rate",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass
class _MutableProduct:
    source: ProductState
    units: float
    list_price: float
    discount_rate: float
    unit_cost: float
    prepared_units: int | None


SyntheticExecutor = Callable[[_MutableProduct, CandidateAction, SyntheticResponseProfile], None]


def _price_change(row: _MutableProduct, action: CandidateAction, profile: SyntheticResponseProfile) -> None:
    ratio = 1 + action.parameters["change_pct"] / 100
    if ratio <= 0:
        raise ValueError("Synthetic price change would make price nonpositive")
    row.list_price *= ratio
    row.units *= ratio ** profile.price_elasticity


def _discount_change(row: _MutableProduct, action: CandidateAction, profile: SyntheticResponseProfile) -> None:
    old_effective = row.list_price * (1 - row.discount_rate)
    row.discount_rate += action.parameters["change_percentage_points"] / 100
    if not 0 <= row.discount_rate <= 1:
        raise ValueError("Synthetic discount change would leave the valid rate domain")
    new_effective = row.list_price * (1 - row.discount_rate)
    if old_effective > 0 and new_effective > 0:
        row.units *= (new_effective / old_effective) ** profile.price_elasticity


def _promotion(row: _MutableProduct, action: CandidateAction, profile: SyntheticResponseProfile) -> None:
    row.units *= 1 + action.parameters["intensity_pct"] * profile.promotion_units_lift_per_pct


def _procurement(row: _MutableProduct, action: CandidateAction, _profile: SyntheticResponseProfile) -> None:
    row.unit_cost *= 1 - action.parameters["target_cost_reduction_pct"] / 100


def _preparation(row: _MutableProduct, action: CandidateAction, _profile: SyntheticResponseProfile) -> None:
    if row.prepared_units is None:
        raise ValueError("Synthetic preparation action requires prepared_units")
    row.prepared_units += action.parameters["units_delta"]
    if row.prepared_units < 0:
        raise ValueError("Prepared units cannot be negative")
    row.units = min(row.units, float(row.prepared_units))


def _bundle_create(row: _MutableProduct, _action: CandidateAction, profile: SyntheticResponseProfile) -> None:
    row.units *= max(0.0, 1 + profile.bundle_uptake_rate - profile.bundle_substitution_rate)


def _cross_sell(row: _MutableProduct, action: CandidateAction, profile: SyntheticResponseProfile) -> None:
    consequent = action.evidence.get("consequent_product")
    if consequent == row.source.product_id:
        row.units *= 1 + profile.cross_sell_acceptance_rate
    else:
        row.units *= 1 - profile.cross_sell_cannibalization_rate


def _review_only(_row: _MutableProduct, _action: CandidateAction, _profile: SyntheticResponseProfile) -> None:
    return None


_DEFAULT_EXECUTORS: Mapping[str, SyntheticExecutor] = {
    "price_change": _price_change,
    "discount_change": _discount_change,
    "promotion": _promotion,
    "procurement_scenario": _procurement,
    "preparation_change": _preparation,
    "bundle_create": _bundle_create,
    "cross_sell": _cross_sell,
    "branch_price_variance_review": _review_only,
    "branch_cost_variance_review": _review_only,
}

_CAPABILITIES_BY_EXECUTOR = {
    "price_change": "action_conditioned_price_and_demand",
    "discount_change": "action_conditioned_discount_and_demand",
    "promotion": "promotion_response_and_cost",
    "procurement_scenario": "validated_operational_cost_intervention",
    "preparation_change": "inventory_waste_and_stockout",
    "bundle_create": "bundle_uptake_substitution_and_price_response",
    "cross_sell": "cross_sell_response_and_cannibalization",
    "branch_price_variance_review": "branch_variance_review_workflow",
    "branch_cost_variance_review": "branch_variance_review_workflow",
}


class SyntheticSimulator:
    """A deterministic-under-seed sandbox, never a real-effect estimator."""

    label = "synthetic_only_unvalidated_intervention_effects"

    def __init__(
        self,
        profile: SyntheticResponseProfile | None = None,
        executors: Mapping[str, SyntheticExecutor] | None = None,
    ) -> None:
        self.profile = profile or SyntheticResponseProfile()
        self.executors = dict(_DEFAULT_EXECUTORS if executors is None else executors)
        self._supported_capabilities = frozenset(
            capability
            for executor_name, capability in _CAPABILITIES_BY_EXECUTOR.items()
            if executor_name in self.executors
        )

    @property
    def supported_capabilities(self) -> frozenset[str]:
        return self._supported_capabilities

    @staticmethod
    def _matches(product: ProductState, action: CandidateAction) -> bool:
        target = action.target
        if target.branch_id is not None and target.branch_id != product.branch_id:
            return False
        if target.category_id is not None and target.category_id != product.category_id:
            return False
        if target.product_id is not None and target.product_id != product.product_id:
            return False
        if target.bundle_product_ids and product.product_id not in target.bundle_product_ids:
            return False
        return True

    def simulate_next_week(
        self,
        current_state: BusinessState,
        actions: list[CandidateAction],
        exogenous_scenario: Any,
        seed: int,
    ) -> NextWeekForecast:
        rng = random.Random(seed)
        scenario_id = None
        scenario_multiplier = 1.0
        if isinstance(exogenous_scenario, Mapping):
            scenario_id = str(exogenous_scenario.get("id", "synthetic-default"))
            scenario_multiplier = float(exogenous_scenario.get("units_multiplier", 1.0))
        rows: list[ForecastRow] = []
        for source in current_state.products:
            random_multiplier = max(0.0, rng.gauss(1.0, self.profile.weekly_noise_stddev))
            row = _MutableProduct(
                source=source,
                units=source.expected_units * scenario_multiplier * random_multiplier,
                list_price=source.listed_price_usd,
                discount_rate=source.discount_rate,
                unit_cost=source.unit_cost_usd,
                prepared_units=source.prepared_units,
            )
            for action in actions:
                if not self._matches(source, action):
                    continue
                try:
                    executor = self.executors[action.decision_id]
                except KeyError as error:
                    raise ValueError(
                        f"Synthetic simulator does not support action {action.decision_id!r}"
                    ) from error
                executor(row, action, self.profile)
            effective_price = row.list_price * (1 - row.discount_rate)
            revenue = row.units * effective_price
            cogs = row.units * row.unit_cost
            rows.append(
                ForecastRow(
                    branch_id=source.branch_id,
                    product_id=source.product_id,
                    category_id=source.category_id,
                    predicted_units=row.units,
                    predicted_list_price_usd=row.list_price,
                    predicted_effective_price_usd=effective_price,
                    predicted_unit_cost_usd=row.unit_cost,
                    predicted_revenue_usd=revenue,
                    predicted_cogs_usd=cogs,
                    predicted_gross_profit_usd=revenue - cogs,
                    uncertainty={"synthetic_noise_stddev": self.profile.weekly_noise_stddev},
                )
            )
        return NextWeekForecast(
            week_start=current_state.week_start + timedelta(weeks=1),
            rows=tuple(rows),
            simulator_support=self.label,
            assumptions=(
                "Synthetic response coefficients are explicit test assumptions, not learned causal effects.",
                "Sales are treated as demand except where preparation caps synthetic units.",
            ),
            scenario_id=scenario_id,
            metadata={"seed": seed, "synthetic_response_profile": asdict(self.profile)},
        )

    def advance_state(
        self,
        current_state: BusinessState,
        forecast: NextWeekForecast,
        new_actions: Iterable[CandidateAction],
    ) -> BusinessState:
        # Keep the sandbox's counterfactual base rows unchanged. Active actions
        # are reapplied to that base in each covered week, avoiding accidental
        # compounding of percentage changes. A production simulator should
        # provide its own transition when business dynamics are available.
        new_actions = tuple(new_actions)
        continuing: list[CandidateAction] = []
        for action in (*current_state.active_actions, *new_actions):
            remaining = action.parameters.get("duration_weeks")
            if remaining is not None and remaining > 1:
                parameters = dict(action.parameters)
                parameters["duration_weeks"] = remaining - 1
                continuing.append(replace(action, parameters=parameters))
        budget = current_state.remaining_action_budget_usd
        if budget is not None:
            budget -= sum(action.implementation_cost_usd for action in new_actions)
        return replace(
            current_state,
            week_start=forecast.week_start,
            active_actions=tuple(continuing),
            remaining_action_budget_usd=budget,
        )
