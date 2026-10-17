"""In-memory adapters for numerical outputs from specialist components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

from .schemas import BusinessState, CandidateAction, ProductState


@dataclass(frozen=True)
class SpecialistProductInput:
    branch_id: str
    product_id: str
    category_id: str
    expected_units: float
    listed_price_usd: float
    unit_cost_usd: float
    discount_rate: float = 0.0
    prepared_units: int | None = None
    signals: Mapping[str, float] = field(default_factory=dict)
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


class BusinessStateAssembler:
    """Creates a state after callers decide where each input comes from.

    This class deliberately accepts records rather than opening repository files.
    """

    def assemble(
        self,
        week_start: date,
        product_inputs: Iterable[SpecialistProductInput],
        company_features: Mapping[str, float] | None = None,
        active_actions: Iterable[CandidateAction] = (),
        remaining_action_budget_usd: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> BusinessState:
        products = tuple(
            ProductState(
                branch_id=item.branch_id,
                product_id=item.product_id,
                category_id=item.category_id,
                expected_units=item.expected_units,
                listed_price_usd=item.listed_price_usd,
                unit_cost_usd=item.unit_cost_usd,
                discount_rate=item.discount_rate,
                prepared_units=item.prepared_units,
                features=dict(item.signals),
                uncertainty=dict(item.uncertainty),
            )
            for item in product_inputs
        )
        keys = [(item.branch_id, item.product_id) for item in products]
        if len(keys) != len(set(keys)):
            raise ValueError("Specialist inputs contain duplicate branch/product rows")
        if not products:
            raise ValueError("At least one branch/product input is required")
        return BusinessState(
            week_start=week_start,
            products=products,
            active_actions=tuple(active_actions),
            company_features=dict(company_features or {}),
            remaining_action_budget_usd=remaining_action_budget_usd,
            provenance=dict(provenance or {}),
        )
