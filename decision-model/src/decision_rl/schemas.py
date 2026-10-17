"""Typed, file-format-independent contracts used by the decision engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from math import isfinite
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised when an input violates a public decision-engine contract."""


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    dtype: str
    unit: str
    validation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionDefinition:
    id: str
    label: str
    enabled: bool
    scopes: tuple[str, ...]
    parameters: tuple[ParameterDefinition, ...]
    simulator_capability: str
    interaction_group: str
    executor: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class Target:
    """Categorical target identifiers. None means the dimension is not targeted."""

    branch_id: str | None = None
    category_id: str | None = None
    product_id: str | None = None
    bundle_product_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.branch_id is not None:
            result["branch"] = self.branch_id
        if self.category_id is not None:
            result["category"] = self.category_id
        if self.product_id is not None:
            result["product"] = self.product_id
        if self.bundle_product_ids:
            result["products"] = list(self.bundle_product_ids)
        return result


@dataclass(frozen=True)
class CandidateAction:
    decision_id: str
    scope: str
    target: Target
    parameters: Mapping[str, int]
    implementation_cost_usd: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.parameters.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(f"Action parameter {name!r} must be an integer")
        if not isfinite(self.implementation_cost_usd) or self.implementation_cost_usd < 0:
            raise ContractError("implementation_cost_usd must be finite and nonnegative")

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.decision_id,
            self.scope,
            self.target,
            tuple(sorted(self.parameters.items())),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.decision_id,
            "scope": self.scope,
            "target": self.target.as_dict(),
            "parameters": dict(self.parameters),
            "implementation_cost_usd": self.implementation_cost_usd,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class IntegerDomain:
    """A credible inclusive range supplied by a simulator or operating policy.

    This is a validity/support range, not a preset grid of policy choices.
    """

    minimum: int
    maximum: int
    source: str

    def __post_init__(self) -> None:
        if isinstance(self.minimum, bool) or isinstance(self.maximum, bool):
            raise ContractError("Integer domain bounds must be integers")
        if not isinstance(self.minimum, int) or not isinstance(self.maximum, int):
            raise ContractError("Integer domain bounds must be integers")
        if self.minimum > self.maximum:
            raise ContractError("Integer domain minimum cannot exceed maximum")
        if not self.source.strip():
            raise ContractError("Integer domain must record its support source")

    def contains(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class ProductState:
    branch_id: str
    product_id: str
    category_id: str
    expected_units: float
    listed_price_usd: float
    unit_cost_usd: float
    discount_rate: float = 0.0
    prepared_units: int | None = None
    features: Mapping[str, float] = field(default_factory=dict)
    uncertainty: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (self.expected_units, self.listed_price_usd, self.unit_cost_usd, self.discount_rate)
        if not all(isfinite(x) for x in numeric):
            raise ContractError("Product-state financial values must be finite")
        if self.expected_units < 0 or self.listed_price_usd <= 0 or self.unit_cost_usd < 0:
            raise ContractError("Units/cost must be nonnegative and listed price must be positive")
        if not 0 <= self.discount_rate <= 1:
            raise ContractError("discount_rate must be between 0 and 1")


@dataclass(frozen=True)
class BusinessState:
    week_start: date
    products: tuple[ProductState, ...]
    active_actions: tuple[CandidateAction, ...] = ()
    company_features: Mapping[str, float] = field(default_factory=dict)
    remaining_action_budget_usd: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def product(self, branch_id: str, product_id: str) -> ProductState:
        matches = [p for p in self.products if p.branch_id == branch_id and p.product_id == product_id]
        if len(matches) != 1:
            raise ContractError(
                f"Expected exactly one product row for branch={branch_id!r}, product={product_id!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class ForecastRow:
    branch_id: str
    product_id: str
    category_id: str
    predicted_units: float
    predicted_list_price_usd: float
    predicted_effective_price_usd: float
    predicted_unit_cost_usd: float
    predicted_revenue_usd: float
    predicted_cogs_usd: float
    predicted_gross_profit_usd: float
    uncertainty: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (
            self.predicted_units,
            self.predicted_list_price_usd,
            self.predicted_effective_price_usd,
            self.predicted_unit_cost_usd,
            self.predicted_revenue_usd,
            self.predicted_cogs_usd,
            self.predicted_gross_profit_usd,
        )
        if not all(isfinite(x) for x in values):
            raise ContractError("Forecast values must be finite")
        if self.predicted_units < 0 or self.predicted_list_price_usd <= 0:
            raise ContractError("Forecast units must be nonnegative and listed price positive")
        if self.predicted_effective_price_usd < 0 or self.predicted_unit_cost_usd < 0:
            raise ContractError("Effective price and unit cost must be nonnegative")

    def assert_reconciled(self, tolerance: float = 1e-8) -> None:
        expected_revenue = self.predicted_units * self.predicted_effective_price_usd
        expected_cogs = self.predicted_units * self.predicted_unit_cost_usd
        expected_profit = expected_revenue - expected_cogs
        for label, actual, expected in (
            ("revenue", self.predicted_revenue_usd, expected_revenue),
            ("COGS", self.predicted_cogs_usd, expected_cogs),
            ("gross profit", self.predicted_gross_profit_usd, expected_profit),
        ):
            if abs(actual - expected) > tolerance * max(1.0, abs(expected)):
                raise ContractError(f"Forecast {label} does not reconcile for {self.branch_id}/{self.product_id}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NextWeekForecast:
    week_start: date
    rows: tuple[ForecastRow, ...]
    simulator_support: str
    assumptions: tuple[str, ...] = ()
    scenario_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = [(row.branch_id, row.product_id) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ContractError("Forecast must contain at most one row per branch/product")
        for row in self.rows:
            row.assert_reconciled()

    @property
    def total_revenue(self) -> float:
        return sum(row.predicted_revenue_usd for row in self.rows)

    @property
    def total_cogs(self) -> float:
        return sum(row.predicted_cogs_usd for row in self.rows)

    @property
    def total_gross_profit(self) -> float:
        return sum(row.predicted_gross_profit_usd for row in self.rows)

    def keyed_rows(self) -> dict[tuple[str, str], ForecastRow]:
        return {(row.branch_id, row.product_id): row for row in self.rows}
