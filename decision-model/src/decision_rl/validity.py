"""Registry-driven feasibility and portfolio conflict checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .registry import DecisionRegistry
from .schemas import BusinessState, CandidateAction, DecisionDefinition, IntegerDomain, Target


@dataclass(frozen=True)
class FeasibilityResult:
    valid: bool
    reasons: tuple[str, ...] = ()


def target_matches_scope(scope: str, target: Target) -> bool:
    if scope == "company":
        return target.branch_id is None and target.category_id is None and target.product_id is None
    if scope == "branch":
        return target.branch_id is not None and target.category_id is None and target.product_id is None
    if scope == "category":
        return target.category_id is not None and target.product_id is None
    if scope == "product":
        return target.product_id is not None
    return False


def targets_overlap(left: Target, right: Target) -> bool:
    for left_value, right_value in (
        (left.branch_id, right.branch_id),
        (left.category_id, right.category_id),
        (left.product_id, right.product_id),
    ):
        if left_value is not None and right_value is not None and left_value != right_value:
            return False
    left_bundle = set(left.bundle_product_ids)
    right_bundle = set(right.bundle_product_ids)
    if left_bundle and right_bundle and left_bundle.isdisjoint(right_bundle):
        return False
    return True


def actions_conflict(
    left: CandidateAction,
    right: CandidateAction,
    registry: DecisionRegistry,
) -> bool:
    if not targets_overlap(left.target, right.target):
        return False
    left_definition = registry.get(left.decision_id)
    right_definition = registry.get(right.decision_id)
    if left_definition.interaction_group == "none" or right_definition.interaction_group == "none":
        return False
    return left_definition.interaction_group == right_definition.interaction_group


def _relevant_products(state: BusinessState, action: CandidateAction):
    for product in state.products:
        target = action.target
        if target.branch_id is not None and target.branch_id != product.branch_id:
            continue
        if target.category_id is not None and target.category_id != product.category_id:
            continue
        if target.product_id is not None and target.product_id != product.product_id:
            continue
        yield product


ValidationFunction = Callable[[int, CandidateAction, BusinessState], bool]


def _has_evidence_path(evidence: Mapping[str, object], path: str) -> bool:
    if path in evidence:
        return evidence[path] is not None
    current: object = evidence
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return current is not None


def _positive(value: int, _action: CandidateAction, _state: BusinessState) -> bool:
    return value > 0


def _between_zero_and_100(value: int, _action: CandidateAction, _state: BusinessState) -> bool:
    return 0 <= value <= 100


def _final_discount(value: int, action: CandidateAction, state: BusinessState) -> bool:
    products = tuple(_relevant_products(state, action))
    return bool(products) and all(0 <= product.discount_rate * 100 + value <= 100 for product in products)


def _positive_price(value: int, action: CandidateAction, state: BusinessState) -> bool:
    products = tuple(_relevant_products(state, action))
    return bool(products) and all(product.listed_price_usd * (1 + value / 100) > 0 for product in products)


def _final_preparation_nonnegative(value: int, action: CandidateAction, state: BusinessState) -> bool:
    products = tuple(_relevant_products(state, action))
    return bool(products) and all(
        product.prepared_units is not None and product.prepared_units + value >= 0 for product in products
    )


_VALIDATORS: dict[str, ValidationFunction] = {
    "positive": _positive,
    "between_0_and_100": _between_zero_and_100,
    "between_0_and_100_and_feasible": _between_zero_and_100,
    "final_discount_between_0_and_100": _final_discount,
    "resulting_price_must_be_positive_and_within_simulator_support": _positive_price,
    "final_preparation_nonnegative": _final_preparation_nonnegative,
}


class ActionValidator:
    """Validates definitions and values without embedding an action catalogue."""

    def __init__(
        self,
        registry: DecisionRegistry,
        supported_capabilities: Iterable[str],
        domains: Mapping[tuple[str, str], IntegerDomain],
        custom_validators: Mapping[str, ValidationFunction] | None = None,
    ) -> None:
        self.registry = registry
        self.supported_capabilities = frozenset(supported_capabilities) | {"none_required"}
        self.domains = dict(domains)
        self.validators = dict(_VALIDATORS)
        if custom_validators:
            self.validators.update(custom_validators)

    def validate(
        self,
        action: CandidateAction,
        state: BusinessState,
        portfolio: Iterable[CandidateAction] = (),
        domains: Mapping[tuple[str, str], IntegerDomain] | None = None,
    ) -> FeasibilityResult:
        reasons: list[str] = []
        try:
            definition = self.registry.get(action.decision_id)
        except KeyError:
            return FeasibilityResult(False, (f"unknown decision {action.decision_id!r}",))
        if not definition.enabled:
            reasons.append("decision is disabled")
        if action.scope not in definition.scopes:
            reasons.append(f"scope {action.scope!r} is not declared by the decision")
        elif not target_matches_scope(action.scope, action.target):
            reasons.append("target identifiers do not match the selected scope")
        if definition.simulator_capability not in self.supported_capabilities:
            reasons.append(f"simulator lacks capability {definition.simulator_capability!r}")
        for field_name in definition.metadata.get("required_target_fields", ()):
            if not isinstance(field_name, str) or not hasattr(action.target, field_name):
                reasons.append(f"registry declares unknown required target field {field_name!r}")
            elif not getattr(action.target, field_name):
                reasons.append(f"target is missing required field {field_name!r}")
        for evidence_name in definition.metadata.get("required_evidence", ()):
            if not isinstance(evidence_name, str):
                reasons.append("registry required_evidence entries must be strings")
            elif not _has_evidence_path(action.evidence, evidence_name):
                reasons.append(f"action is missing required evidence {evidence_name!r}")
        expected_names = {parameter.name for parameter in definition.parameters}
        actual_names = set(action.parameters)
        if expected_names != actual_names:
            reasons.append(
                f"parameter names must be exactly {sorted(expected_names)}, got {sorted(actual_names)}"
            )
        active_domains = self.domains if domains is None else domains
        for parameter in definition.parameters:
            if parameter.name not in action.parameters:
                continue
            value = action.parameters[parameter.name]
            if isinstance(value, bool) or not isinstance(value, int):
                reasons.append(f"parameter {parameter.name!r} must be an integer")
                continue
            domain = active_domains.get((definition.id, parameter.name))
            if domain is None:
                reasons.append(f"no credible simulator/operational domain for {parameter.name!r}")
            elif not domain.contains(value):
                reasons.append(
                    f"{parameter.name!r}={value} is outside supported domain "
                    f"[{domain.minimum}, {domain.maximum}] from {domain.source}"
                )
            validator = self.validators.get(parameter.validation)
            if validator is None:
                reasons.append(f"unknown validation rule {parameter.validation!r}")
            elif not validator(value, action, state):
                reasons.append(f"parameter {parameter.name!r} failed {parameter.validation!r}")
        selected = tuple(portfolio)
        for other in (*state.active_actions, *selected):
            if actions_conflict(action, other, self.registry):
                reasons.append(f"conflicts with {other.decision_id!r} on an overlapping target")
        total_cost = action.implementation_cost_usd + sum(item.implementation_cost_usd for item in selected)
        if state.remaining_action_budget_usd is not None and total_cost > state.remaining_action_budget_usd:
            reasons.append("portfolio exceeds remaining action budget")
        return FeasibilityResult(not reasons, tuple(reasons))

    def supports_definition(
        self,
        definition: DecisionDefinition,
        domains: Mapping[tuple[str, str], IntegerDomain] | None = None,
    ) -> bool:
        if definition.simulator_capability not in self.supported_capabilities:
            return False
        active_domains = self.domains if domains is None else domains
        return all((definition.id, parameter.name) in active_domains for parameter in definition.parameters)
