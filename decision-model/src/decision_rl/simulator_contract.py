"""Simulator protocol and capability audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .schemas import BusinessState, CandidateAction, NextWeekForecast


class SimulatorCapabilityError(RuntimeError):
    pass


@runtime_checkable
class NextWeekSimulator(Protocol):
    @property
    def supported_capabilities(self) -> frozenset[str]: ...

    def simulate_next_week(
        self,
        current_state: BusinessState,
        actions: list[CandidateAction],
        exogenous_scenario: Any,
        seed: int,
    ) -> NextWeekForecast: ...


@dataclass(frozen=True)
class CapabilityAudit:
    capability: str
    passed: bool
    changed_units: bool
    changed_prices: bool
    message: str


def audit_action_response(
    simulator: NextWeekSimulator,
    state: BusinessState,
    action: CandidateAction,
    capability: str,
    exogenous_scenario: Any = None,
    seed: int = 0,
    tolerance: float = 1e-10,
) -> CapabilityAudit:
    """Check that a declared intervention produces the outcomes it promises."""

    baseline = simulator.simulate_next_week(state, [], exogenous_scenario, seed)
    changed = simulator.simulate_next_week(state, [action], exogenous_scenario, seed)
    baseline_rows = baseline.keyed_rows()
    changed_rows = changed.keyed_rows()
    if baseline_rows.keys() != changed_rows.keys():
        return CapabilityAudit(capability, False, False, False, "Action changed forecast row identity")
    changed_units = any(
        abs(changed_rows[key].predicted_units - row.predicted_units) > tolerance
        for key, row in baseline_rows.items()
    )
    changed_prices = any(
        abs(changed_rows[key].predicted_effective_price_usd - row.predicted_effective_price_usd) > tolerance
        for key, row in baseline_rows.items()
    )
    if capability == "action_conditioned_price_and_demand":
        passed = changed_units and changed_prices
        message = "price and demand both respond" if passed else "price actions must change both price and units"
    else:
        passed = changed_units or changed_prices or (
            abs(changed.total_gross_profit - baseline.total_gross_profit) > tolerance
        )
        message = "forecast responds to action" if passed else "action has no forecast effect"
    return CapabilityAudit(capability, passed, changed_units, changed_prices, message)


def require_capability_audit(audit: CapabilityAudit) -> None:
    if not audit.passed:
        raise SimulatorCapabilityError(f"Capability {audit.capability!r} failed: {audit.message}")
