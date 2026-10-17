"""Adapter for a future action-conditioned production simulator.

The adapter is intentionally callable-based: choosing the eventual files and
models is an integration concern, not a PPO/environment concern.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .schemas import BusinessState, CandidateAction, NextWeekForecast


SimulationFunction = Callable[[BusinessState, list[CandidateAction], Any, int], NextWeekForecast]


class RealSimulatorAdapter:
    def __init__(
        self,
        simulation_function: SimulationFunction,
        supported_capabilities: Iterable[str],
        version: str,
    ) -> None:
        if not callable(simulation_function):
            raise TypeError("simulation_function must be callable")
        if not version.strip():
            raise ValueError("Simulator version is required for reproducibility")
        self._simulation_function = simulation_function
        self._supported_capabilities = frozenset(supported_capabilities)
        self.version = version

    @property
    def supported_capabilities(self) -> frozenset[str]:
        return self._supported_capabilities

    def simulate_next_week(
        self,
        current_state: BusinessState,
        actions: list[CandidateAction],
        exogenous_scenario: Any,
        seed: int,
    ) -> NextWeekForecast:
        forecast = self._simulation_function(current_state, actions, exogenous_scenario, seed)
        if not isinstance(forecast, NextWeekForecast):
            raise TypeError("Production simulator must return NextWeekForecast")
        expected_keys = {(row.branch_id, row.product_id) for row in current_state.products}
        if set(forecast.keyed_rows()) != expected_keys:
            raise ValueError("Production simulator forecast does not cover exactly the current branch/product rows")
        return forecast
