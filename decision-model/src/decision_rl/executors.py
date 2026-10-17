"""Extensible executor dispatch; action families do not enter PPO core logic."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .schemas import CandidateAction


ActionExecutor = Callable[[Any, CandidateAction], Any]


class ExecutorRegistry:
    def __init__(self, executors: Mapping[str, ActionExecutor] | None = None) -> None:
        self._executors = dict(executors or {})

    def register(self, name: str, executor: ActionExecutor) -> None:
        if not name.strip() or not callable(executor):
            raise ValueError("Executor name must be non-empty and executor callable")
        self._executors[name] = executor

    def require(self, name: str) -> ActionExecutor:
        try:
            return self._executors[name]
        except KeyError as error:
            raise KeyError(f"No executor is registered for {name!r}") from error

    def execute(self, name: str, simulator_state: Any, action: CandidateAction) -> Any:
        return self.require(name)(simulator_state, action)
