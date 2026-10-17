"""Loading and validation for the editable top-level decision array."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import DecisionDefinition, ParameterDefinition


class RegistryValidationError(ValueError):
    pass


_REQUIRED_DECISION_FIELDS = {
    "id",
    "label",
    "enabled",
    "scopes",
    "parameters",
    "simulator_capability",
    "interaction_group",
    "executor",
}
_REQUIRED_PARAMETER_FIELDS = {"name", "dtype", "unit", "validation"}
_VALID_SCOPES = {"company", "branch", "category", "product"}


@dataclass(frozen=True)
class DecisionRegistry:
    decisions: tuple[DecisionDefinition, ...]
    source_path: Path
    sha256: str

    def enabled(self, *, include_stop: bool = True) -> tuple[DecisionDefinition, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.enabled and (include_stop or decision.id != "stop")
        )

    def get(self, decision_id: str) -> DecisionDefinition:
        for decision in self.decisions:
            if decision.id == decision_id:
                return decision
        raise KeyError(f"Unknown decision id: {decision_id}")


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{location} must be a non-empty string")
    return value


def _parse_parameter(raw: Any, location: str) -> ParameterDefinition:
    if not isinstance(raw, Mapping):
        raise RegistryValidationError(f"{location} must be an object")
    missing = sorted(_REQUIRED_PARAMETER_FIELDS - raw.keys())
    if missing:
        raise RegistryValidationError(f"{location} is missing fields: {missing}")
    name = _nonempty_string(raw["name"], f"{location}.name")
    dtype = _nonempty_string(raw["dtype"], f"{location}.dtype")
    if dtype != "int":
        raise RegistryValidationError(f"{location}.dtype must be 'int', got {dtype!r}")
    metadata = {k: v for k, v in raw.items() if k not in _REQUIRED_PARAMETER_FIELDS}
    return ParameterDefinition(
        name=name,
        dtype=dtype,
        unit=_nonempty_string(raw["unit"], f"{location}.unit"),
        validation=_nonempty_string(raw["validation"], f"{location}.validation"),
        metadata=metadata,
    )


def _parse_decision(raw: Any, index: int) -> DecisionDefinition:
    location = f"decisions[{index}]"
    if not isinstance(raw, Mapping):
        raise RegistryValidationError(f"{location} must be an object")
    missing = sorted(_REQUIRED_DECISION_FIELDS - raw.keys())
    if missing:
        raise RegistryValidationError(f"{location} is missing fields: {missing}")
    if not isinstance(raw["enabled"], bool):
        raise RegistryValidationError(f"{location}.enabled must be boolean")
    scopes = raw["scopes"]
    if not isinstance(scopes, list) or not scopes:
        raise RegistryValidationError(f"{location}.scopes must be a non-empty array")
    if any(scope not in _VALID_SCOPES for scope in scopes):
        raise RegistryValidationError(f"{location}.scopes contains an unsupported scope")
    if len(scopes) != len(set(scopes)):
        raise RegistryValidationError(f"{location}.scopes contains duplicates")
    raw_parameters = raw["parameters"]
    if not isinstance(raw_parameters, list):
        raise RegistryValidationError(f"{location}.parameters must be an array")
    parameters = tuple(
        _parse_parameter(parameter, f"{location}.parameters[{parameter_index}]")
        for parameter_index, parameter in enumerate(raw_parameters)
    )
    names = [parameter.name for parameter in parameters]
    if len(names) != len(set(names)):
        raise RegistryValidationError(f"{location}.parameters contains duplicate names")
    decision_id = _nonempty_string(raw["id"], f"{location}.id")
    if decision_id == "stop" and parameters:
        raise RegistryValidationError("The stop decision cannot have parameters")
    metadata = {k: v for k, v in raw.items() if k not in _REQUIRED_DECISION_FIELDS}
    return DecisionDefinition(
        id=decision_id,
        label=_nonempty_string(raw["label"], f"{location}.label"),
        enabled=raw["enabled"],
        scopes=tuple(scopes),
        parameters=parameters,
        simulator_capability=_nonempty_string(
            raw["simulator_capability"], f"{location}.simulator_capability"
        ),
        interaction_group=_nonempty_string(raw["interaction_group"], f"{location}.interaction_group"),
        executor=_nonempty_string(raw["executor"], f"{location}.executor"),
        metadata=metadata,
    )


def load_decisions(path: str | Path) -> DecisionRegistry:
    """Load a registry while preserving unknown metadata for schema evolution."""

    source = Path(path).resolve()
    try:
        payload_bytes = source.read_bytes()
    except OSError as error:
        raise RegistryValidationError(f"Cannot read decision registry {source}: {error}") from error
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise RegistryValidationError(f"Invalid JSON in {source}: {error}") from error
    if not isinstance(payload, list):
        raise RegistryValidationError("Decision registry must be a top-level JSON array")
    if not payload:
        raise RegistryValidationError("Decision registry cannot be empty")
    decisions = tuple(_parse_decision(raw, index) for index, raw in enumerate(payload))
    ids = [decision.id for decision in decisions]
    if len(ids) != len(set(ids)):
        raise RegistryValidationError("Decision ids must be unique")
    if "stop" not in ids:
        raise RegistryValidationError("Decision registry must include a stop decision")
    stop = next(decision for decision in decisions if decision.id == "stop")
    if not stop.enabled:
        raise RegistryValidationError("The stop decision must be enabled")
    return DecisionRegistry(
        decisions=decisions,
        source_path=source,
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )
