"""Dynamic integer candidate proposals for a finite MaskablePPO action pool."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from typing import Mapping

from .registry import DecisionRegistry
from .schemas import BusinessState, CandidateAction, IntegerDomain, Target
from .validity import ActionValidator


@dataclass(frozen=True)
class ProposalContext:
    """Targets and support ranges supplied by upstream systems at decision time."""

    targets_by_scope: Mapping[str, tuple[Target, ...]]
    integer_domains: Mapping[tuple[str, str], IntegerDomain]
    local_centers: Mapping[tuple[str, str, Target], int] = field(default_factory=dict)
    sourced_parameter_values: Mapping[tuple[str, str, Target], tuple[int, ...]] = field(
        default_factory=dict
    )
    evidence_by_target: Mapping[tuple[str, str, Target], Mapping[str, object]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ProposalCoverage:
    decision_id: str
    parameter_name: str
    supported_minimum: int
    supported_maximum: int
    proposed_minimum: int
    proposed_maximum: int
    proposed_unique_values: int
    domain_source: str


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[CandidateAction, ...]
    coverage: tuple[ProposalCoverage, ...]
    rejected_count: int


def _integer_samples(domain: IntegerDomain, count: int, rng: random.Random, center: int | None) -> list[int]:
    """Stratified samples across an inclusive range plus local refinement.

    No literal parameter grid is used; every value is derived from the supplied
    support interval and deterministic random seed.
    """

    width = domain.maximum - domain.minimum + 1
    if count >= width:
        values = list(range(domain.minimum, domain.maximum + 1))
        rng.shuffle(values)
        return values
    values: set[int] = set()
    strata = max(1, count // 2)
    for stratum in range(strata):
        lower = domain.minimum + (width * stratum) // strata
        upper = domain.minimum + (width * (stratum + 1)) // strata - 1
        values.add(rng.randint(lower, max(lower, upper)))
    focus = center if center is not None and domain.contains(center) else rng.randint(domain.minimum, domain.maximum)
    radius = 0
    while len(values) < count:
        for candidate in (focus - radius, focus + radius):
            if domain.contains(candidate):
                values.add(candidate)
                if len(values) >= count:
                    break
        radius += 1
        if radius > width:
            break
    return sorted(values)


class CandidateGenerator:
    def __init__(
        self,
        registry: DecisionRegistry,
        validator: ActionValidator,
        proposals_per_step: int,
        seed: int = 42,
    ) -> None:
        if proposals_per_step < 1:
            raise ValueError("proposals_per_step must be positive")
        self.registry = registry
        self.validator = validator
        self.proposals_per_step = proposals_per_step
        self.seed = seed

    def generate(
        self,
        state: BusinessState,
        context: ProposalContext,
        portfolio: tuple[CandidateAction, ...] = (),
        step_number: int = 0,
    ) -> CandidatePool:
        seed_material = f"{self.seed}|{state.week_start.isoformat()}|{step_number}".encode()
        step_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(step_seed)
        streams: list[list[CandidateAction]] = []
        proposed_values: dict[tuple[str, str], set[int]] = {}
        rejected = 0
        definitions = [
            item
            for item in self.registry.enabled(include_stop=False)
            if self.validator.supports_definition(item, context.integer_domains)
        ]
        template_count = max(
            1,
            sum(len(context.targets_by_scope.get(scope, ())) for item in definitions for scope in item.scopes),
        )
        values_per_parameter = max(2, self.proposals_per_step // template_count)
        for definition in definitions:
            for scope in definition.scopes:
                for target in context.targets_by_scope.get(scope, ()):
                    samples_by_parameter: list[tuple[str, list[int]]] = []
                    source_values_missing = False
                    for parameter in definition.parameters:
                        domain = context.integer_domains[(definition.id, parameter.name)]
                        source_key = (definition.id, parameter.name, target)
                        if "value_source" in parameter.metadata:
                            values = sorted(set(context.sourced_parameter_values.get(source_key, ())))
                            if not values:
                                source_values_missing = True
                                break
                            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                                raise ValueError(f"Sourced values for {source_key!r} must all be integers")
                            values = [value for value in values if domain.contains(value)]
                            if not values:
                                source_values_missing = True
                                break
                        else:
                            center = context.local_centers.get(source_key)
                            values = _integer_samples(domain, values_per_parameter, rng, center)
                        samples_by_parameter.append((parameter.name, values))
                        proposed_values.setdefault((definition.id, parameter.name), set()).update(values)
                    if source_values_missing:
                        rejected += 1
                        continue
                    combinations = itertools.product(*(values for _, values in samples_by_parameter))
                    stream: list[CandidateAction] = []
                    for combination in combinations:
                        action = CandidateAction(
                            decision_id=definition.id,
                            scope=scope,
                            target=target,
                            parameters={
                                name: value
                                for (name, _values), value in zip(samples_by_parameter, combination)
                            },
                            evidence=dict(
                                context.evidence_by_target.get((definition.id, scope, target), {})
                            ),
                        )
                        if self.validator.validate(action, state, portfolio, context.integer_domains).valid:
                            stream.append(action)
                        else:
                            rejected += 1
                    rng.shuffle(stream)
                    if stream:
                        streams.append(stream)
        rng.shuffle(streams)
        chosen: list[CandidateAction] = []
        while streams and len(chosen) < self.proposals_per_step:
            next_streams: list[list[CandidateAction]] = []
            for stream in streams:
                if stream and len(chosen) < self.proposals_per_step:
                    chosen.append(stream.pop())
                if stream:
                    next_streams.append(stream)
            streams = next_streams
        chosen.sort(key=CandidateAction.sort_key)
        coverage: list[ProposalCoverage] = []
        for (decision_id, parameter_name), values in sorted(proposed_values.items()):
            if not values:
                continue
            domain = context.integer_domains[(decision_id, parameter_name)]
            coverage.append(
                ProposalCoverage(
                    decision_id=decision_id,
                    parameter_name=parameter_name,
                    supported_minimum=domain.minimum,
                    supported_maximum=domain.maximum,
                    proposed_minimum=min(values),
                    proposed_maximum=max(values),
                    proposed_unique_values=len(values),
                    domain_source=domain.source,
                )
            )
        return CandidatePool(tuple(chosen), tuple(coverage), rejected)
