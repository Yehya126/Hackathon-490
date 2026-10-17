"""Fixed-size, padded observations for a dynamic candidate list."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .candidate_proposals import CandidatePool, ProposalContext
from .schemas import BusinessState, CandidateAction


def _stable_identifier(value: str | None) -> float:
    if value is None:
        return 0.0
    number = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
    return (number / (2**64 - 1)) * 2 - 1


@dataclass(frozen=True)
class NormalizationStat:
    mean: float
    scale: float

    def apply(self, value: float) -> float:
        return (value - self.mean) / self.scale if self.scale > 0 else value - self.mean


@dataclass(frozen=True)
class ObservationSchema:
    max_products: int
    max_candidates: int
    max_parameters_per_action: int
    company_feature_names: tuple[str, ...] = ()
    product_feature_names: tuple[str, ...] = ()
    candidate_evidence_names: tuple[str, ...] = ()
    normalization: Mapping[str, NormalizationStat] = field(default_factory=dict)
    normalization_provenance: str = "unscaled"

    def __post_init__(self) -> None:
        if min(self.max_products, self.max_candidates, self.max_parameters_per_action) < 1:
            raise ValueError("Observation maxima must be positive")
        if self.normalization and not self.normalization_provenance.startswith("train"):
            raise ValueError("Observation normalization must be fitted on TRAIN only")


class ObservationEncoder:
    """Encodes provided numerical inputs; it never reads or fits source data."""

    BASE_PRODUCT_FEATURES = (
        "expected_units",
        "listed_price_usd",
        "unit_cost_usd",
        "discount_rate",
        "prepared_units",
    )

    def __init__(self, schema: ObservationSchema) -> None:
        self.schema = schema
        self.product_width = 3 + len(self.BASE_PRODUCT_FEATURES) + len(schema.product_feature_names)
        self.state_width = (
            len(schema.company_feature_names)
            + schema.max_products * self.product_width
            + 2
        )
        self.candidate_width = 6 + schema.max_parameters_per_action + len(schema.candidate_evidence_names)

    def _numeric(self, feature_name: str, value: float) -> float:
        stat = self.schema.normalization.get(feature_name)
        return float(stat.apply(value) if stat else value)

    def _encode_product(self, product) -> list[float]:
        values = [
            _stable_identifier(product.branch_id),
            _stable_identifier(product.category_id),
            _stable_identifier(product.product_id),
            self._numeric("expected_units", product.expected_units),
            self._numeric("listed_price_usd", product.listed_price_usd),
            self._numeric("unit_cost_usd", product.unit_cost_usd),
            self._numeric("discount_rate", product.discount_rate),
            self._numeric("prepared_units", float(product.prepared_units or 0)),
        ]
        values.extend(
            self._numeric(name, float(product.features.get(name, 0.0)))
            for name in self.schema.product_feature_names
        )
        return values

    def _encode_candidate(self, action: CandidateAction, context: ProposalContext) -> list[float]:
        row = [
            _stable_identifier(action.decision_id),
            _stable_identifier(action.scope),
            _stable_identifier(action.target.branch_id),
            _stable_identifier(action.target.category_id),
            _stable_identifier(action.target.product_id),
            float(action.implementation_cost_usd),
        ]
        parameter_values: list[float] = []
        for name, value in sorted(action.parameters.items()):
            domain = context.integer_domains.get((action.decision_id, name))
            if domain is None or domain.maximum == domain.minimum:
                parameter_values.append(float(value))
            else:
                parameter_values.append(
                    2 * (value - domain.minimum) / (domain.maximum - domain.minimum) - 1
                )
        parameter_values = parameter_values[: self.schema.max_parameters_per_action]
        row.extend(parameter_values)
        row.extend([0.0] * (self.schema.max_parameters_per_action - len(parameter_values)))
        row.extend(float(action.evidence.get(name, 0.0)) for name in self.schema.candidate_evidence_names)
        return row

    def encode(
        self,
        state: BusinessState,
        pool: CandidatePool,
        context: ProposalContext,
        selected_count: int,
    ) -> dict[str, np.ndarray]:
        if len(state.products) > self.schema.max_products:
            raise ValueError("State has more products than observation max_products")
        if len(pool.candidates) > self.schema.max_candidates:
            raise ValueError("Candidate pool exceeds observation max_candidates")
        state_values = [
            self._numeric(name, float(state.company_features.get(name, 0.0)))
            for name in self.schema.company_feature_names
        ]
        for product in sorted(state.products, key=lambda item: (item.branch_id, item.product_id)):
            state_values.extend(self._encode_product(product))
        missing_products = self.schema.max_products - len(state.products)
        state_values.extend([0.0] * missing_products * self.product_width)
        budget = state.remaining_action_budget_usd
        state_values.extend([float(selected_count), -1.0 if budget is None else float(budget)])
        candidate_rows = [self._encode_candidate(action, context) for action in pool.candidates]
        candidate_rows.extend(
            [[0.0] * self.candidate_width for _ in range(self.schema.max_candidates - len(candidate_rows))]
        )
        candidate_mask = [1.0] * len(pool.candidates) + [0.0] * (
            self.schema.max_candidates - len(pool.candidates)
        )
        return {
            "state": np.asarray(state_values, dtype=np.float32),
            "candidates": np.asarray(candidate_rows, dtype=np.float32),
            "candidate_mask": np.asarray(candidate_mask, dtype=np.float32),
        }
