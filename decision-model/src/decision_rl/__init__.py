"""Registry-driven weekly decision engine.

Importing this package never loads data, fits a model, or starts training.
"""

from .candidate_proposals import CandidateGenerator, ProposalContext
from .registry import DecisionRegistry, load_decisions
from .schemas import (
    BusinessState,
    CandidateAction,
    DecisionDefinition,
    ForecastRow,
    IntegerDomain,
    NextWeekForecast,
    ProductState,
    Target,
)

__all__ = [
    "BusinessState",
    "CandidateAction",
    "CandidateGenerator",
    "DecisionDefinition",
    "DecisionRegistry",
    "ForecastRow",
    "IntegerDomain",
    "NextWeekForecast",
    "ProductState",
    "ProposalContext",
    "Target",
    "load_decisions",
]
