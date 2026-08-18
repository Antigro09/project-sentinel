"""Test-time adaptation — claim 3 of the plan.

A frozen system cannot be general. This layer changes the core's behaviour
during the task it is currently facing, scored only by the verifier, with
no labels and no oracle.

Search and learning are kept apart on purpose: `select` searches without
touching weights, `adapt` takes gradient steps. Any benefit attributed to
adaptation has to beat search, or it is not evidence for the claim.
"""

from .hypothesis import (
    CHARGE_FROM_CLASS,
    HEAD_ORDER,
    ScoredHypothesis,
    classes_from_mechanics,
    scorable_segment,
    mechanics_from_classes,
    score_hypothesis,
)
from .search import ALL_HYPOTHESES, SIMPLICITY_ORDER, SearchResult, exhaustive_search
from .ttt import (
    AdaptConfig,
    AdaptResult,
    adapt,
    adapted_mechanics,
    frozen,
    head_distributions,
    select,
)

__all__ = [
    "ALL_HYPOTHESES",
    "SIMPLICITY_ORDER",
    "SearchResult",
    "exhaustive_search",
    "CHARGE_FROM_CLASS",
    "HEAD_ORDER",
    "AdaptConfig",
    "AdaptResult",
    "ScoredHypothesis",
    "adapt",
    "adapted_mechanics",
    "classes_from_mechanics",
    "scorable_segment",
    "frozen",
    "head_distributions",
    "mechanics_from_classes",
    "score_hypothesis",
    "select",
]
