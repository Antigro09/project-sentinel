"""Choosing actions for what they would reveal.

Evidence coverage is a consequence of behaviour, not a property of a world.
This layer designs experiments using the rules already inferred, which is
forced rather than optional: an agent cannot aim an experiment about target
ORDER until it knows how far it moves.
"""

from .staged import StagedResult, staged_exploration
from .version_space import (
    VersionSpace,
    information_gain_history,
    planned_information_gain_history,
    observed_facts,
    state_key,
)

__all__ = [
    "StagedResult",
    "VersionSpace",
    "information_gain_history",
    "planned_information_gain_history",
    "observed_facts",
    "staged_exploration",
    "state_key",
]
