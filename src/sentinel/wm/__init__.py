"""World models — executable hypotheses about environments.

A model here is Python that runs forward and can be proven wrong, not a
set of weights. That choice is the core of the architecture bet: a program
can be verified, repaired at the point it broke, simplified toward general
rules, and read by a human. Weights can do none of those.
"""

from .contract import (
    ABSTAIN,
    ModelError,
    Outcome,
    RenderedGrid,
    State,
    WorldModel,
    full_abstain,
    grid_to_rendered,
    validate_rendered,
)

__all__ = [
    "ABSTAIN",
    "ModelError",
    "Outcome",
    "RenderedGrid",
    "State",
    "WorldModel",
    "full_abstain",
    "grid_to_rendered",
    "validate_rendered",
]
