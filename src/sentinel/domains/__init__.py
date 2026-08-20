"""Environments other than the grid, for testing whether anything generalises.

Phase 6 of the plan is the gate that separates a general system from an
excellent ARC solver. It only means something if the new environment is
structurally different rather than cosmetically different.
"""

from .dials import (
    DialMechanics,
    DialModel,
    DialState,
    DialWorld,
    mechanic_space,
)

__all__ = [
    "DialMechanics",
    "DialModel",
    "DialState",
    "DialWorld",
    "mechanic_space",
]
