"""Environment generation.

Phase 2 scales this into thousands of procedurally generated worlds — the
corpus the core trains on and the held-out set that measures whether it
generalized. `toy` is the first instance and the template.
"""

from .toy import (
    AGENT,
    BACKGROUND,
    CHARGE_PERIOD,
    FIELD,
    HAZARD,
    LEGAL_ACTIONS,
    TARGET,
    WALL,
    Level,
    ToyState,
    ToyWorld,
    default_levels,
    initial_state,
    render_state,
    solve_level,
    step_state,
)
from .toy_model import GridOnlyToyModel, ToyModel

__all__ = [
    "AGENT",
    "BACKGROUND",
    "CHARGE_PERIOD",
    "FIELD",
    "HAZARD",
    "LEGAL_ACTIONS",
    "TARGET",
    "WALL",
    "GridOnlyToyModel",
    "Level",
    "ToyModel",
    "ToyState",
    "ToyWorld",
    "default_levels",
    "initial_state",
    "render_state",
    "solve_level",
    "step_state",
]
