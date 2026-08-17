"""Environment generation.

Phase 2 scales this into thousands of procedurally generated worlds — the
corpus the core trains on and the held-out set that measures whether it
generalized. `toy` is the first instance and the template.
"""

from .generator import (
    GeneratorConfig,
    Split,
    generate,
    generate_many,
    iter_worlds,
    make_split,
    mechanic_space,
)
from .grid import GridState, GridWorld, GridWorldModel, solve_world
from .grid import solve_level as solve_generated_level
from .spec import LevelSpec, Mechanics, WorldSpec
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
    "GeneratorConfig",
    "GridOnlyToyModel",
    "GridState",
    "GridWorld",
    "GridWorldModel",
    "Level",
    "LevelSpec",
    "Mechanics",
    "Split",
    "ToyModel",
    "ToyState",
    "ToyWorld",
    "WorldSpec",
    "default_levels",
    "generate",
    "generate_many",
    "initial_state",
    "iter_worlds",
    "make_split",
    "mechanic_space",
    "render_state",
    "solve_generated_level",
    "solve_level",
    "solve_world",
    "step_state",
]
