"""Environment layer.

The boundary between Sentinel and whatever world it is currently learning.
Today that is the ARC-AGI-3 engine; Phase 6 replaces it with something
structurally different, and nothing above this package should notice.
"""

from .episode import EpisodeLog, ReplayMismatch, record
from .history import History, classify
from .runner import Runner, available_games
from .types import (
    GRID_SIZE,
    MAX_CELL_VALUE,
    Action,
    FrameKind,
    GameStateName,
    Grid,
    Observation,
    Step,
)

__all__ = [
    "GRID_SIZE",
    "MAX_CELL_VALUE",
    "Action",
    "EpisodeLog",
    "FrameKind",
    "GameStateName",
    "Grid",
    "History",
    "Observation",
    "ReplayMismatch",
    "Runner",
    "Step",
    "available_games",
    "classify",
    "record",
]
