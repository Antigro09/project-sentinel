"""Planning layer — search inside the model, not the world.

Environment actions are what the benchmark charges for; simulated actions
are free. Everything here exists to shift work from the former to the
latter, and to notice immediately when the model it is searching in turns
out to be wrong.
"""

from .search import (
    DEFAULT_ACTIONS,
    BFSPlanner,
    ExecutionResult,
    Plan,
    PlanExecutor,
    SearchStats,
)

__all__ = [
    "DEFAULT_ACTIONS",
    "BFSPlanner",
    "ExecutionResult",
    "Plan",
    "PlanExecutor",
    "SearchStats",
]
