"""The world model contract.

A world model is a *falsifiable hypothesis about an environment, expressed
as an executable program*. Not weights, not a description — code that runs
forward and can be proven wrong.

Formally this is a Rendered Deterministic Moore Machine:

    init_state()              -> State      may include hidden variables
    transition(state, action) -> State      deterministic
    render(state)             -> Grid       may abstain per cell
    outcome(state)            -> Outcome

Two properties of this contract carry most of its weight.

**Hidden state is mandatory, not optional.** ARC-AGI-3 observations are
non-Markov: the same visible 64x64 grid can have different successors
depending on history. A model whose State is just the grid therefore
cannot be correct in general. Forcing State to be a separate type the
author defines is what makes "posit unobserved structure" the default
rather than an afterthought.

**Abstention is a first-class answer.** `render` may return ABSTAIN for
any cell, meaning "I do not predict this". A model that knows the edges of
its own competence is far more useful than one that confabulates, and it
lets the verifier report accuracy and coverage as independent numbers.
Collapsing those two into a single score is the single easiest way to
destroy the reward signal: a model that abstains everywhere would look
perfect, and a model that predicts everything slightly wrong would look
worse than one that predicts nothing at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Hashable, Protocol, runtime_checkable

from sentinel.env.types import GRID_SIZE, Action, Grid

ABSTAIN = -1
"""Cell value meaning "no prediction". Real cells are 0..15, so -1 is unambiguous."""

RenderedGrid = tuple[tuple[int, ...], ...]
"""Like Grid, but cells may be ABSTAIN."""


class Outcome(str, Enum):
    """What the model believes the current state means."""

    ONGOING = "ongoing"
    LEVEL_COMPLETE = "level_complete"
    GAME_OVER = "game_over"


class ModelError(RuntimeError):
    """A world model failed while executing.

    Crashes are evidence too — a model that raises on a state the real
    environment reached is falsified by that fact, and the verifier records
    it as such rather than letting the exception escape.
    """


@runtime_checkable
class State(Protocol):
    """Model state must be hashable so planners can dedup search nodes.

    Frozen dataclasses and tuples both satisfy this. Mutable state is
    rejected by construction, which also keeps `transition` honest about
    being a pure function.
    """

    def __hash__(self) -> int: ...


class WorldModel(ABC):
    """Base class for an executable hypothesis about an environment.

    Subclasses are written by hand in Phase 1, by an LLM in Phase 2, and
    by the trained core from Phase 3 onward. The contract does not change
    across those — only the author does.
    """

    name: str = "unnamed"

    @abstractmethod
    def init_state(self) -> Hashable:
        """State at the start of a level, before any action."""

    @abstractmethod
    def transition(self, state: Any, action: Action) -> Hashable:
        """Successor state. Must be deterministic and side-effect free.

        Illegal or no-op actions should return a state equal to the input
        rather than raising — the environment tolerates them, so the model
        must model that tolerance.
        """

    @abstractmethod
    def render(self, state: Any) -> RenderedGrid:
        """Draw state back to a 64x64 grid. Use ABSTAIN where unsure."""

    @abstractmethod
    def outcome(self, state: Any) -> Outcome:
        """Classify the state."""

    def available_actions(self, state: Any) -> tuple[int, ...] | None:
        """Optional: which action ids are legal here.

        Returning None means "unknown, try everything". A model that can
        narrow this shrinks the planner's branching factor, which is the
        cheapest speedup available to search.
        """
        return None

    def reset_to(self, state: Any) -> Hashable:
        """State after an engine-imposed reset.

        Defaults to a fresh init_state. Models with cross-level memory
        (score, inventory carried between levels) override this to keep
        what genuinely survives a reset.
        """
        return self.init_state()

    def describe(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


def full_abstain() -> RenderedGrid:
    """A render that predicts nothing. The honest default for a new model."""
    return tuple(tuple(ABSTAIN for _ in range(GRID_SIZE)) for _ in range(GRID_SIZE))


def grid_to_rendered(grid: Grid) -> RenderedGrid:
    """Treat a fully-known grid as a render with no abstentions."""
    return tuple(tuple(int(c) for c in row) for row in grid)


def validate_rendered(rendered: Any) -> RenderedGrid:
    """Check a render is well-formed before the verifier scores it.

    A model that returns a malformed grid is broken in a different way
    than one that predicts wrongly, and the distinction matters when
    deciding whether to repair or discard it.
    """
    if not isinstance(rendered, tuple) or len(rendered) != GRID_SIZE:
        raise ModelError(
            f"render must return {GRID_SIZE} rows as a tuple, "
            f"got {type(rendered).__name__} of length "
            f"{len(rendered) if hasattr(rendered, '__len__') else '?'}"
        )
    for y, row in enumerate(rendered):
        if not isinstance(row, tuple) or len(row) != GRID_SIZE:
            raise ModelError(
                f"render row {y} must be a tuple of {GRID_SIZE} cells, "
                f"got {type(row).__name__} of length "
                f"{len(row) if hasattr(row, '__len__') else '?'}"
            )
        for x, cell in enumerate(row):
            if not isinstance(cell, int) or isinstance(cell, bool):
                raise ModelError(f"render cell ({x},{y}) is {type(cell).__name__}, not int")
            if cell != ABSTAIN and not 0 <= cell <= 15:
                raise ModelError(
                    f"render cell ({x},{y}) is {cell}; must be 0..15 or ABSTAIN ({ABSTAIN})"
                )
    return rendered
