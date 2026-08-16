"""Core observation types for the environment layer.

Everything here is immutable and hashable. The verifier replays recorded
history through candidate world models and compares predictions against
what actually happened, so observations must have stable identity and a
stable digest across processes and runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

Grid = tuple[tuple[int, ...], ...]

GRID_SIZE = 64
MAX_CELL_VALUE = 15


class FrameKind(str, Enum):
    """Why a frame exists in the history.

    The engine returns a *list* of grids per step: intermediate animation
    frames followed by the settled state. Collapsing these together corrupts
    the verifier, because a world model that correctly predicts the settled
    transition would be scored against animation frames it was never asked
    to model. So every frame carries its kind and the verifier filters.
    """

    RESET = "reset"
    """First frame of an episode, or after a full reset."""

    DECISION = "decision"
    """Settled state the agent actually chooses an action from."""

    TRANSIENT = "transient"
    """Intermediate animation frame emitted between settled states."""

    TERMINAL = "terminal"
    """Frame where the episode ended (WIN or GAME_OVER)."""


class GameStateName(str, Enum):
    """Mirror of arcengine.GameState, decoupled so logs stay readable."""

    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


@dataclass(frozen=True, slots=True)
class Observation:
    """A single settled or transient view of the environment."""

    grid: Grid
    state: GameStateName
    levels_completed: int
    win_levels: int
    available_actions: tuple[int, ...]
    full_reset: bool
    kind: FrameKind

    @property
    def is_terminal(self) -> bool:
        return self.state in (GameStateName.WIN, GameStateName.GAME_OVER)

    def digest(self) -> str:
        """Stable content hash.

        Used for replay-determinism assertions and for detecting divergence
        between a world model's predicted frame and the real one.
        """
        h = hashlib.blake2b(digest_size=16)
        for row in self.grid:
            h.update(bytes(row))
        h.update(self.state.value.encode())
        h.update(
            f"|{self.levels_completed}|{self.win_levels}|{self.full_reset}|"
            f"{','.join(map(str, self.available_actions))}".encode()
        )
        return h.hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "grid": [list(row) for row in self.grid],
            "state": self.state.value,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "available_actions": list(self.available_actions),
            "full_reset": self.full_reset,
            "kind": self.kind.value,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Observation:
        return cls(
            grid=tuple(tuple(int(c) for c in row) for row in d["grid"]),
            state=GameStateName(d["state"]),
            levels_completed=int(d["levels_completed"]),
            win_levels=int(d["win_levels"]),
            available_actions=tuple(int(a) for a in d["available_actions"]),
            full_reset=bool(d["full_reset"]),
            kind=FrameKind(d["kind"]),
        )


@dataclass(frozen=True, slots=True)
class Action:
    """An action submitted to the environment.

    ACTION6 carries x/y coordinates; the rest carry nothing. Keeping the
    coordinates in the action rather than a side channel means the recorded
    action sequence alone is enough to reproduce an episode.
    """

    action_id: int
    x: int | None = None
    y: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.action_id <= 7:
            raise ValueError(f"action_id must be 1..7, got {self.action_id}")
        if self.action_id == 6:
            if self.x is None or self.y is None:
                raise ValueError("ACTION6 requires x and y coordinates")
            if not (0 <= self.x < GRID_SIZE and 0 <= self.y < GRID_SIZE):
                raise ValueError(f"ACTION6 coords out of range: ({self.x}, {self.y})")
        elif self.x is not None or self.y is not None:
            raise ValueError(f"ACTION{self.action_id} does not take coordinates")

    @property
    def data(self) -> dict[str, int]:
        """Payload in the shape the engine expects."""
        if self.action_id == 6:
            assert self.x is not None and self.y is not None
            return {"x": self.x, "y": self.y}
        return {}

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action_id": self.action_id}
        if self.action_id == 6:
            d["x"], d["y"] = self.x, self.y
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Action:
        return cls(
            action_id=int(d["action_id"]),
            x=None if d.get("x") is None else int(d["x"]),
            y=None if d.get("y") is None else int(d["y"]),
        )

    def __str__(self) -> str:
        if self.action_id == 6:
            return f"ACTION6({self.x},{self.y})"
        return f"ACTION{self.action_id}"


@dataclass(frozen=True, slots=True)
class Step:
    """One action and everything the environment emitted in response.

    `frames` holds the full emission in order. `settled` is the frame the
    agent reasons from. A world model is asked to predict `settled`; the
    transient frames are retained because some games leak hidden state
    through animation, which is evidence worth keeping even if we do not
    score against it.
    """

    index: int
    action: Action
    frames: tuple[Observation, ...]
    level_index: int

    @property
    def settled(self) -> Observation:
        return self.frames[-1]

    @property
    def transients(self) -> tuple[Observation, ...]:
        return self.frames[:-1]

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.to_json(),
            "frames": [f.to_json() for f in self.frames],
            "level_index": self.level_index,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Step:
        return cls(
            index=int(d["index"]),
            action=Action.from_json(d["action"]),
            frames=tuple(Observation.from_json(f) for f in d["frames"]),
            level_index=int(d["level_index"]),
        )
