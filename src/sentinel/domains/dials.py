"""A domain with no space in it, for testing whether any of this generalises.

Phase 6 of the plan asks the only question that separates a general system
from an excellent ARC solver: swap the environment for something
structurally different and see whether anything above `env/` had to change.
`gen/toy.py` was billed as a rehearsal for that, but it is still a grid with
an agent walking around it -- the same surface features, the same
inductive biases, the same everything. It cannot fail the test.

DialWorld shares the observation TYPE and nothing else:

    no agent          nothing occupies a cell or moves between cells
    no walls          no notion of blocked, adjacent, or reachable
    no space          a dial's value is a magnitude, not a position
    no collision      actions never interact through geometry

Four dials hold values, rendered as bars. Actions change one dial. The
hidden mechanic is a COUPLING: turning one dial may also turn its neighbour,
which is invisible in any single frame for the same reason `charge_period`
is -- you only see it by comparing what you did with what changed.

The point is not that this is a hard domain. It is that the machinery above
`env/` -- the world-model contract, the verifier, hypothesis search --
should work here unchanged, while the core's spatial features should NOT
transfer. Both halves are informative, and the second is the honest one:
per-value centroids and displacements describe a board with things on it,
and this domain has no things and no board.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Hashable

from sentinel.env.history import History, Step
from sentinel.env.types import (
    GRID_SIZE,
    Action,
    FrameKind,
    GameStateName,
    Observation,
)
from sentinel.wm.contract import Outcome, RenderedGrid, WorldModel

N_DIALS = 4
BAR_COLOUR = 2
TARGET_COLOUR = 5
BACKGROUND = 0
COLUMN_WIDTH = 3
LEGAL_ACTIONS = (1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class DialMechanics:
    """The rules a hypothesis has to get right."""

    step: int = 1
    """How much a turn advances a dial."""
    modulus: int = 10
    """Dial values wrap here."""
    coupling: int = 0
    """0 none; otherwise turning dial i also turns (i + coupling) % N_DIALS.

    The hidden rule. It is invisible in any single frame -- two identical
    boards can respond differently to the same action depending on nothing
    you can see -- and recoverable only by comparing actions to changes."""
    reverse: bool = False
    """Turns decrease instead of increase."""

    def summary(self) -> str:
        bits = [f"step={self.step}", f"mod={self.modulus}"]
        if self.coupling:
            bits.append(f"couple+{self.coupling}")
        if self.reverse:
            bits.append("reverse")
        return " ".join(bits)


def mechanic_space() -> list[DialMechanics]:
    return [
        DialMechanics(step=s, modulus=m, coupling=c, reverse=r)
        for s in (1, 2)
        for m in (8, 10)
        for c in (0, 1, 2)
        for r in (False, True)
    ]


@dataclass(frozen=True, slots=True)
class DialState:
    values: tuple[int, ...]
    target: tuple[int, ...]
    solved: bool = False

    def __hash__(self) -> int:
        return hash((self.values, self.target, self.solved))


def turn(state: DialState, action: Action, mech: DialMechanics) -> DialState:
    """The authoritative rule. Shared by the world and its exact model."""
    if state.solved:
        return state
    index = action.action_id - 1
    if not 0 <= index < N_DIALS:
        return state

    values = list(state.values)
    delta = -mech.step if mech.reverse else mech.step
    values[index] = (values[index] + delta) % mech.modulus
    if mech.coupling:
        other = (index + mech.coupling) % N_DIALS
        values[other] = (values[other] + delta) % mech.modulus

    new = tuple(values)
    return replace(state, values=new, solved=new == state.target)


def render(state: DialState) -> tuple[tuple[int, ...], ...]:
    """Draw dials as bars. Height is a magnitude, not a position."""
    grid = [[BACKGROUND] * GRID_SIZE for _ in range(GRID_SIZE)]
    for i, value in enumerate(state.values):
        x0 = i * COLUMN_WIDTH
        for h in range(value):
            for x in range(x0, x0 + COLUMN_WIDTH - 1):
                grid[GRID_SIZE - 1 - h][x] = BAR_COLOUR
    for i, want in enumerate(state.target):
        x0 = i * COLUMN_WIDTH
        y = GRID_SIZE - 1 - want
        if 0 <= y < GRID_SIZE:
            grid[y][x0 + COLUMN_WIDTH - 1] = TARGET_COLOUR
    return tuple(tuple(row) for row in grid)


class DialWorld:
    """Emits the same History type the ARC engine does, sharing no code."""

    def __init__(self, mech: DialMechanics, start: tuple[int, ...],
                 target: tuple[int, ...], world_id: str = "dials") -> None:
        self.mech = mech
        self.game_id = world_id
        self._start = DialState(values=tuple(start), target=tuple(target))
        self._state = self._start
        self._history: History | None = None

    @property
    def history(self) -> History:
        if self._history is None:
            self.reset()
        assert self._history is not None
        return self._history

    @property
    def done(self) -> bool:
        return self._state.solved

    def _observe(self, kind: FrameKind, full_reset: bool = False) -> Observation:
        return Observation(
            grid=render(self._state),
            state=GameStateName.WIN if self._state.solved else GameStateName.NOT_FINISHED,
            levels_completed=1 if self._state.solved else 0,
            win_levels=1,
            available_actions=LEGAL_ACTIONS,
            full_reset=full_reset,
            kind=kind,
        )

    def reset(self) -> Observation:
        self._state = self._start
        obs = self._observe(FrameKind.RESET, full_reset=True)
        self._history = History(game_id=self.game_id, seed=0, initial=obs)
        return obs

    def step(self, action: Action | int) -> Observation:
        if isinstance(action, int):
            action = Action(action)
        history = self.history
        self._state = turn(self._state, action, self.mech)
        kind = FrameKind.TERMINAL if self._state.solved else FrameKind.DECISION
        obs = self._observe(kind)
        history.append(
            Step(index=len(history.steps), action=action, frames=(obs,), level_index=0)
        )
        return obs


class DialModel(WorldModel):
    """A hypothesis about a dial world, under the same contract as a grid one."""

    def __init__(self, mech: DialMechanics, start: tuple[int, ...],
                 target: tuple[int, ...], name: str = "dials") -> None:
        self.mech = mech
        self.name = name
        self._start = DialState(values=tuple(start), target=tuple(target))

    def init_state(self) -> Hashable:
        return self._start

    def transition(self, state: Any, action: Action) -> Hashable:
        return turn(state, action, self.mech)

    def render(self, state: Any) -> RenderedGrid:
        return render(state)

    def outcome(self, state: Any) -> Outcome:
        return Outcome.LEVEL_COMPLETE if state.solved else Outcome.ONGOING

    def available_actions(self, state: Any) -> tuple[int, ...]:
        return LEGAL_ACTIONS
