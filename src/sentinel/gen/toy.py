"""A tiny, fully-controlled grid world.

Built for a specific reason: the verifier's detection power can only be
tested against evidence that actually exercises the thing being tested.
Random play on a real ARC game routinely never completes a level and never
dies, so a model that says "nothing ever ends" is *factually correct* on
that history and cannot be falsified by it. That is not a flaw in the
verifier — it is the exploration problem showing up early, and it is why
`explore/` exists as a first-class layer later.

ToyWorld solves that by being small enough to drive into any state on
demand: actions matter, levels complete, hazards kill. Every channel the
verifier scores can be exercised deliberately.

It also carries the property that makes ARC-AGI-3 hard, deliberately:

    **The world is non-Markov.** A hidden `charge` counter makes every
    third move travel two cells instead of one. Two moments with a
    byte-identical grid can therefore have different successors. A model
    whose state is "just the grid" cannot be correct here — which is
    exactly the discipline the real benchmark demands.

This is the seed of the Phase 2 procedural generator, not throwaway test
scaffolding. It is also a rehearsal for Phase 6: it produces the same
`History` type the ARC engine does, through none of the same code, which
is the first evidence that the `env/` abstraction actually abstracts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Hashable, Iterable

from sentinel.env.history import History
from sentinel.env.types import (
    GRID_SIZE,
    Action,
    FrameKind,
    GameStateName,
    Observation,
    Step,
)

BACKGROUND = 0
WALL = 1
HAZARD = 2
TARGET = 3
AGENT = 4

FIELD = 16
"""Playfield is FIELD x FIELD in the top-left; the rest of the 64x64 is background."""

CHARGE_PERIOD = 3
"""Every CHARGE_PERIOD-th move travels two cells. The hidden mechanic."""

_DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
LEGAL_ACTIONS = (1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class Level:
    """A static layout. Walls and hazards never move."""

    start: tuple[int, int]
    walls: frozenset[tuple[int, int]]
    hazards: frozenset[tuple[int, int]]
    targets: frozenset[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class ToyState:
    """Complete state — visible position plus the invisible charge counter."""

    level: int
    x: int
    y: int
    remaining: frozenset[tuple[int, int]]
    charge: int
    dead: bool
    cleared: bool

    def __hash__(self) -> int:
        return hash((self.level, self.x, self.y, self.remaining, self.charge, self.dead, self.cleared))


def default_levels() -> tuple[Level, ...]:
    """Three hand-built levels: a walk, a wall to route around, a hazard gauntlet."""
    return (
        Level(
            start=(1, 1),
            walls=frozenset({(x, 5) for x in range(0, 10)}),
            hazards=frozenset(),
            targets=frozenset({(8, 1), (3, 3)}),
        ),
        Level(
            start=(1, 1),
            walls=frozenset({(4, y) for y in range(0, 12)} | {(9, y) for y in range(4, 16)}),
            hazards=frozenset({(6, 6)}),
            targets=frozenset({(7, 2), (11, 9)}),
        ),
        Level(
            start=(0, 0),
            walls=frozenset({(2, y) for y in range(0, 8)}),
            hazards=frozenset({(5, 5), (6, 5), (7, 5), (5, 9)}),
            targets=frozenset({(12, 12), (1, 14), (13, 1)}),
        ),
    )


def _blocked(pos: tuple[int, int], level: Level) -> bool:
    x, y = pos
    return not (0 <= x < FIELD and 0 <= y < FIELD) or pos in level.walls


def step_state(state: ToyState, action: Action, levels: tuple[Level, ...]) -> ToyState:
    """The authoritative transition rule.

    Both the world and its reference model call this, so the model is
    correct *by construction*. That matters: the verifier gate needs a
    subject whose correctness is not itself in question, otherwise a failed
    mutation test is ambiguous between "verifier is blind" and "model was
    already wrong".
    """
    if state.dead or state.cleared:
        return state  # absorbing

    level = levels[state.level]
    aid = action.action_id

    if aid not in _DELTA:
        # Waiting still burns charge — otherwise the hidden counter would be
        # observable by elimination, which would defeat the point.
        return replace(state, charge=state.charge + 1)

    charge = state.charge + 1
    distance = 2 if charge % CHARGE_PERIOD == 0 else 1
    dx, dy = _DELTA[aid]

    x, y = state.x, state.y
    for _ in range(distance):
        nxt = (x + dx, y + dy)
        if _blocked(nxt, level):
            break
        x, y = nxt
        if (x, y) in level.hazards:
            return replace(state, x=x, y=y, charge=charge, dead=True)

    remaining = state.remaining - {(x, y)}
    return replace(
        state,
        x=x,
        y=y,
        remaining=remaining,
        charge=charge,
        cleared=not remaining,
    )


def render_state(state: ToyState, levels: tuple[Level, ...]) -> tuple[tuple[int, ...], ...]:
    """Draw state to a 64x64 grid. Note what is *not* drawn: `charge`."""
    level = levels[state.level]
    grid = [[BACKGROUND] * GRID_SIZE for _ in range(GRID_SIZE)]
    for x, y in level.walls:
        grid[y][x] = WALL
    for x, y in level.hazards:
        grid[y][x] = HAZARD
    for x, y in state.remaining:
        grid[y][x] = TARGET
    grid[state.y][state.x] = AGENT
    return tuple(tuple(row) for row in grid)


def initial_state(level_index: int, levels: tuple[Level, ...]) -> ToyState:
    level = levels[level_index]
    return ToyState(
        level=level_index,
        x=level.start[0],
        y=level.start[1],
        remaining=level.targets,
        charge=0,
        dead=False,
        cleared=False,
    )


class ToyWorld:
    """Drives ToyState and emits the same `History` type the ARC engine does."""

    def __init__(self, levels: tuple[Level, ...] | None = None, seed: int = 0) -> None:
        self.levels = levels or default_levels()
        self.seed = seed
        self.game_id = "toy"
        self._state = initial_state(0, self.levels)
        self._levels_completed = 0
        self._history: History | None = None

    @property
    def state(self) -> ToyState:
        return self._state

    @property
    def history(self) -> History:
        if self._history is None:
            raise RuntimeError("call reset() first")
        return self._history

    @property
    def done(self) -> bool:
        return self._state.dead or self._levels_completed >= len(self.levels)

    def _observe(self, *, kind: FrameKind, full_reset: bool = False) -> Observation:
        if self._state.dead:
            game_state = GameStateName.GAME_OVER
        elif self._levels_completed >= len(self.levels):
            game_state = GameStateName.WIN
        else:
            game_state = GameStateName.NOT_FINISHED
        return Observation(
            grid=render_state(self._state, self.levels),
            state=game_state,
            levels_completed=self._levels_completed,
            win_levels=len(self.levels),
            available_actions=LEGAL_ACTIONS,
            full_reset=full_reset,
            kind=kind,
        )

    def reset(self) -> Observation:
        self._state = initial_state(0, self.levels)
        self._levels_completed = 0
        obs = self._observe(kind=FrameKind.RESET)
        self._history = History(game_id=self.game_id, seed=self.seed, initial=obs)
        return obs

    def step(self, action: Action | int) -> Observation:
        if isinstance(action, int):
            action = Action(action)
        history = self.history

        self._state = step_state(self._state, action, self.levels)

        advanced = False
        if self._state.cleared:
            self._levels_completed += 1
            advanced = True
            if self._levels_completed < len(self.levels):
                self._state = initial_state(self._levels_completed, self.levels)

        kind = (
            FrameKind.TERMINAL
            if (self._state.dead or self._levels_completed >= len(self.levels))
            else FrameKind.DECISION
        )
        obs = self._observe(kind=kind)
        history.append(
            Step(
                index=len(history.steps),
                action=action,
                frames=(obs,),
                level_index=self._levels_completed,
            )
        )
        return obs

    def play(self, actions: Iterable[Action | int]) -> History:
        if self._history is None:
            self.reset()
        for a in actions:
            if self.done:
                break
            self.step(a)
        return self.history


def solve_level(state: ToyState, levels: tuple[Level, ...], limit: int = 20000) -> list[Action] | None:
    """BFS to clear the current level. Used to generate histories that
    actually reach level boundaries — the evidence random play never finds."""
    from collections import deque

    start = state
    queue: deque[tuple[ToyState, list[Action]]] = deque([(start, [])])
    seen = {start}
    explored = 0

    while queue and explored < limit:
        current, path = queue.popleft()
        explored += 1
        for aid in LEGAL_ACTIONS:
            action = Action(aid)
            nxt = step_state(current, action, levels)
            if nxt.dead or nxt in seen:
                continue
            if nxt.cleared:
                return [*path, action]
            seen.add(nxt)
            queue.append((nxt, [*path, action]))
    return None
