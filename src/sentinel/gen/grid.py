"""The generic grid-world engine and its exact model.

One transition rule, parameterised by `Mechanics`, used by both the world
and its reference model. That sharing is deliberate: the reference model
must be correct *by construction*, or every measurement taken against it
inherits an unknown error.

Every generated world ships with this exact model. It gives three things
the corpus needs: a solvability check before a world is emitted, a ceiling
to measure the teacher's output against, and — most interestingly — a way
to notice when the teacher invents a model that is *different from* the
reference but equally correct. That last case is not a failure. It is the
system finding a second valid description of the same world, which is
exactly what a general induction system should sometimes do.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Hashable, Iterable

from sentinel.env.history import History
from sentinel.env.types import (
    GRID_SIZE,
    Action,
    FrameKind,
    GameStateName,
    Observation,
    Step,
)
from sentinel.wm.contract import Outcome, RenderedGrid, WorldModel

from .spec import LevelSpec, Mechanics, WorldSpec

BACKGROUND = 0
WALL = 1
HAZARD = 2
TARGET = 3
AGENT = 4
SWITCH = 5
GATE_CLOSED = 6
GATE_OPEN = 7

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
LEGAL_ACTIONS = (1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class GridState:
    """Complete state. `charge` and `gates_open` are invisible in the render."""

    level: int
    x: int
    y: int
    collected: int
    """Count of targets taken, in order. Under unordered rules the set is
    derived from `remaining` instead."""
    remaining: frozenset[tuple[int, int]]
    charge: int
    gates_open: bool
    dead: bool
    cleared: bool


def _wrap(value: int, size: int) -> int:
    return value % size


def blocked(pos: tuple[int, int], level: LevelSpec, size: int, gates_open: bool) -> bool:
    x, y = pos
    if not (0 <= x < size and 0 <= y < size):
        return True
    if pos in level.walls:
        return True
    if pos in level.gates and not gates_open:
        return True
    return False


def transition_state(
    state: GridState, action: Action, spec: WorldSpec
) -> GridState:
    """The single authoritative transition rule."""
    if state.dead or state.cleared:
        return state

    level = spec.levels[state.level]
    mech = spec.mechanics
    size = spec.field_size
    aid = action.action_id

    if aid not in MOVES:
        # Waiting still advances charge. Otherwise the hidden counter could
        # be recovered by elimination, which would make it not hidden.
        return replace(state, charge=state.charge + 1)

    charge = state.charge + 1
    distance = mech.step_distance
    if mech.charge_period and charge % mech.charge_period == 0:
        distance += 1

    dx, dy = MOVES[aid]
    x, y = state.x, state.y
    gates_open = state.gates_open

    for _ in range(distance):
        nx, ny = x + dx, y + dy
        if mech.wrap_edges:
            nx, ny = _wrap(nx, size), _wrap(ny, size)
        if blocked((nx, ny), level, size, gates_open):
            break
        x, y = nx, ny
        if mech.has_hazards and (x, y) in level.hazards:
            return replace(state, x=x, y=y, charge=charge, dead=True)
        if mech.has_switches and (x, y) in level.switches:
            gates_open = not gates_open

    remaining = state.remaining
    collected = state.collected
    here = (x, y)

    if here in remaining:
        if mech.ordered_targets:
            # Only the next target in sequence counts. Stepping on a later
            # one does nothing, which is what defeats greedy routing.
            if collected < len(level.targets) and level.targets[collected] == here:
                remaining = remaining - {here}
                collected += 1
        else:
            remaining = remaining - {here}
            collected += 1

    return replace(
        state,
        x=x,
        y=y,
        collected=collected,
        remaining=remaining,
        charge=charge,
        gates_open=gates_open,
        cleared=not remaining,
    )


def render_grid(state: GridState, spec: WorldSpec) -> tuple[tuple[int, ...], ...]:
    """Draw state. Note what is absent: `charge`, and which gates are which."""
    level = spec.levels[state.level]
    mech = spec.mechanics
    grid = [[BACKGROUND] * GRID_SIZE for _ in range(GRID_SIZE)]

    for x, y in level.walls:
        grid[y][x] = WALL
    if mech.has_hazards:
        for x, y in level.hazards:
            grid[y][x] = HAZARD
    if mech.has_switches:
        for x, y in level.switches:
            grid[y][x] = SWITCH
        for x, y in level.gates:
            grid[y][x] = GATE_OPEN if state.gates_open else GATE_CLOSED
    for x, y in state.remaining:
        grid[y][x] = TARGET
    grid[state.y][state.x] = AGENT

    return tuple(tuple(row) for row in grid)


def initial_state(level_index: int, spec: WorldSpec) -> GridState:
    level = spec.levels[level_index]
    return GridState(
        level=level_index,
        x=level.start[0],
        y=level.start[1],
        collected=0,
        remaining=frozenset(level.targets),
        charge=0,
        gates_open=False,
        dead=False,
        cleared=not level.targets,
    )


class GridWorld:
    """Drives GridState and emits the same History type the ARC engine does."""

    def __init__(self, spec: WorldSpec) -> None:
        self.spec = spec
        self.game_id = spec.world_id
        self._state = initial_state(0, spec)
        self._levels_completed = 0
        self._history: History | None = None

    @property
    def state(self) -> GridState:
        return self._state

    @property
    def history(self) -> History:
        if self._history is None:
            raise RuntimeError("call reset() first")
        return self._history

    @property
    def done(self) -> bool:
        return self._state.dead or self._levels_completed >= self.spec.num_levels

    def _observe(self, kind: FrameKind, full_reset: bool = False) -> Observation:
        if self._state.dead:
            game_state = GameStateName.GAME_OVER
        elif self._levels_completed >= self.spec.num_levels:
            game_state = GameStateName.WIN
        else:
            game_state = GameStateName.NOT_FINISHED
        return Observation(
            grid=render_grid(self._state, self.spec),
            state=game_state,
            levels_completed=self._levels_completed,
            win_levels=self.spec.num_levels,
            available_actions=LEGAL_ACTIONS,
            full_reset=full_reset,
            kind=kind,
        )

    def reset(self) -> Observation:
        self._state = initial_state(0, self.spec)
        self._levels_completed = 0
        obs = self._observe(FrameKind.RESET, full_reset=True)
        self._history = History(
            game_id=self.game_id, seed=self.spec.seed, initial=obs
        )
        return obs

    def step(self, action: Action | int) -> Observation:
        if isinstance(action, int):
            action = Action(action)
        history = self.history

        self._state = transition_state(self._state, action, self.spec)

        if self._state.cleared:
            self._levels_completed += 1
            if self._levels_completed < self.spec.num_levels:
                self._state = initial_state(self._levels_completed, self.spec)

        kind = (
            FrameKind.TERMINAL
            if (self._state.dead or self._levels_completed >= self.spec.num_levels)
            else FrameKind.DECISION
        )
        obs = self._observe(kind)
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


class GridWorldModel(WorldModel):
    """The exact model of a GridWorld. Correct by construction."""

    def __init__(self, spec: WorldSpec, level_index: int = 0) -> None:
        self.spec = spec
        self.name = f"exact:{spec.world_id}"
        self._level_index = level_index

    def init_state(self) -> Hashable:
        return initial_state(self._level_index, self.spec)

    def transition(self, state: Any, action: Action) -> Hashable:
        return transition_state(state, action, self.spec)

    def render(self, state: Any) -> RenderedGrid:
        return render_grid(state, self.spec)

    def outcome(self, state: Any) -> Outcome:
        if state.dead:
            return Outcome.GAME_OVER
        if state.cleared:
            return Outcome.LEVEL_COMPLETE
        return Outcome.ONGOING

    def available_actions(self, state: Any) -> tuple[int, ...]:
        return LEGAL_ACTIONS

    def reset_to(self, state: Any) -> Hashable:
        nxt = min(state.level + 1, self.spec.num_levels - 1)
        return initial_state(nxt, self.spec)


def solve_level(
    spec: WorldSpec,
    level_index: int,
    limit: int = 60_000,
    start: GridState | None = None,
) -> list[Action] | None:
    """Shortest action sequence clearing one level, via BFS on the exact model.

    Used as the solvability gate before a generated world is emitted. An
    unsolvable world in the corpus would teach the teacher that giving up is
    sometimes right.

    `start` defaults to the level's opening state, but may be supplied to
    solve from wherever the agent currently stands. That matters after
    exploration: a solution computed from the opening position is invalid
    once the agent has moved, and replaying it blindly walks into hazards.
    """
    if start is None:
        start = initial_state(level_index, spec)
    if start.cleared:
        return []

    queue: deque[tuple[GridState, list[Action]]] = deque([(start, [])])
    seen = {start}
    explored = 0

    while queue and explored < limit:
        state, path = queue.popleft()
        explored += 1
        for aid in LEGAL_ACTIONS:
            action = Action(aid)
            nxt = transition_state(state, action, spec)
            if nxt.dead or nxt in seen:
                continue
            if nxt.cleared:
                return [*path, action]
            seen.add(nxt)
            queue.append((nxt, [*path, action]))
    return None


def solve_world(spec: WorldSpec, limit: int = 60_000) -> list[Action] | None:
    """Concatenated solution for every level, or None if any is unreachable."""
    full: list[Action] = []
    for i in range(spec.num_levels):
        path = solve_level(spec, i, limit=limit)
        if path is None:
            return None
        full.extend(path)
    return full
