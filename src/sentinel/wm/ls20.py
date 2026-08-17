"""A hand-written, deliberately partial world model of ls20.

Phase 1 exists to prove the machinery works end to end on a real game, and
this model is the proof — but it is also an honest demonstration of the
contract's most important affordance: **it abstains where it does not
know**, and the verifier reports that as reduced coverage rather than as
error.

What it models, established empirically (and confirmed against the
obfuscated source only for the action mapping):

- The playfield is a 5x5-cell lattice with origin (4, 0).
- Cell value 3 is corridor, 4 is wall.
- The player is one 5x5 lattice block; it moves exactly one lattice cell
  per action and is blocked by walls.
- ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right.

What it does NOT model, and therefore abstains on:

- The HUD: the digit display on the left and the depleting step-budget bar
  along the bottom. The bar drains on moves that fail to make progress and
  ends the game at zero.
- Moving hazards. The source shows a collection of sprites that step each
  turn and can collide with the player; their movement rules were not
  established.
- Level layouts beyond the one observed. There are 7 levels and this model
  is built from whichever it is shown.

That list is the point. A model that guessed at the hazards would score
worse *and* mislead the planner; abstaining is the correct answer to a
question you cannot answer, and the architecture is built to reward it.

From Phase 2 onward the agent must induce all of this from interaction
alone. Reading the game source is legitimate for a hand-written Phase 1
baseline and forbidden after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

from sentinel.env.types import GRID_SIZE, Action, Grid, Observation

from .contract import ABSTAIN, Outcome, RenderedGrid, WorldModel

CELL = 5
ORIGIN_X, ORIGIN_Y = 4, 0

CORRIDOR = 3
WALL = 4

# Regions the model knowingly does not explain. Abstaining here is a claim
# about the limits of the hypothesis, not a failure of it.
HUD_LEFT = (0, 13, 52, 64)   # x0, x1, y0, y1 — digit display
HUD_BOTTOM = (0, 64, 59, 64)  # step-budget bar

_MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@dataclass(frozen=True, slots=True)
class Ls20State:
    """Player lattice position. Everything else is unmodelled by design."""

    bx: int
    by: int


def lattice_dims() -> tuple[int, int]:
    return ((GRID_SIZE - ORIGIN_X) // CELL, (GRID_SIZE - ORIGIN_Y) // CELL)


def block_bounds(bx: int, by: int) -> tuple[int, int, int, int]:
    x0 = ORIGIN_X + bx * CELL
    y0 = ORIGIN_Y + by * CELL
    return x0, y0, x0 + CELL, y0 + CELL


def block_value(grid: Grid, bx: int, by: int) -> int | None:
    """Uniform value of a lattice block, or None if mixed."""
    x0, y0, x1, y1 = block_bounds(bx, by)
    if x1 > GRID_SIZE or y1 > GRID_SIZE:
        return None
    first = grid[y0][x0]
    for y in range(y0, y1):
        for x in range(x0, x1):
            if grid[y][x] != first:
                return None
    return first


def _in_hud(x: int, y: int) -> bool:
    for x0, x1, y0, y1 in (HUD_LEFT, HUD_BOTTOM):
        if x0 <= x < x1 and y0 <= y < y1:
            return True
    return False


class Ls20Model(WorldModel):
    """Built by observing one frame; models movement, abstains elsewhere."""

    name = "ls20-partial"

    def __init__(self, first: Observation) -> None:
        self.bw, self.bh = lattice_dims()
        self._base: list[list[int]] = [list(row) for row in first.grid]

        self._walkable: set[tuple[int, int]] = set()
        self._mixed: set[tuple[int, int]] = set()
        for by in range(self.bh):
            for bx in range(self.bw):
                value = block_value(first.grid, bx, by)
                if value == CORRIDOR:
                    self._walkable.add((bx, by))
                elif value is None:
                    self._mixed.add((bx, by))

        self._start = self._find_player(first.grid)
        # The player's own block reads as mixed; it is walkable by definition.
        if self._start is not None:
            self._walkable.add(self._start)

    def _find_player(self, grid: Grid) -> tuple[int, int] | None:
        """The player block is the mixed block inside the playfield.

        Mixed blocks in the HUD are excluded; among the rest the one
        containing the highest-valued non-corridor cells is taken as the
        player. Ambiguity here is reported by abstaining, not guessed at.
        """
        best: tuple[int, int] | None = None
        best_score = -1
        for bx, by in self._mixed:
            x0, y0, x1, y1 = block_bounds(bx, by)
            if _in_hud(x0, y0):
                continue
            cells = [grid[y][x] for y in range(y0, y1) for x in range(x0, x1)]
            score = sum(1 for c in cells if c not in (CORRIDOR, WALL))
            if score > best_score:
                best, best_score = (bx, by), score
        return best

    # -- contract ---------------------------------------------------------

    def init_state(self) -> Hashable:
        if self._start is None:
            return Ls20State(0, 0)
        return Ls20State(*self._start)

    def transition(self, state: Any, action: Action) -> Hashable:
        delta = _MOVES.get(action.action_id)
        if delta is None:
            return state
        nx, ny = state.bx + delta[0], state.by + delta[1]
        if not (0 <= nx < self.bw and 0 <= ny < self.bh):
            return state
        if (nx, ny) not in self._walkable:
            return state
        return Ls20State(nx, ny)

    def render(self, state: Any) -> RenderedGrid:
        """Draw the maze with the player moved; abstain on the unmodelled.

        The player's appearance is copied from where it was first observed
        rather than invented, and the block it vacated is drawn as corridor.
        Both HUD regions abstain unconditionally: the step bar changes on
        rules this model does not track, so predicting it would be a guess
        dressed as knowledge.
        """
        grid = [row[:] for row in self._base]

        if self._start is not None and (state.bx, state.by) != self._start:
            sx0, sy0, sx1, sy1 = block_bounds(*self._start)
            sprite = [
                [self._base[y][x] for x in range(sx0, sx1)] for y in range(sy0, sy1)
            ]
            for y in range(sy0, sy1):
                for x in range(sx0, sx1):
                    grid[y][x] = CORRIDOR
            dx0, dy0, dx1, dy1 = block_bounds(state.bx, state.by)
            if dx1 <= GRID_SIZE and dy1 <= GRID_SIZE:
                for j, y in enumerate(range(dy0, dy1)):
                    for i, x in enumerate(range(dx0, dx1)):
                        grid[y][x] = sprite[j][i]

        for by in range(self.bh):
            for bx in range(self.bw):
                if (bx, by) in self._mixed and (bx, by) != (state.bx, state.by):
                    x0, y0, x1, y1 = block_bounds(bx, by)
                    for y in range(min(y1, GRID_SIZE)):
                        if y < y0:
                            continue
                        for x in range(x0, min(x1, GRID_SIZE)):
                            grid[y][x] = ABSTAIN

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if _in_hud(x, y):
                    grid[y][x] = ABSTAIN

        return tuple(tuple(row) for row in grid)

    def outcome(self, state: Any) -> Outcome:
        """Always ONGOING — an honest admission, not a claim.

        Level completion depends on reaching a goal this model has not
        identified, and game-over depends on the step budget it does not
        track. Reporting ONGOING is wrong at exactly the boundaries, and the
        verifier's outcome channel will say so.
        """
        return Outcome.ONGOING

    def available_actions(self, state: Any) -> tuple[int, ...]:
        return (1, 2, 3, 4)

    def reset_to(self, state: Any) -> Hashable:
        return self.init_state()
