"""Deliberate bug injection — the verifier's own test suite.

The Phase 1 gate is not "the verifier runs". It is "the verifier cannot be
fooled". A reward signal that misses subtle errors is worse than no reward
signal, because everything downstream will optimise into the blind spot:
the proposer will learn to write models that score well without being
right, and by Phase 3 the core will be trained on that corruption.

So we take a model known to be correct, break it in specific ways, and
require detection every time. These mutations are ordered roughly by how
hard they are to catch — `CorruptOneCell` is the sharpest test, since a
verifier that catches a single wrong cell in a single frame has genuine
resolution.

The harness is generic: it wraps any WorldModel, so it keeps working as
models graduate from hand-written to LLM-written to core-generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable

from sentinel.env.types import GRID_SIZE, Action
from sentinel.wm.contract import ABSTAIN, Outcome, RenderedGrid, WorldModel


class Mutation(WorldModel):
    """Wraps a correct model and corrupts exactly one aspect of it."""

    label: str = "mutation"

    def __init__(self, base: WorldModel) -> None:
        self.base = base
        self.name = f"{base.name}+{self.label}"

    def init_state(self) -> Hashable:
        return self.base.init_state()

    def transition(self, state: Any, action: Action) -> Hashable:
        return self.base.transition(state, action)

    def render(self, state: Any) -> RenderedGrid:
        return self.base.render(state)

    def outcome(self, state: Any) -> Outcome:
        return self.base.outcome(state)

    def reset_to(self, state: Any) -> Hashable:
        return self.base.reset_to(state)


class CorruptOneCell(Mutation):
    """Change a single predicted cell in every frame.

    The sharpest test of verifier resolution. A model this close to correct
    would pass any coarse similarity check, but it is still wrong, and a
    planner simulating forward on it would eventually act on that wrongness.
    """

    label = "corrupt-one-cell"

    def __init__(self, base: WorldModel, x: int = 32, y: int = 32) -> None:
        super().__init__(base)
        self.x, self.y = x, y
        self.name = f"{base.name}+{self.label}({x},{y})"

    def render(self, state: Any) -> RenderedGrid:
        grid = [list(row) for row in self.base.render(state)]
        current = grid[self.y][self.x]
        if current == ABSTAIN:
            # Claiming a value where the base abstained is also a corruption:
            # it fabricates knowledge the model does not have.
            grid[self.y][self.x] = 0
        else:
            grid[self.y][self.x] = (current + 1) % 16
        return tuple(tuple(r) for r in grid)


class ShiftRender(Mutation):
    """Translate the whole rendered grid — a classic off-by-one."""

    label = "shift-render"

    def __init__(self, base: WorldModel, dx: int = 1, dy: int = 0) -> None:
        super().__init__(base)
        self.dx, self.dy = dx, dy
        self.name = f"{base.name}+{self.label}({dx},{dy})"

    def render(self, state: Any) -> RenderedGrid:
        src = self.base.render(state)
        out = []
        for y in range(GRID_SIZE):
            row = []
            for x in range(GRID_SIZE):
                sx, sy = x - self.dx, y - self.dy
                row.append(src[sy][sx] if 0 <= sx < GRID_SIZE and 0 <= sy < GRID_SIZE else 0)
            out.append(tuple(row))
        return tuple(out)


class FreezeState(Mutation):
    """Ignore actions entirely — the world never changes.

    Catches a verifier that is accidentally comparing a frame against
    itself instead of against the successor.
    """

    label = "freeze-state"

    def transition(self, state: Any, action: Action) -> Hashable:
        return state


class SwapActions(Mutation):
    """Two actions do each other's job — a plausible mislabelling."""

    label = "swap-actions"

    def __init__(self, base: WorldModel, a: int = 1, b: int = 2) -> None:
        super().__init__(base)
        self.a, self.b = a, b
        self.name = f"{base.name}+{self.label}({a}<->{b})"

    def transition(self, state: Any, action: Action) -> Hashable:
        if action.action_id == self.a:
            action = Action(self.b)
        elif action.action_id == self.b:
            action = Action(self.a)
        return self.base.transition(state, action)


class BlindOutcome(Mutation):
    """Never notice a level ending. Tests the outcome channel specifically."""

    label = "blind-outcome"

    def outcome(self, state: Any) -> Outcome:
        return Outcome.ONGOING


class LateTransition(Mutation):
    """Apply each action one step late — a stale-state bug.

    Renders correctly but always one step behind, which is exactly the
    failure mode of a model that updates its state after rendering rather
    than before.
    """

    label = "late-transition"

    def init_state(self) -> Hashable:
        return (self.base.init_state(), None)

    def transition(self, state: Any, action: Action) -> Hashable:
        inner, pending = state
        if pending is None:
            return (inner, action)
        return (self.base.transition(inner, pending), action)

    def render(self, state: Any) -> RenderedGrid:
        return self.base.render(state[0])

    def outcome(self, state: Any) -> Outcome:
        return self.base.outcome(state[0])

    def reset_to(self, state: Any) -> Hashable:
        return (self.base.reset_to(state[0]), None)


class CrashOnStep(Mutation):
    """Raise partway through. A crash is a verification result, not an escape."""

    label = "crash"

    def __init__(self, base: WorldModel, after: int = 5) -> None:
        super().__init__(base)
        self.after = after
        self._calls = 0
        self.name = f"{base.name}+{self.label}(after={after})"

    def transition(self, state: Any, action: Action) -> Hashable:
        self._calls += 1
        if self._calls > self.after:
            raise ValueError("injected fault")
        return self.base.transition(state, action)


@dataclass(frozen=True, slots=True)
class MutationSpec:
    """A named way to break a model."""

    label: str
    build: Callable[[WorldModel], WorldModel]
    targets: str
    """Which channel this corrupts: render, transition, or outcome."""


def standard_mutations(cell: tuple[int, int] = (32, 32)) -> list[MutationSpec]:
    """The battery the verifier must catch in full."""
    x, y = cell
    return [
        MutationSpec("corrupt-one-cell", lambda m: CorruptOneCell(m, x, y), "render"),
        MutationSpec("shift-render-x", lambda m: ShiftRender(m, 1, 0), "render"),
        MutationSpec("shift-render-y", lambda m: ShiftRender(m, 0, 1), "render"),
        MutationSpec("freeze-state", FreezeState, "transition"),
        MutationSpec("swap-actions", lambda m: SwapActions(m, 1, 2), "transition"),
        MutationSpec("late-transition", LateTransition, "transition"),
        MutationSpec("blind-outcome", BlindOutcome, "outcome"),
        MutationSpec("crash", lambda m: CrashOnStep(m, 5), "transition"),
    ]
