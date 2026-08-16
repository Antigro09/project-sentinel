"""The exact world model of ToyWorld.

Correct by construction: it calls the same `step_state` the world does. That
is deliberate. A verifier gate needs a subject whose correctness is not
itself in question — otherwise a mutation that goes undetected is ambiguous
between "the verifier is blind" and "the model was already wrong".

Note that this model tracks `charge`, a variable that appears nowhere in
the rendered grid. That is the point. A model restricted to visible state
would mispredict every third move, and the verifier should catch it — which
`GridOnlyToyModel` below exists to demonstrate.
"""

from __future__ import annotations

from typing import Any, Hashable

from sentinel.env.types import Action
from sentinel.wm.contract import ABSTAIN, Outcome, RenderedGrid, WorldModel

from .toy import (
    CHARGE_PERIOD,
    Level,
    ToyState,
    default_levels,
    initial_state,
    render_state,
    step_state,
)


class ToyModel(WorldModel):
    """A complete, correct hypothesis about ToyWorld."""

    name = "toy-exact"

    def __init__(self, levels: tuple[Level, ...] | None = None) -> None:
        self.levels = levels or default_levels()
        self._level_index = 0

    def init_state(self) -> Hashable:
        return initial_state(self._level_index, self.levels)

    def transition(self, state: Any, action: Action) -> Hashable:
        return step_state(state, action, self.levels)

    def render(self, state: Any) -> RenderedGrid:
        return render_state(state, self.levels)

    def outcome(self, state: Any) -> Outcome:
        if state.dead:
            return Outcome.GAME_OVER
        if state.cleared:
            return Outcome.LEVEL_COMPLETE
        return Outcome.ONGOING

    def available_actions(self, state: Any) -> tuple[int, ...]:
        from .toy import LEGAL_ACTIONS

        return LEGAL_ACTIONS

    def reset_to(self, state: Any) -> Hashable:
        """Advance to the next level after a boundary."""
        nxt = min(state.level + 1, len(self.levels) - 1)
        return initial_state(nxt, self.levels)


class GridOnlyToyModel(ToyModel):
    """A model that refuses to posit hidden state.

    Identical to ToyModel except it assumes every move travels exactly one
    cell — i.e. it models only what the grid shows. It is right most of the
    time and wrong every third move.

    This is the single most important negative control in the project. The
    entire architecture bet rests on the claim that understanding requires
    positing unobserved structure. If the verifier cannot tell this model
    apart from the correct one, that claim is untestable and the reward
    signal is blind to the exact thing it exists to measure.
    """

    name = "toy-grid-only"

    def transition(self, state: Any, action: Action) -> Hashable:
        # Pin charge so the two-cell move never triggers.
        pinned = ToyState(
            level=state.level,
            x=state.x,
            y=state.y,
            remaining=state.remaining,
            charge=1 % CHARGE_PERIOD,
            dead=state.dead,
            cleared=state.cleared,
        )
        result = step_state(pinned, action, self.levels)
        return ToyState(
            level=result.level,
            x=result.x,
            y=result.y,
            remaining=result.remaining,
            charge=state.charge + 1,
            dead=result.dead,
            cleared=result.cleared,
        )
