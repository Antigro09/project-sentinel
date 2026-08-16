"""Reference models: the ceiling and the floor.

Neither of these is intelligent. They exist to bound the scale that real
induced models are measured on, and to give the verifier's own tests a
subject that is correct by construction.

`OracleModel` is worth dwelling on, because it is the whole thesis of this
project in miniature. It scores *perfectly* on any history it was built
from and is *useless* on any other, because it does not model anything —
it memorises the answer. That is precisely the failure the program is
designed around: a system can look flawless on data it has seen and have
understood nothing. Every generalization metric in this codebase exists to
tell OracleModel apart from a model that actually knows how the world
works, and the moment those metrics stop being able to, they are broken.
"""

from __future__ import annotations

from typing import Any, Hashable

from sentinel.env.history import History
from sentinel.env.types import Action, Observation
from sentinel.verify.verifier import observed_outcome

from .contract import Outcome, RenderedGrid, WorldModel, full_abstain, grid_to_rendered


class OracleModel(WorldModel):
    """Replays a recorded history exactly. Correct by construction, general never.

    Use as the upper bound for verifier tests: any mutation of this model
    must be detected, because the unmutated version is known-perfect.
    """

    def __init__(self, history: History, name: str = "oracle") -> None:
        self.name = name
        self._frames: list[Observation] = [history.initial] + [
            s.settled for s in history.steps
        ]

    def _at(self, index: int) -> Observation:
        return self._frames[min(index, len(self._frames) - 1)]

    def init_state(self) -> Hashable:
        return 0

    def transition(self, state: Any, action: Action) -> Hashable:
        return int(state) + 1

    def render(self, state: Any) -> RenderedGrid:
        return grid_to_rendered(self._at(int(state)).grid)

    def outcome(self, state: Any) -> Outcome:
        i = int(state)
        return observed_outcome(self._at(i), self._frames[i - 1] if i > 0 else None)

    def reset_to(self, state: Any) -> Hashable:
        # The index keeps advancing: this model tracks position in a tape,
        # not position in a world, so a reset does not rewind it.
        return state


class AbstainModel(WorldModel):
    """Predicts nothing at all. The honest null model.

    Scores accuracy 1.0 (it is never wrong) at coverage 0.0 (it never
    speaks). Any scoring scheme that ranks this alongside a real model has
    collapsed its metrics and needs fixing.
    """

    name = "abstain"

    def init_state(self) -> Hashable:
        return 0

    def transition(self, state: Any, action: Action) -> Hashable:
        return int(state) + 1

    def render(self, state: Any) -> RenderedGrid:
        return full_abstain()

    def outcome(self, state: Any) -> Outcome:
        return Outcome.ONGOING

    def reset_to(self, state: Any) -> Hashable:
        return state


class StaticModel(WorldModel):
    """Predicts the opening frame forever. The laziest non-trivial hypothesis.

    A useful floor above AbstainModel: it has full coverage and is right
    about every cell that never changes, which in a sparse grid world is
    most of them. Any induced model that cannot beat this has learned
    nothing about the environment's *dynamics*, only its wallpaper.
    """

    def __init__(self, first: Observation, name: str = "static") -> None:
        self.name = name
        self._grid = grid_to_rendered(first.grid)

    def init_state(self) -> Hashable:
        return 0

    def transition(self, state: Any, action: Action) -> Hashable:
        return int(state) + 1

    def render(self, state: Any) -> RenderedGrid:
        return self._grid

    def outcome(self, state: Any) -> Outcome:
        return Outcome.ONGOING

    def reset_to(self, state: Any) -> Hashable:
        return state
