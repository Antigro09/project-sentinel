"""Mechanic hypotheses, and scoring them against what actually happened.

At test time there are no labels. The core's guess about an unfamiliar
world cannot be checked against ground truth, because ground truth is the
thing being inferred. What *is* available is the verifier: build the world
model each hypothesis implies, replay the episode through it, and see which
one explains the frames that were actually observed.

That makes hypothesis quality measurable without supervision, which is the
whole basis of Phase 4. Note what it is not: the verifier does not say
which hypothesis is *true*, only which is consistent with the evidence so
far. Two rule sets that differ solely in behaviour the episode never
exercised will score identically, and the honest response to that is to go
and exercise it -- which is why `ordered_targets` needed probing before it
was learnable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations

import numpy as np

from sentinel.core.encoding import HEADS, MechanicLabels
from sentinel.env.boundary import longest_run
from sentinel.env.history import History
from sentinel.gen.grid import GridWorldModel
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec
from sentinel.verify.verifier import verify

HEAD_ORDER: tuple[str, ...] = tuple(name for name, _ in HEADS)
NCLASS: tuple[int, ...] = tuple(n for _, n in HEADS)

MAX_ORDER_SEARCH = 4
"""Permutations are factorial; 4 targets is 24 replays, 6 would be 720."""


def mechanics_from_classes(classes: list[int] | tuple[int, ...] | np.ndarray) -> Mechanics:
    """Turn one class per head into the rule set it denotes."""
    return MechanicLabels(*[int(v) for v in classes]).to_mechanics()


def classes_from_mechanics(mech: Mechanics) -> tuple[int, ...]:
    """Inverse of `mechanics_from_classes`."""
    return tuple(int(v) for v in MechanicLabels.from_mechanics(mech).as_array())


@dataclass(frozen=True, slots=True)
class ScoredHypothesis:
    """One candidate rule set and how well it explained the evidence."""

    classes: tuple[int, ...]
    mechanics: Mechanics
    fitness: float
    transition_match: float
    explained_prefix: int
    crashed: bool
    target_order: tuple = ()
    """Sequence that scored best. Meaningless unless `ordered_targets`."""

    def summary(self) -> str:
        return (
            f"{self.mechanics.summary():40} fit={self.fitness:.3f} "
            f"match={self.transition_match:.2f} prefix={self.explained_prefix}"
        )


def scorable_segment(history: History) -> History:
    """The longest stretch of an episode a one-level hypothesis can be judged on.

    A hypothesis is built from a layout read off ONE frame, so it describes
    one level. Replaying a multi-level episode through it charges it for
    frames showing a board it was never given, and the truth then scores
    worse than a lie. Measured on 12 held-out worlds before this existed,
    wrong rule sets outscored the true ones on 3 of them.

    Boundaries come from `env.boundary`, which uses `levels_completed`
    rather than `level_index` -- see that module for why the obvious check
    silently misses the crossing step.
    """
    frame, steps = longest_run(history)
    cut = History(
        game_id=history.game_id,
        seed=history.seed,
        initial=frame,
        steps=list(steps),
    )
    return cut


def score_hypothesis(
    classes: tuple[int, ...],
    history: History,
    observed: LevelSpec,
    field_size: int,
    world_id: str = "hypothesis",
) -> ScoredHypothesis:
    """Replay a real episode through the model a hypothesis implies.

    The layout comes from observation, so the only thing under test is the
    *mechanics*. That separation is deliberate: a hypothesis that scores
    badly is being told its dynamics are wrong, not that it misread the
    board.
    """
    mech = mechanics_from_classes(classes)
    return score_mechanics(mech, history, observed, field_size, world_id, classes)


def score_mechanics(
    mech: Mechanics,
    history: History,
    observed: LevelSpec,
    field_size: int,
    world_id: str = "hypothesis",
    classes: tuple[int, ...] | None = None,
) -> ScoredHypothesis:
    """Score a rule set directly.

    The six-head class encoding only spans the narrow 26-combination space.
    The compositional space has no such encoding, so search there works in
    `Mechanics` objects and this is the entry point it uses.
    """
    history = scorable_segment(history)

    # Under ordered rules the REQUIRED SEQUENCE is part of the hypothesis,
    # not something the layout reveals. A frame shows which cells hold
    # targets and never which must come first, so `read_layout` returns
    # them in raster order -- an arbitrary guess that is usually wrong.
    # Scoring the true mechanics against an arbitrary order charges them
    # for a mistake they did not make: measured, that alone made the truth
    # lose to a wrong rule set on 2 of 12 held-out worlds.
    #
    # So the order is searched. This does give ordered hypotheses more
    # freedom than unordered ones, which is a real asymmetry and the reason
    # ordered/unordered can only be told apart by evidence that actually
    # contains a FAILED collection.
    orders: list[tuple] = [observed.targets]
    if mech.ordered_targets and 2 <= len(observed.targets) <= MAX_ORDER_SEARCH:
        orders = list(permutations(observed.targets))

    best = None
    for order in orders:
        level = replace(observed, targets=tuple(order))
        spec = WorldSpec(
            world_id=world_id,
            seed=0,
            field_size=field_size,
            mechanics=mech,
            levels=(level,),
        )
        report = verify(GridWorldModel(spec), history)
        candidate = ScoredHypothesis(
            classes=tuple(int(c) for c in classes) if classes is not None else (),
            mechanics=mech,
            fitness=report.fitness,
            transition_match=report.transition_match,
            explained_prefix=report.explained_prefix,
            crashed=report.crashed,
            target_order=tuple(order),
        )
        if best is None or (candidate.fitness, candidate.explained_prefix) > (
            best.fitness,
            best.explained_prefix,
        ):
            best = candidate
        if best.fitness >= 1.0:
            break
    assert best is not None
    return best
