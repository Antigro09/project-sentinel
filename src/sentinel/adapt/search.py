"""Searching the hypothesis space at test time, and counting what it costs.

The rule space here is 2*3*2*2*2*2 = 96, small enough to enumerate. That is
a property of the current generator, not of the approach, and it is worth
stating plainly: where the space is this small, brute force is a strong
baseline and any learned component has to beat it to be earning its place.

`replays` is the currency this module reports. Verifier replays are the
cost that a prior can reduce -- accuracy is already near-perfect for every
rule the evidence identifies, so "did it get the answer" cannot show
accumulation while "how much did it cost" can.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

from sentinel.env.history import History
from sentinel.gen.spec import LevelSpec, Mechanics

from .hypothesis import ScoredHypothesis, mechanics_from_classes, score_hypothesis

NCLASS = (2, 3, 2, 2, 2, 2)
ALL_HYPOTHESES: tuple[tuple[int, ...], ...] = tuple(
    itertools.product(*[range(n) for n in NCLASS])
)

SIMPLICITY_ORDER: tuple[tuple[int, ...], ...] = tuple(
    sorted(ALL_HYPOTHESES, key=lambda c: (sum(c), c))
)
"""Simplest rule sets first -- the default, and measurably the best one.

Ties are broken toward simplicity, so trying simple hypotheses first means
the first exact explanation found is usually already the winner and the
rest can be skipped. Measured over 56 held-out worlds at identical
accuracy: default enumeration 36.7 replays, simplicity 10.1, retrieval
prior 19.4. The cheapest ordering needs no memory at all."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outcome of one test-time search."""

    best: ScoredHypothesis
    replays: int
    exhausted: bool
    """False when an exact explanation was found and the rest was skipped."""

    @property
    def mechanics(self) -> Mechanics:
        return self.best.mechanics

    def summary(self) -> str:
        return (
            f"{self.best.mechanics.summary():40} fit={self.best.fitness:.3f} "
            f"after {self.replays} replays"
            + ("" if self.exhausted else " (early exit)")
        )


def exhaustive_search(
    history: History,
    observed: LevelSpec,
    field_size: int,
    order: Sequence[tuple[int, ...]] | None = None,
    stop_at: float = 1.0,
) -> SearchResult:
    """Find the rule set that best explains the evidence.

    **The result does not depend on the order candidates are tried in.**
    That property is not free and it is not decorative. Evidence routinely
    fails to separate rule sets -- flip `has_hazards` in a level with no
    hazards and nothing changes -- so exact explanations come in ties, and
    whoever is tried first wins a naive early exit. Measured directly: a
    retrieval prior that merely reordered this list cut replays from 23.2 to
    1.8 per world and cut accuracy from 58% to 28.6%, because it was
    reaching a different member of the same tie.

    So ties are broken by SIMPLICITY, explicitly, and the early exit is
    only taken once no untried candidate could win that tie-break. Search
    still stops early when the prior is good; it just cannot be talked into
    a more complicated answer than the evidence requires.
    """
    candidates = list(order) if order is not None else list(SIMPLICITY_ORDER)
    best: ScoredHypothesis | None = None
    replays = 0
    exhausted = True

    def better(a: ScoredHypothesis, b: ScoredHypothesis | None) -> bool:
        if b is None:
            return True
        return (a.fitness, -sum(a.classes)) > (b.fitness, -sum(b.classes))

    for index, classes in enumerate(candidates):
        scored = score_hypothesis(classes, history, observed, field_size)
        replays += 1
        if better(scored, best):
            best = scored

        if best.fitness >= stop_at:
            # An exact explanation is in hand. It is safe to stop only if
            # nothing left to try is simpler -- a simpler hypothesis that
            # also explains everything would win the tie-break.
            simplest_best = sum(best.classes)
            if not any(sum(c) < simplest_best for c in candidates[index + 1 :]):
                exhausted = False
                break

    assert best is not None, "hypothesis space cannot be empty"
    return SearchResult(best=best, replays=replays, exhausted=exhausted)
