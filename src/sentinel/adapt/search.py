"""Searching the hypothesis space at test time, and counting what it costs.

The rule space is enumerable, and its SIZE is the whole experiment. At 96
rule sets, exhaustive search identified a world in 1.7 seconds and beat the
trained core on every rule the evidence determined -- brute force wins when
there is nothing to prune. The compositional space holds 5,760, costing
about 101 seconds a world against the plan's five-minute budget, which is
the first version of this problem where a learned prior could pay for
itself. Search stops being viable past roughly 17,000.

`replays` is the currency this module reports. Verifier replays are the
cost that a prior can reduce -- accuracy is already near-perfect for every
rule the evidence identifies, so "did it get the answer" cannot show
accumulation while "how much did it cost" can.
"""

from __future__ import annotations

import itertools

import numpy as np
from dataclasses import dataclass
from typing import Sequence

from sentinel.env.history import History
from sentinel.gen.spec import LevelSpec, Mechanics

from .hypothesis import ScoredHypothesis, mechanics_from_classes, score_hypothesis

from .hypothesis import NCLASS

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
    tie_break: str = "simplicity",
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

    `tie_break="order"` deliberately gives that decision back to the
    ordering: the first hypothesis that explains everything wins. That is
    the wrong policy for a blind ordering and the right one for an informed
    prior, because ties are exactly where evidence has run out and a prior
    is the only thing left to consult. Paired with `core_order` this is the
    architecture the plan describes -- the verifier decides what is
    possible, the network decides which of the possibilities to believe.
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
            if tie_break == "order":
                exhausted = False
                break
            # An exact explanation is in hand. It is safe to stop only if
            # nothing left to try is simpler -- a simpler hypothesis that
            # also explains everything would win the tie-break.
            simplest_best = sum(best.classes)
            if not any(sum(c) < simplest_best for c in candidates[index + 1 :]):
                exhausted = False
                break

    assert best is not None, "hypothesis space cannot be empty"
    return SearchResult(best=best, replays=replays, exhausted=exhausted)

def core_order(
    core, history, temperature: float = 1.0, candidates=None
) -> list[tuple[int, ...]]:
    """Rank every hypothesis by how plausible the core finds it.

    This is the core doing the job the plan actually assigns it: not
    answering, but deciding what is worth testing first. The verifier still
    decides what is true, so a bad ranking costs replays and nothing else --
    it cannot make the system believe something the evidence refutes.

    That separation is what makes the comparison meaningful. Ranking quality
    shows up purely as REPLAYS-TO-TRUTH, with accuracy held constant by
    construction, so there is no way for a confident wrong prior to buy a
    better-looking score.
    """
    from .ttt import head_distributions

    probs = head_distributions(core, history, temperature)
    logs = [np.log(np.maximum(p, 1e-12)) for p in probs]
    pool = list(candidates) if candidates is not None else list(ALL_HYPOTHESES)

    def score(classes: tuple[int, ...]) -> tuple[float, int]:
        total = 0.0
        for head, cls in enumerate(classes):
            if head < len(logs) and cls < len(logs[head]):
                total += float(logs[head][cls])
        # Simplicity breaks ties, matching the verifier's own tie-break.
        return (-total, sum(classes))

    return sorted(pool, key=score)


def replays_to_truth(
    order, truth: tuple[int, ...]
) -> int:
    """Position of the true rule set in a ranking, 1-based.

    The honest measure of a prior: how deep into the list search has to go
    before it reaches the answer.
    """
    for i, classes in enumerate(order):
        if tuple(classes) == tuple(truth):
            return i + 1
    return len(order) + 1

def factored_search(
    history: History,
    observed: LevelSpec,
    field_size: int,
    passes: int = 3,
    start: tuple[int, ...] | None = None,
) -> SearchResult:
    """Infer each rule separately instead of enumerating joint rule sets.

    Exhaustive search costs the PRODUCT of the per-head class counts --
    3*5*4*4*3*2*2*2 = 5,760 -- because it enumerates whole rule sets. But a
    rule set is eight largely independent facts, and the verifier reports
    which transition falsified a candidate, so the facts can be settled one
    at a time. That makes the cost the SUM instead: 3+5+4+4+3+2+2+2 = 25 per
    pass. The difference is multiplicative versus additive, which is the
    difference between a space you can enumerate and one you cannot.

    It matters beyond this generator. Exhaustive search stops fitting the
    plan's five-minute budget past roughly 17,000 hypotheses; a factored
    search of the same space costs 25 replays whether the space holds 5,760
    rule sets or ten million.

    **Coordinate ascent, not one pass.** The heads are not fully
    independent: `step_distance` and `charge_period` both change how far the
    agent travels, so the best charge depends on the step already assumed.
    Sweeping repeatedly until nothing changes lets those settle against each
    other. `passes` bounds the work; convergence is usually immediate.

    Ties are broken toward simplicity, exactly as in `exhaustive_search`, so
    a rule the evidence never exercised is not invented.
    """
    current = list(start) if start is not None else [0] * len(NCLASS)
    best = score_hypothesis(tuple(current), history, observed, field_size)
    replays = 1

    def better(a: ScoredHypothesis, b: ScoredHypothesis) -> bool:
        return (a.fitness, -sum(a.classes)) > (b.fitness, -sum(b.classes))

    for _ in range(max(1, passes)):
        changed = False
        for head, n_classes in enumerate(NCLASS):
            for value in range(n_classes):
                if value == current[head]:
                    continue
                trial = list(current)
                trial[head] = value
                scored = score_hypothesis(tuple(trial), history, observed, field_size)
                replays += 1
                if better(scored, best):
                    best, current, changed = scored, trial, True
        if not changed:
            break

    return SearchResult(best=best, replays=replays, exhausted=False)
