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
from sentinel.gen.grid import AGENT, GATE_CLOSED, GATE_OPEN, TARGET, GridWorldModel
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec

from .hypothesis import (
    ScoredHypothesis,
    mechanics_from_classes,
    scorable_segment,
    score_hypothesis,
)

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
    steps_simulated: int = 0
    """Model steps actually run. The honest cost for incremental search,
    which never performs a whole replay."""
    survivors: int = 1
    """Hypotheses the evidence could not separate. This is the uncertainty,
    and it is what an experiment should be chosen to reduce."""

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

def equivalence_search(
    history: History,
    observed: LevelSpec,
    field_size: int,
    candidates: Sequence[tuple[int, ...]] | None = None,
) -> SearchResult:
    """Step every hypothesis forward together, and stop paying for the dead.

    Verifying candidates one at a time pays the FULL episode for each, even
    though most are refuted within a few moves. This walks the episode once
    with every hypothesis live at the same time, drops each the moment it
    mispredicts, and never simulates it again. Cost stops being
    "hypotheses x episode length" and becomes the sum over steps of however
    many are still standing -- which collapses quickly, because being wrong
    about movement shows up immediately.

    **Nothing is merged.** An earlier version also collapsed hypotheses
    whose predicted state matched, keeping one representative per behaviour.
    That took 5,760 candidates to 2 in a single step and was simply wrong:
    agreeing on step 1 does not mean agreeing on step 5, so the merge threw
    away hypotheses that later evidence would have separated -- including,
    as the check revealed, the true one. Prefix-equivalence is not
    equivalence. Refutation is the only sound pruning here.

    What it CANNOT do is separate hypotheses the evidence never separates.
    Survivors are returned as one equivalence class and the tie is broken by
    simplicity, exactly as elsewhere, so a rule the episode never exercised
    is still not invented.

    `steps_simulated` is the honest cost here, not `replays`: this never
    performs a full replay, so counting replays would flatter it.
    """
    pool = list(candidates) if candidates is not None else list(SIMPLICITY_ORDER)
    segment = scorable_segment(history)
    if not segment.steps:
        best = score_hypothesis(pool[0], history, observed, field_size)
        return SearchResult(best=best, replays=1, exhausted=True)

    # One model per hypothesis, all starting from the observed layout.
    live: list[tuple[tuple[int, ...], object, object]] = []
    for classes in pool:
        spec = WorldSpec(
            world_id="eq",
            seed=0,
            field_size=field_size,
            mechanics=mechanics_from_classes(classes),
            levels=(observed,),
        )
        model = GridWorldModel(spec)
        live.append((classes, model, model.init_state()))

    # Compare STATE, not rendered frames.
    #
    # Rendering 64x64 for every live candidate at every step is 4,096 cell
    # writes x 5,760 candidates x 30 steps, which does not finish. But the
    # render is a pure function of a handful of state variables, so two
    # candidates render identically exactly when those variables agree --
    # and the informative ones are the agent's position, which targets are
    # still out, and whether the gates are open. That is the same
    # observational-equivalence test at a thousandth of the cost.
    #
    # The agent OCCLUDES a target it stands on, so a target under the agent
    # is invisible and must be excluded before comparing.
    def observed_facts(grid):
        agent = None
        targets = set()
        gates_open = None
        for y in range(field_size):
            row = grid[y]
            for x in range(field_size):
                v = row[x]
                if v == AGENT:
                    agent = (x, y)
                elif v == TARGET:
                    targets.add((x, y))
                elif v == GATE_OPEN:
                    gates_open = True
                elif v == GATE_CLOSED:
                    gates_open = False
        return agent, frozenset(targets), gates_open

    steps_simulated = 0
    for step in segment.steps:
        if len(live) <= 1:
            break
        want_agent, want_targets, want_gates = observed_facts(step.settled.grid)
        survivors = []

        for classes, model, state in live:
            steps_simulated += 1
            try:
                nxt = model.transition(state, step.action)
            except Exception:
                continue
            here = (nxt.x, nxt.y)
            visible = frozenset(t for t in nxt.remaining if t != here)
            if here != want_agent or visible != want_targets:
                continue
            if want_gates is not None and bool(nxt.gates_open) != want_gates:
                continue
            survivors.append((classes, model, nxt))

        if not survivors:
            break
        live = survivors

    remaining = [classes for classes, _, _ in live] or [pool[0]]
    winner = min(remaining, key=lambda c: (sum(c), c))
    best = score_hypothesis(winner, history, observed, field_size)
    return SearchResult(
        best=best,
        replays=max(1, steps_simulated // max(1, len(segment.steps))),
        exhausted=False,
        steps_simulated=steps_simulated,
        survivors=len(remaining),
    )


def version_space_search(
    history: History,
    observed: LevelSpec,
    field_size: int,
    core=None,
    candidates: Sequence[tuple[int, ...]] | None = None,
) -> SearchResult:
    """Refute in bulk, then let the prior choose among what is left.

    This is the architecture the plan describes, finally in one function:
    the verifier decides what is POSSIBLE and the network decides which
    possibility to believe. Stepping every hypothesis forward together and
    dropping each as it mispredicts costs 0.04s against 47s for verifying
    candidates one at a time -- 1,162x -- and keeps the true rule set in the
    surviving set 97% of the time.

    Selection is the part that has to be right. Taking the SIMPLEST survivor
    is sound but weak: it scores 25.0% solve against exhaustive search's
    55.6%, because roughly fifty hypotheses typically survive and simplicity
    is a poor way to choose among them. Ranking the survivors with the core
    is what the core is actually for, and it costs one forward pass rather
    than fifty replays.
    """
    from sentinel.explore.version_space import VersionSpace

    segment = scorable_segment(history)
    space = VersionSpace.over(observed, field_size, candidates)
    for step in segment.steps:
        space.observe(step.action, step.settled)

    survivors = space.candidates()
    if not survivors:
        survivors = [min(candidates or SIMPLICITY_ORDER, key=lambda c: (sum(c), c))]

    if core is not None and len(survivors) > 1:
        ranked = core_order(core, history, candidates=survivors)
        winner = ranked[0]
    else:
        winner = min(survivors, key=lambda c: (sum(c), c))

    best = score_hypothesis(winner, history, observed, field_size)
    return SearchResult(
        best=best,
        replays=1,
        exhausted=False,
        steps_simulated=len(segment.steps) * len(space),
        survivors=len(survivors),
    )
