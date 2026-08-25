"""Agreement ordering is nearly the whole of X53's result. Pin that.

X54 tried three ways to train a ranker for the frontier queue and could not
get data from any of them. The positive finding underneath is that ordering
by agreement is doing almost all the work -- random ordering solves nothing
at the same budget -- so a later change that quietly swaps the ordering would
break X53's 5/5 without breaking any other test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x50_stack as X  # noqa: E402
import x51_deceptive_valley as V  # noqa: E402
import x54_frontier_ranker as R  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    alpha = sorted({c for t in X.TAPES for c in t})
    space = X.Space(X.TAPES, alpha)
    _, _, rules = V.build(space, alpha)
    keep = [("AT", 0, c) for c in list(alpha) + ["$"]] + [("EMPTY",)]
    keep += [("TOP", c) for c in "[("]
    lp = max(len(t) for t in X.TAPES) + 2
    return space, keep, [space.pred(p) for p in keep], rules, lp


def test_agreement_ordering_solves_in_a_handful_of_states(setup):
    space, preds, masks, rules, lp = setup
    truth = X.TRUTHS["strip brackets"]
    target = space.interpret(truth)
    expr, expanded, evals, moves, _ = R.frontier(
        space, target, rules, preds, masks, lambda t: space.loop(t, lp),
        budget=20)
    assert expr is not None
    # The experiment reports 1 expanded state for this target; it also
    # supplies the derived tests, which this fixture does not. Without them
    # it takes 4. The bound is loose on purpose -- what matters is that it is
    # a handful and not the hundreds the random arm cannot finish in.
    assert expanded <= 6, f"took {expanded} states; ordering may have changed"
    assert all(X.output(("LOOP", expr), tp) == X.output(truth, tp)
               for tp in X.EVAL)


def test_random_ordering_solves_nothing_at_the_same_budget(setup):
    """The control that makes the agreement result mean something."""
    space, preds, masks, rules, lp = setup
    rng = np.random.default_rng(0)
    target = space.interpret(X.TRUTHS["strip brackets"])
    expr, expanded, evals, _, _ = R.frontier(
        space, target, rules, preds, masks, lambda t: space.loop(t, lp),
        budget=20, priority=lambda t, wt, nr, a: float(rng.random()))
    assert expr is None, "random ordering now solves it; the comparison changed"


def test_a_bad_ordering_cannot_cost_correctness(setup):
    """Ordering is speed only -- the monotone constraint and the exact-match
    test are what decide, so a wrong order must never yield a wrong program."""
    space, preds, masks, rules, lp = setup
    rng = np.random.default_rng(1)
    truth = X.TRUTHS["copy all"]
    target = space.interpret(truth)
    for pri in (None, lambda t, wt, nr, a: float(rng.random())):
        expr, _, _, _, _ = R.frontier(
            space, target, rules, preds, masks, lambda t: space.loop(t, lp),
            budget=10, priority=pri)
        if expr is not None:
            assert all(X.output(("LOOP", expr), tp) == X.output(truth, tp)
                       for tp in X.EVAL)
