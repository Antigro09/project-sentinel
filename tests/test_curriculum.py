"""Difficulty is not rule count, which is why the curriculum band is thin.

X55 generated 22 random 1-7 rule targets and only one expanded more than 19
states. The reason is that a randomly assembled seven-rule decision list is
usually behaviourally identical to a much shorter program, so the frontier
resolves it instantly. These tests pin that -- if random targets ever stop
collapsing, the survey's conclusion needs revisiting.
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
import x55_curriculum as C  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    alpha = sorted({c for t in X.TAPES for c in t})
    space = X.Space(X.TAPES, alpha)
    _, _, rules = V.build(space, alpha)
    keep = [("AT", 0, c) for c in list(alpha) + ["$"]] + [("EMPTY",)]
    keep += [("TOP", c) for c in "[("]
    lp = max(len(t) for t in X.TAPES) + 2
    return space, keep, [space.pred(p) for p in keep], rules, lp


def test_long_random_lists_collapse_onto_their_own_prefixes(setup):
    """The mechanism behind the thin band, measured directly.

    A seven-rule random decision list is behaviourally identical to one of its
    own shorter prefixes about half the time, so the frontier resolves it as
    fast as the short one. Rule count is not difficulty.
    """
    space, preds, masks, rules, lp = setup
    wrap = lambda t: space.loop(t, lp)
    rng = np.random.default_rng(3)
    collapsed = 0
    for _ in range(30):
        chain = [[preds[int(rng.integers(0, len(preds)))],
                  rules[int(rng.integers(0, len(rules)))][0]] for _ in range(7)]
        dflt = rules[int(rng.integers(0, len(rules)))][0]
        full = wrap(space.table(X._join(chain, dflt)))
        if any(np.array_equal(full, wrap(space.table(X._join(chain[:k], dflt))))
               for k in range(1, 7)):
            collapsed += 1
    assert collapsed >= 10, \
        f"only {collapsed}/30 collapsed; the thin-band explanation changed"


def test_the_derived_tests_are_load_bearing(setup):
    """`copy inside any` sits in the band only because of them.

    Without the derived tests the frontier does not solve it at all inside 200
    expanded states -- so the difficulty band and the derivation engine are
    not independent parts of the result.
    """
    space, preds, masks, rules, lp = setup
    wrap = lambda t: space.loop(t, lp)
    target = space.interpret(X.TRUTHS["copy inside any"])
    expr, expanded, _, _, _ = R.frontier(
        space, target, rules, preds, masks, wrap, budget=200)
    assert expr is None, "it now solves without derived tests; re-read X55"

    dv = X.derive(space, target, X.families(space,
                                            sorted({c for t in X.TAPES for c in t})))
    preds2 = list(preds) + [d[1] for d in dv]
    masks2 = [space.pred(p) for p in preds2]
    expr2, expanded2, _, _, _ = R.frontier(
        space, target, rules, preds2, masks2, wrap, budget=140)
    assert expr2 is not None, "derived tests no longer make it reachable"
    assert expanded2 >= C.BAND[0], f"expanded {expanded2}; the band moved"


def test_random_target_never_returns_a_real_target_unchecked(setup):
    """The generator must be capable of producing a duplicate, so that the
    experiment's duplicate filter is doing real work rather than decoration."""
    space, preds, masks, rules, lp = setup
    wrap = lambda t: space.loop(t, lp)
    rng = np.random.default_rng(1)
    body, tab, n = C.random_target(space, rules, preds, masks, wrap, rng)
    assert 1 <= n <= 7
    assert tab.shape == space.interpret(X.TRUTHS["copy all"]).shape
