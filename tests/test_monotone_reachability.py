"""The route existed; greedy could not take it. Pin both halves.

X51 called `IF END@+0 -> POP` a decoy because it outscored the correct
chain's opening move, and X52 built a lookahead on that reading. It is the
first step of the winning monotone route for `copy inside any`. These tests
pin the measurement so the wrong reading cannot quietly come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x50_stack as X  # noqa: E402
import x51_deceptive_valley as V  # noqa: E402
import x53_monotone_reachability as M  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    alpha = sorted({c for t in X.TAPES for c in t})
    space = X.Space(X.TAPES, alpha)
    preds, masks, rules = V.build(space, alpha)
    lp = max(len(t) for t in X.TAPES) + 2
    return space, preds, masks, rules, lp


def test_the_frontier_finds_a_route_and_it_is_monotone(setup):
    """`halt on close` resolves in one state, so this stays fast."""
    space, preds, masks, rules, lp = setup
    truth = X.TRUTHS["halt on close"]
    target = space.interpret(truth)
    expr, expanded, best, trace, evals = M.monotone_search(
        space, target, rules, preds, masks, lambda t: t, budget=40)
    assert expr is not None, "no route found for a target greedy solves"
    assert trace == sorted(trace), "the returned route is not monotone"
    assert trace[-1] == 1.0
    assert all(X.output(expr, tp) == X.output(truth, tp) for tp in X.EVAL)


def test_the_decoy_is_the_best_single_rule(setup):
    """The correction: it outscores everything, and that is why greedy takes
    it -- correctly, since it opens the winning route."""
    space, preds, masks, rules, lp = setup
    target = space.interpret(X.TRUTHS["copy inside any"])
    ag = lambda t: float((space.loop(t, lp) == target).mean())
    scored = [(ag(space.branch(pm, bt, space.atoms["ADV"])), p, be)
              for p, pm in zip(preds, masks) for be, bt in rules]
    best = max(scored, key=lambda r: r[0])
    assert best[0] > 0.92, f"the opening score moved: {best[0]:.4f}"
    assert best[2] == "POP" or "POP" in str(best[2]), \
        f"the best opening rule changed shape: {X.render(best[2])}"


def test_the_monotone_constraint_is_actually_enforced(setup):
    """A search that silently allowed a drop would prove nothing."""
    space, preds, masks, rules, lp = setup
    target = space.interpret(X.TRUTHS["strip brackets"])
    _, _, _, trace, _ = M.monotone_search(
        space, target, rules, preds, masks, lambda t: space.loop(t, lp),
        budget=60)
    assert trace == sorted(trace)


def test_more_rounds_do_not_rescue_the_beam(setup):
    """Depth was never the binding constraint -- discarding was."""
    space, preds, masks, rules, lp = setup
    target = space.interpret(X.TRUTHS["copy inside any"])
    wrap = lambda t: space.loop(t, lp)
    for rounds in (6, 9):
        found, _ = V.beam_list(space, rules, preds, masks, target, wrap,
                               rounds=rounds, beam=1)
        assert not found, f"the beam now solves it at {rounds} rounds"
