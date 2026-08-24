"""Greedy agreement points away from the answer, and by how much.

X50 could not account for missing `copy inside any`. The reason is that the
correct chain is DOMINATED -- below what a single rule scores on its own --
for three consecutive rounds. These pin the measurement, because it is the
one thing that explains the miss and the one thing a later change might
accidentally erase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x50_stack as X  # noqa: E402
import x51_deceptive_valley as V  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    alpha = sorted({c for t in X.TAPES for c in t})
    space = X.Space(X.TAPES, alpha)
    preds, masks, rules = V.build(space, alpha)
    lp = max(len(t) for t in X.TAPES) + 2
    return space, preds, masks, rules, lp


def _ceiling(space, masks, rules, target, lp):
    return max(float((space.loop(space.branch(pm, bt, space.atoms["ADV"]), lp)
                      == target).mean())
               for pm in masks for _, bt in rules)


def test_the_decoy_beats_the_correct_opening_move(setup):
    space, preds, masks, rules, lp = setup
    beaten = []
    for name in ("strip brackets", "copy inside any", "copy inside []"):
        truth = X.TRUTHS[name]
        target = space.interpret(truth)
        chain, _ = X._split(truth[1])
        first = float((space.loop(space.table(X._join(chain[:1], "ADV")), lp)
                       == target).mean())
        if _ceiling(space, masks, rules, target, lp) > first:
            beaten.append(name)
    assert len(beaten) == 3, f"valley no longer present for {beaten}"


def test_the_correct_chain_is_dominated_for_three_rounds(setup):
    """The number that explains the miss."""
    space, preds, masks, rules, lp = setup
    truth = X.TRUTHS["copy inside any"]
    target = space.interpret(truth)
    ceiling = _ceiling(space, masks, rules, target, lp)
    atomic = [(("AT", 0, "["), X.PUSH_ON), (("AT", 0, "("), X.PUSH_ON),
              (("AT", 0, "]"), X.POP_ON), (("AT", 0, ")"), X.POP_ON),
              (("EMPTY",), "ADV")]
    scores = [float((space.loop(space.table(X._join(atomic[:i], X.EMIT_ON)), lp)
                     == target).mean()) for i in range(len(atomic) + 1)]
    assert scores[-1] == 1.0, "the chain does not actually reach the target"
    below = sum(1 for s in scores if s < ceiling)
    assert below >= 3, f"only {below} rounds dominated; the explanation changed"
    assert scores == sorted(scores), "the chain should climb, just too slowly"


def test_eqtop_binds_a_variable_rather_than_naming_a_letter():
    """It did not earn its place, but it must at least mean what it says."""
    for ch in "[(a":
        st = X.St(pos=0, stack=(ch,))
        assert X.test_pred(("EQTOP", 0), ch + "zz", st)
        assert not X.test_pred(("EQTOP", 0), "z" + ch, st)
    assert not X.test_pred(("EQTOP", 0), "abc", X.St(pos=0)), "empty stack"


def test_polish_plus_leaves_a_clean_program_alone(setup):
    """No large test in the chain means nothing to insert against."""
    space, preds, masks, rules, lp = setup
    truth = X.TRUTHS["strip brackets"]
    target = space.interpret(truth)
    wrap = lambda t: space.loop(t, lp)
    out, added = V.polish_plus(space, truth[1], preds, masks, rules, target, wrap)
    assert not added
    assert np.array_equal(wrap(space.table(out)), target)
