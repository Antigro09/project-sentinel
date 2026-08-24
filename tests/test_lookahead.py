"""Lookahead lost a target greedy solves. Pin the measurement that says why.

X52's negative result is easy to erase by accident: someone re-reads the
docstring, assumes an optimistic completion bound must help, and turns it back
on. These pin the two facts that make it not help -- the decoy leads on every
component of the signature, and the reference chain is not what the search
follows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x50_stack as X  # noqa: E402
import x51_deceptive_valley as V  # noqa: E402
import x52_lookahead as L  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    alpha = sorted({c for t in X.TAPES for c in t})
    space = X.Space(X.TAPES, alpha)
    preds, masks, rules = V.build(space, alpha)
    lp = max(len(t) for t in X.TAPES) + 2
    return space, preds, masks, rules, lp


def test_the_decoy_leads_on_the_state_component_too(setup):
    """Kills the "reweight the signature" idea before anyone rebuilds it."""
    space, preds, masks, rules, lp = setup
    truth = X.TRUTHS["copy inside any"]
    target = space.interpret(truth)
    e_t, _, _ = space.unpack(target)

    def end_agreement(sig):
        e, _, _ = space.unpack(space.loop(sig, lp))
        return float((e == e_t).mean())

    one_rule = space.table(X._join([[("AT", 0, "["), X.PUSH_ON]], X.EMIT_ON))
    best_decoy = max(end_agreement(space.branch(pm, bt, space.atoms["ADV"]))
                     for pm in masks for _, bt in rules)
    assert best_decoy > end_agreement(one_rule), \
        "the decoy no longer leads on end-state; the X52 reasoning changed"


def test_the_reference_chain_is_not_the_path_the_search_takes(setup):
    """`copy inside []` is recovered although its own chain is non-monotone."""
    space, preds, masks, rules, lp = setup
    truth = X.TRUTHS["copy inside []"]
    target = space.interpret(truth)
    ag = lambda t: float((space.loop(t, lp) == target).mean())
    chain, default = X._split(truth[1])
    scores = [ag(space.table(X._join(chain[:i], default)))
              for i in range(len(chain) + 1)]
    ceiling = max(ag(space.branch(pm, bt, space.atoms["ADV"]))
                  for pm in masks for _, bt in rules)
    assert any(s < ceiling for s in scores[:-1]), \
        "the reference chain is monotone now; the reframing needs revisiting"


def test_grow_still_solves_a_trivial_target_both_ways(setup):
    """A smoke test on the builder itself, so a negative result cannot be an
    artefact of a broken harness."""
    space, preds, masks, rules, lp = setup
    truth = X.TRUTHS["copy all"]
    target = space.interpret(truth)
    wrap = lambda t: space.loop(t, lp)
    follow = [(preds[j], masks[j]) for j in range(3)]
    for kw in ({}, {"topn": 5, "follow": follow}):
        found, cost = L.grow(space, rules, preds, masks, target, wrap,
                             rounds=2, beam=1, **kw)
        assert found, f"builder failed on a trivial target with {kw}"
        assert cost > 0
