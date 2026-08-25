"""Bounded writable memory is exact and cheap; the proposed collapse is not.

X59 stopped at a 7 TB scratchpad. The suggested fix -- merge states that agree
on every predicate the vocabulary can currently test -- is unsound, and these
tests pin the one-step counterexample so it cannot be reintroduced as an
optimisation. They also pin the ablation that says the register earns its keep.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x60_registers as G  # noqa: E402


@pytest.fixture(scope="module")
def space():
    alpha = sorted({c for t in G.TAPES for c in t})
    return G.Space(G.TAPES, alpha)


def test_predicate_value_collapse_is_not_a_congruence():
    """Two states no predicate separates, that diverge after one ADV."""
    tape = "zq"
    a = G.St(0, "q")
    b = G.St(0, "w")
    m = ("MATCH", 0)
    assert G.test_pred(m, tape, a) == G.test_pred(m, tape, b), \
        "the states are separable now; the counterexample is void"
    a2, _ = G.run("ADV", tape, a)
    b2, _ = G.run("ADV", tape, b)
    assert G.test_pred(m, tape, a2) != G.test_pred(m, tape, b2), \
        "they no longer diverge; collapsing by predicate value may be sound"


def test_a_register_is_exponentially_cheaper_than_a_tape(space):
    alpha = sorted({c for t in G.TAPES for c in t})
    longest = max(len(t) for t in G.TAPES)
    tape_states = len(alpha) ** longest
    reg_states = len(alpha) + 1
    assert tape_states > 1000 * reg_states


def test_fast_path_equals_interpreter(space):
    for name, truth in G.TASKS.items():
        assert np.array_equal(space.table(truth), space.interpret(truth)), name


def test_the_register_is_load_bearing(space):
    """Without LOAD the memory tasks are not expressible: two tapes identical
    up to the head but differing in the first byte must behave the same for
    any program that cannot remember it."""
    truth = G.TASKS["emit matches"]
    assert G.output(truth, "abab")[0] == "aa"
    assert G.output(truth, "bbab")[0] == "bbb"
    # a memoryless program sees the same suffix from position 1 in both
    assert G.output(truth, "abab")[0] != G.output(truth, "bbab")[0]


def test_a_prologue_shape_is_needed_and_reachable(space):
    """The third shape gap: SEQ(LOAD, LOOP(...)) is not a looped chain."""
    truth = G.TASKS["scan to repeat"]
    target = space.interpret(truth)
    k = max(len(t) for t in G.TAPES) + 2
    body = ("IF", ("MATCH", 0), "NOP", "ADV")
    with_prologue = ("SEQ", ("SEQ", "LOAD", "ADV"), ("LOOP", body))
    assert np.array_equal(space.table(with_prologue), target)
    assert not np.array_equal(space.table(("LOOP", body)), target)
