"""A counter cannot remember which bracket it is inside.

X49 broke the window wall with a counter and called unbounded nesting the
next one. Removing the clamp is not that wall -- a counter with no bound is
still a counter. The wall one place further out is knowing WHICH bracket is
open, and it has a certificate exactly like X49's, one term stronger: two
positions with the same window AND the same depth where the truth disagrees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x50_stack as X  # noqa: E402


@pytest.fixture(scope="module")
def space():
    return X.Space(X.TAPES, sorted({c for t in X.TAPES for c in t}))


def test_certificate_rules_out_window_and_counter():
    key, byd = X.certificate(X.TAPES, X.TRUTHS[X.TYPED], use_depth=True)
    assert key is not None, "no window+depth collision: the claim is unproven"
    assert set(byd) == {True, False}


def test_certificate_discriminates():
    """It must NOT fire on a task a counter can actually do."""
    key, _ = X.certificate(X.TAPES, X.TRUTHS["copy inside any"], use_depth=True)
    assert key is None, "the counter certificate fires on a counter-solvable task"


def test_fast_path_equals_interpreter(space):
    for name, truth in X.TRUTHS.items():
        assert np.array_equal(space.table(truth), space.interpret(truth)), name


def test_search_tapes_never_reach_the_abstraction_bound(space):
    """If they did, the table would describe a machine the verifier doesn't run."""
    for t in X.TAPES:
        deepest = max(X.scan_depth(t, i) for i in range(len(t) + 1))
        assert deepest <= space.bound


def test_push_is_domain_agnostic():
    """PUSH pushes whatever is under the head -- it knows nothing of brackets."""
    for ch in "a[(x":
        st, _ = X.run("PUSH", ch + "zz", X.St(pos=0))
        assert st.stack == (ch,)


def test_the_typed_rule_needs_the_top_of_stack(space):
    """Swapping TOP for EMPTY -- all a counter can ask -- changes behaviour."""
    counter_only = ("LOOP", ("IF", X.IS_OPEN, X.PUSH_ON,
                             ("IF", X.IS_CLOSE, X.POP_ON,
                              ("IF", ("EMPTY",), "ADV", X.EMIT_ON))))
    assert any(X.output(counter_only, t) != X.output(X.TRUTHS[X.TYPED], t)
               for t in X.EVAL)


def test_truth_is_depth_agnostic_on_an_unbounded_stack():
    """The rule carries no depth constant, so nesting far past the search
    abstraction must still work under the unbounded interpreter."""
    deep = "[" * 12 + "abc" + "]" * 12
    out, _, _ = X.output(X.TRUTHS[X.TYPED], deep, bound=None)
    assert out == "abc"
    mixed = "[" * 6 + "(" + "abc" + ")" + "]" * 6
    assert X.output(X.TRUTHS[X.TYPED], mixed, bound=None)[0] == ""


def test_polish_deletes_a_dead_rule(space):
    """A rule shadowed by an identical earlier one must not survive."""
    truth = X.TRUTHS["strip brackets"]
    target = space.interpret(truth)
    wrap = lambda t: space.loop(t, max(len(t2) for t2 in X.TAPES) + 2)
    chain, default = X._split(truth[1])
    bloated = X._join([chain[0]] + chain, default)
    assert X.size_of(bloated) > X.size_of(truth[1])
    assert np.array_equal(wrap(space.table(bloated)), target), "premise broken"
    lean = X.polish(space, bloated, [p for p, _ in chain], target, wrap)
    assert X.size_of(lean) < X.size_of(bloated)
    assert np.array_equal(wrap(space.table(lean)), target)
