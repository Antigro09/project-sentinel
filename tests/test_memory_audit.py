"""Expressible and findable are different, and so are bounded and constant.

X62's two near-misses are both easy to reintroduce. Reading the index from
its last two values calls a plateau-in-progress "linear" -- the set tasks run
6, 16, 26, 31, 32 and stop at exactly 2^|alphabet|. And bounding PUSH in the
interpreter as well as in the search abstraction judges a correct program
wrong, which is the sixth time a primitive has been quietly crippled here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x62_memory_audit as A  # noqa: E402


def test_the_classifier_reads_the_trend_not_the_last_pair():
    """Set tasks plateau at 2^|alphabet|; the final +1 is arrival, not growth."""
    idx = [A.nerode_index(A.first_occurrence_only, n, A.ALPHA) for n in (1, 2, 3, 4, 5)]
    assert idx[-1] == 2 ** len(A.ALPHA), idx
    assert idx[-1] > idx[-2], "no longer a near-miss; the test is moot"
    growth, mem = A.classify(idx, A.ALPHA)
    assert growth == "converging" and "set-like" in mem, (growth, mem)


def test_a_table_task_is_not_labelled_set_shaped():
    idx = [A.nerode_index(A.substitute, n, A.ALPHA) for n in (1, 2, 3, 4, 5)]
    _, mem = A.classify(idx, A.ALPHA)
    assert "table-like" in mem, mem


def test_only_reverse_grows_with_input_length():
    """The headline refinement over X61: set and table memory are bounded for
    a fixed alphabet, so the gap there is shape rather than capacity."""
    growing = []
    for _fam, tasks in A.FAMILIES.items():
        for name, f in tasks:
            idx = [A.nerode_index(f, n, A.ALPHA) for n in (1, 2, 3, 4, 5)]
            if A.classify(idx, A.ALPHA)[0] == "exponential":
                growing.append(name)
    assert growing == ["reverse"], growing


def test_the_interpreter_stack_is_unbounded(setup=None):
    """The bound belongs to the search abstraction only."""
    deep = "(" * 6 + "z" + ")" * 6
    st, _ = A.run(("LOOP", ("SEQ", "PUSH", "ADV")), deep, A.St(0))
    assert len(st.stack) > A.DEPTH, len(st.stack)


def test_every_witness_reproduces_its_task():
    for _fam, tasks in A.FAMILIES.items():
        for name, f in tasks:
            wit = A.WITNESS[name]
            if wit is None:
                continue
            for tape in A.TAPES + A.HELD_OUT:
                assert A.emit(wit, tape) == f(tape), f"{name} on {tape!r}"


def test_the_inexpressible_claims_have_a_counting_argument():
    """A set over the alphabet needs more states than one register holds."""
    assert 2 ** len(A.ALPHA) > len(A.ALPHA) + 1
    for name in ("first occurrence only", "emit if seen before", "reverse",
                 "substitute"):
        assert A.WITNESS[name] is None, f"{name} now has a witness"
