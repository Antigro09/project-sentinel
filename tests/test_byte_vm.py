"""The stack quotient must stay exact, or every X56 number is about a
different machine than the verifier runs.

Real text at depth 2 costs 14.5 MB per behaviour with a full stack alphabet
and the frontier holds thousands at once. The quotient collapses every byte
with no TOP test into one symbol, which is lossless only because a program
cannot tell those bytes apart. If a future change adds a TOP test without
adding the byte to the marker set, that stops being true silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x56_byte_vm as B  # noqa: E402


@pytest.fixture(scope="module")
def built():
    out = {}
    for name, truth in B.TASKS.items():
        markers, _ = B.derive_markers(B.TAPES, truth)
        out[name] = (B.Space(B.TAPES, markers, bound=2), markers, truth)
    return out


def test_the_quotient_is_lossless(built):
    """The table and the interpreter must agree under the quotient."""
    for name, (space, markers, truth) in built.items():
        assert np.array_equal(space.table(truth), space.interpret(truth)), name


def test_markers_are_derived_and_are_delimiters(built):
    """Nothing in the source names a quote, bracket or hash as special."""
    expect = {"strip comment": set("#"), "capture quoted": set('"'),
              "capture brackets": set("([]")}
    for name, (_, markers, _) in built.items():
        assert markers == expect[name], f"{name}: derived {markers}"


def test_never_emitted_alone_is_not_enough(built):
    """The boundary test is load-bearing: on real text most bytes are never
    emitted, so without it almost the whole alphabet becomes a marker."""
    truth = B.TASKS["capture brackets"]
    every, emitted = set(), set()
    for tape in B.TAPES:
        res, _ = B.run(truth, tape, B.St(pos=0), set(tape))
        em = set(res.out)
        for p, c in enumerate(tape):
            every.add(c)
            if p in em:
                emitted.add(c)
    never = every - emitted
    markers, _ = B.derive_markers(B.TAPES, truth)
    assert len(never) > 4 * len(markers), \
        f"{len(never)} never-emitted vs {len(markers)} markers"


def test_the_quotient_saves_an_order_of_magnitude(built):
    space, markers, _ = built["capture brackets"]
    alpha = sorted({c for t in B.TAPES for c in t})
    full = len(alpha) + 1
    full_n = sum(len(t) + 1 for t in B.TAPES) * (1 + full + full ** 2)
    assert full_n > 10 * space.n, \
        f"full {full_n} vs quotient {space.n}; the saving vanished"


def test_recovered_bracket_capture_handles_nesting(built):
    """The task itself is real: nesting must actually be exercised."""
    space, markers, truth = built["capture brackets"]
    # Brackets are consumed at every level, so the payload of a nested pair
    # is the letters only -- and 'c' at depth 2 proves the counter is running.
    assert B.output(truth, "a(b[c]d)e", markers)[0] == "bcd"
    assert B.output(truth, "x(y)z", markers)[0] == "y"
    assert B.output(truth, "plain", markers)[0] == ""
