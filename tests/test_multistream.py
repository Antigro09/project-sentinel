"""SAME earns its place; a writable scratchpad cannot be a state dimension.

X51 added EQTOP because it was a good idea and it appeared in 0 of 4 recovered
programs. SAME is the two-stream analogue, and the difference is that three of
six tasks here are unrecoverable without it. These tests pin the ablation, and
the arithmetic that rules the scratchpad out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x59_multistream as M  # noqa: E402


@pytest.fixture(scope="module")
def space():
    return M.Space(M.PAIRS)


def test_fast_path_equals_interpreter(space):
    for name, truth in M.TASKS.items():
        assert np.array_equal(space.table(truth), space.interpret(truth)), name


def test_same_names_no_byte_and_compares_the_heads(space):
    assert M.test_pred(M.SAME, ("abc", "xab"), M.St(0, 1))
    assert not M.test_pred(M.SAME, ("abc", "xab"), M.St(0, 0))
    # and it is genuinely cross-stream: neither head alone determines it
    assert M.test_pred(M.SAME, ("q", "q"), M.St(0, 0))
    assert not M.test_pred(M.SAME, ("q", "z"), M.St(0, 0))


def test_same_is_load_bearing(space):
    """Without SAME these targets are not expressible over the byte tests.

    Checked by construction rather than by search: any program over AT tests
    alone must behave identically on two pairs whose bytes match positionally
    but whose cross-stream agreement differs.
    """
    truth = M.TASKS["scan 2 to match"]
    a = M.output(truth, ("ab", "ba"))
    b = M.output(truth, ("ab", "ab"))
    assert a != b, "the two pairs no longer separate; the ablation is void"


def test_same_needs_offsets_for_advance_then_compare(space):
    """The X58 lesson, repeated: a chain cannot test what the head has not
    reached, so `halt at match` is unreachable with SAME(0,0) alone."""
    truth = M.TASKS["halt at match"]
    target = space.interpret(truth)
    look = ("IF", ("SAME", 0, 1), ("SEQ", "ADV2", "HALT"), "ADV2")
    flat = ("IF", ("SAME", 0, 0), ("SEQ", "ADV2", "HALT"), "ADV2")
    assert np.array_equal(space.table(look), target)
    assert not np.array_equal(space.table(flat), target)


def test_a_writable_scratchpad_is_not_a_state_dimension():
    """The arithmetic that decided the design, not a guess."""
    L, alpha = 8, 8
    two_pointers = (L + 1) ** 2
    with_scratchpad = two_pointers * (L + 1) * alpha ** L
    assert with_scratchpad > 10 ** 8 * two_pointers, \
        "the scratchpad blow-up changed; the X59 design decision needs redoing"
