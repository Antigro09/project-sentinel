"""Generalising the emit test needs three edits, not one.

X56's `capture quoted` kept a 199-node enumeration of observed bytes because
the chain's default POPPED: any byte the enumeration did not name fell through
and closed the string early. Fixing it requires a general test, a different
default, and a new rule all at once -- and these tests pin the fact that no
single move gets there, so a future simplifier cannot quietly go back to
one-at-a-time and appear to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x50_stack as S50  # noqa: E402
import x56_byte_vm as B  # noqa: E402
import x57_repair as P  # noqa: E402


@pytest.fixture(scope="module")
def quoted():
    truth = B.TASKS["capture quoted"]
    markers, _ = B.derive_markers(B.TAPES, truth)
    space = B.Space(B.TAPES, markers, bound=2)
    return space, markers, truth, space.interpret(truth)


def _chain(space, markers, truth):
    """The shape X56 recovers: general-looking, with a popping default."""
    return ([[("BOTH", ("AT", 0, '"'), ("EMPTY",)), B.PUSH_ON],
             [("AT", 0, "$"), "NOP"]],
            ("SEQ", "ADV", "POP"))


def test_a_popping_default_makes_a_general_test_wrong(quoted):
    """The mechanism, as a measurement rather than a story."""
    space, markers, truth, target = quoted
    wrap = lambda t: space.loop(t, 13)
    chain, default = _chain(space, markers, truth)
    general = ("OR", ("TOP", '"'), ("TOP", B.OTHER))
    with_pop = S50._join(chain + [[general, B.EMIT_ON]], default)
    with_adv = S50._join(chain + [[("AT", 0, '"'), ("SEQ", "ADV", "POP")],
                                  [general, B.EMIT_ON]], "ADV")
    assert not np.array_equal(wrap(space.table(with_pop)), target), \
        "the popping default is no longer wrong; X57's premise changed"
    assert np.array_equal(wrap(space.table(with_adv)), target), \
        "the three-edit repair no longer reaches the target"


def test_the_repaired_program_generalises_to_unseen_bytes(quoted):
    """What the whole experiment is for."""
    space, markers, truth, target = quoted
    chain, _ = _chain(space, markers, truth)
    general = ("OR", ("TOP", '"'), ("TOP", B.OTHER))
    fixed = ("LOOP", S50._join(
        chain + [[("AT", 0, '"'), ("SEQ", "ADV", "POP")], [general, B.EMIT_ON]],
        "ADV"))
    for tape in ('"GH"+K', 'w="LMN"', 'a["Q"]b', '<"789">'):
        assert B.output(fixed, tape, markers) == B.output(truth, tape, markers), \
            f"differs on {tape!r}"


def test_a_whole_run_residual_is_not_local(quoted):
    """Why deriving the missing rule from the disagreement fails.

    With the emit test generalised and no pop rule the string never closes, so
    the residual spreads far past the quote positions it is meant to identify.
    """
    space, markers, truth, target = quoted
    wrap = lambda t: space.loop(t, 13)
    chain, default = _chain(space, markers, truth)
    general = ("OR", ("TOP", '"'), ("TOP", B.OTHER))
    broken = S50._join(chain + [[general, B.EMIT_ON]], default)
    bad = P.disagreement(space, wrap(space.table(broken)), target)
    quotes = space.pred(("AT", 0, '"'))
    assert bad.sum() > 3 * quotes.sum(), \
        "the residual is local now; the derived-rule route may work after all"
