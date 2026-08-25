"""Two gaps that were invisible on three tasks and obvious across fourteen.

A target that never emits leaves no emission boundary, so marker derivation
came back empty and the task could not even be set up. And a decision list
cannot test a byte the head has not reached, so `halt at m` is not in the
language at all under offset-0 tests. Both read as search failures and were
neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x56_byte_vm as B  # noqa: E402
import x58_sweep as W  # noqa: E402


def test_markers_derive_for_a_target_that_never_emits():
    """The emission-boundary rule alone returns nothing here."""
    truth = W.TASKS["halt at '#'"]
    for tape in W.TAPES:
        res, _ = B.run(truth, tape, B.St(pos=0), set(tape))
        assert not res.out, "this target emits; the test premise is gone"
    markers, _ = B.derive_markers(W.TAPES, truth)
    assert markers == {"#"}, f"derived {markers}"


def test_emission_markers_are_unchanged_by_the_halt_fallback():
    """The fallback must only fire when there is nothing else to read."""
    for name, expect in (("strip after '#'", {"#"}),
                         ("capture nested ()", {"(", ")"})):
        markers, _ = B.derive_markers(W.TAPES, W.TASKS[name])
        assert expect <= markers, f"{name}: derived {markers}"


def test_halt_at_needs_a_lookahead_test(setup=None):
    """With offset-0 tests only, the task is not expressible -- so a search
    failure there is a vocabulary gap, not a search gap."""
    truth = W.TASKS["halt at '#'"]
    markers, _ = B.derive_markers(W.TAPES, truth)
    space = B.Space(W.TAPES, markers, bound=2)
    target = space.interpret(truth)
    lookahead = ("IF", ("AT", 1, "#"), ("SEQ", "ADV", "HALT"), "ADV")
    assert np.array_equal(space.table(lookahead), target)
    same_at_zero = ("IF", ("AT", 0, "#"), ("SEQ", "ADV", "HALT"), "ADV")
    assert not np.array_equal(space.table(same_at_zero), target)


def test_the_task_family_is_not_degenerate():
    """X55's lesson: a family of targets that collapse onto each other
    measures nothing. These must be behaviourally distinct."""
    seen = {}
    for name, truth in W.TASKS.items():
        markers, _ = B.derive_markers(W.TAPES, truth)
        key = tuple(B.output(truth, tp, markers) for tp in W.TAPES)
        assert key not in seen, f"{name} is indistinguishable from {seen[key]}"
        seen[key] = name
    assert len(seen) == len(W.TASKS)
