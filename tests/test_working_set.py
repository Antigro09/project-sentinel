"""Measuring the index along one axis reports an artefact, not a plateau.

X61's first two attempts both concluded that nested brackets need only a
bounded working set. Both were wrong: sweeping prefix length with suffixes
fixed caps the index at what a short suffix can distinguish, and sweeping
suffixes with prefixes fixed caps it at what a short prefix can reach. Only
the diagonal has no ceiling built into it. These tests pin the difference,
because the wrong measurement looks tidier than the right one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x61_working_set as K  # noqa: E402


def test_a_fixed_suffix_horizon_hides_unbounded_memory():
    """The artefact that produced two wrong readings."""
    f = K.TASKS["capture brackets (nested)"]
    capped = [K.nerode_index(f, n, K.ALPHA, suffix_len=3) for n in (3, 4, 5)]
    assert capped[-1] == capped[-2], "the cap no longer bites; re-read X61"
    diag = [K.nerode_index(f, n, K.ALPHA, suffix_len=n) for n in (3, 4, 5)]
    assert diag[-1] > diag[-2], "nested brackets look bounded on the diagonal"


def test_genuinely_bounded_tasks_stay_bounded_on_the_diagonal():
    for name in ("strip comment", "capture quoted", "dedupe adjacent",
                 "emit matching first"):
        f = K.TASKS[name]
        diag = [K.nerode_index(f, n, K.ALPHA, suffix_len=n) for n in (3, 4, 5)]
        assert diag[-1] == diag[-2] == diag[-3], f"{name}: {diag}"


def test_the_split_is_even_not_lopsided():
    """The premise under test was 'almost all'. It is half."""
    bounded = 0
    for name, f in K.TASKS.items():
        diag = [K.nerode_index(f, n, K.ALPHA, suffix_len=n) for n in (4, 5)]
        bounded += int(diag[-1] == diag[-2])
    assert bounded == 4, f"{bounded}/8 bounded; the X61 conclusion changed"


def test_a_register_sized_index_is_not_a_claim_about_difficulty():
    """Index is a memory requirement, not a search cost -- X51 spent three
    experiments on a task whose index is trivial."""
    f = K.TASKS["dedupe adjacent"]
    idx = K.nerode_index(f, 5, K.ALPHA, suffix_len=5)
    assert idx == len(K.ALPHA) + 1, idx
