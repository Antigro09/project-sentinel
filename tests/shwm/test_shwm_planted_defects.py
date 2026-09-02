"""The planted-defect matrix, as a regression test.

Gate T2 is not a number in a report -- it is the property that every planted
sequence defect is caught by its intended guard, that no guard passes everything,
and that no guard fires on the honest pipeline. All three are asserted here so a
later edit cannot quietly reintroduce a vacuous guard.

Building this suite immediately caught two broken guards of my own: `Batch.copy`
dropped the planted flags, so a behavioural guard compared the defective code path
against the honest one and always saw a difference; and the blocked-action guard
used the wrong column offsets and fired on honest data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/shwm"))


@pytest.fixture(scope="module")
def matrix():
    from planted_defects import DEFECTS, GUARDS, INTENDED, ProbeModel, Batch
    import numpy as np

    # A synthetic batch with the same column layout, so the test needs no environment
    # rollout and stays fast. Twenty-six features: the encoder's twenty-four plus the
    # two reset columns.
    rng = np.random.default_rng(0)
    episodes, steps, width = 6, 5, 26
    x = rng.normal(size=(episodes, steps, width)).astype(np.float32)
    y = rng.integers(0, 5, size=(episodes, steps)).astype(np.int64)
    lengths = np.array([5, 4, 5, 3, 5, 4])
    mask = np.zeros((episodes, steps), dtype=np.float32)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
        x[i, n:] = 0.0
        x[i, :, -1] = 0.0
        x[i, 0, -1] = 1.0                      # the reset flag, once, at step 0
        for t in range(n):                     # keep the blocked/query guard consistent
            action = int(np.argmax(x[i, t, 17:21]))
            opposite = (action + 2) % 4
            x[i, t, 0:4] = 0.0
            if y[i, t] == 4:
                x[i, t, action] = 1.0
                x[i, t, opposite] = 1.0
    honest = Batch(x, y, mask, lengths, np.arange(episodes))
    model = ProbeModel(width)
    arms = {"0_honest": honest}
    for name, injector in DEFECTS.items():
        arms[name] = injector(honest)
    out = {}
    for arm_name, arm in arms.items():
        out[arm_name] = {}
        for guard_name, guard in GUARDS.items():
            try:
                out[arm_name][guard_name] = bool(guard(arm, model))
            except Exception:                   # noqa: BLE001
                out[arm_name][guard_name] = False
    return out, INTENDED, list(GUARDS), list(arms)


def test_every_guard_passes_the_honest_pipeline(matrix) -> None:
    out, _, guards, _ = matrix
    failing = [g for g in guards if not out["0_honest"][g]]
    assert not failing, f"guards firing on honest data: {failing}"


def test_no_guard_is_vacuous(matrix) -> None:
    """A guard that passes every arm has no detection power."""
    out, _, guards, arms = matrix
    vacuous = [g for g in guards if all(out[a][g] for a in arms)]
    assert not vacuous, f"vacuous guards: {vacuous}"


def test_every_planted_defect_is_caught_by_its_guard(matrix) -> None:
    out, intended, _, _ = matrix
    missed = [d for d, g in intended.items() if out[d][g]]
    assert not missed, f"defects that survived their guard: {missed}"


def test_batch_copy_carries_planted_flags() -> None:
    """The specific bug that made a behavioural guard vacuous."""
    from planted_defects import Batch, defect_reset_state_every_step
    import numpy as np

    batch = Batch(np.zeros((2, 3, 26), dtype=np.float32), np.zeros((2, 3), dtype=np.int64),
                  np.ones((2, 3), dtype=np.float32), np.array([3, 3]), np.arange(2))
    flagged = defect_reset_state_every_step(batch)
    assert getattr(flagged.copy(), "_flag_reset_every_step", False) is True
