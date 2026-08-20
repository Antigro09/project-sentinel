"""The encoder must never present a board rebuild as the effect of an action.

This is the bug that hid `ordered_targets` for weeks. A level change
reinstates every target, and the only visible consequence of ordered
objectives is a target REAPPEARING after the agent steps off one it could
not collect. Feeding level changes to the network as ordinary transitions
therefore buries the one signal that distinguishes the label -- and buries
it under a louder pattern that is anti-correlated with it.

Measured over 240 held-out episodes: within-level target reappearance
occurs in 1.7% of unordered episodes and 60.8% of ordered ones, while
across-level reappearance occurs in 71.7% of unordered and 53.3% of
ordered.
"""

from __future__ import annotations

import numpy as np

from sentinel.bootstrap.teacher import make_training_history
from sentinel.core.encoding import BOUNDARY_ACTION, MAX_TRANSITIONS, encode_history
from sentinel.env.boundary import is_boundary
from sentinel.env.types import Action
from sentinel.gen.generator import generate
from sentinel.gen.grid import TARGET, GridWorld


def _worlds(n=8, start=5000):
    out, seed = [], start
    while len(out) < n and seed < start + 40 * n:
        w = generate(seed=seed)
        if w is not None:
            out.append(w)
        seed += 1
    return out


def _play(spec, steps=90, seed=0):
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))
    return world.history


def test_boundaries_are_marked_not_dropped():
    """The time axis must survive, and discontinuities must be flagged.

    Dropping boundary transitions was measured at charge_period 0.088 --
    below the 0.33 chance line -- because the hidden counter keeps ticking
    across a level change and a removed transition leaves an invisible gap.
    """
    saw_a_boundary = False
    for spec in _worlds():
        # Solution trajectories, not random play: random play rarely
        # finishes a level, so a boundary never occurs and the test would
        # pass while proving nothing.
        history = make_training_history(spec, seed=0)
        grids, actions = encode_history(history)

        previous = history.initial
        expected = []
        for step in history.steps:
            if len(expected) >= MAX_TRANSITIONS:
                break
            expected.append(is_boundary(previous, step.settled))
            previous = step.settled

        for i, want_boundary in enumerate(expected):
            marked = int(actions[i]) == BOUNDARY_ACTION
            assert marked == want_boundary
            saw_a_boundary |= marked
    assert saw_a_boundary, "no boundary in any episode; the test proves nothing"


def test_boundary_marker_is_distinct_from_padding():
    """A rebuild and an empty slot must not look alike to the network."""
    assert BOUNDARY_ACTION != -1
    for spec in _worlds(4):
        grids, actions = encode_history(_play(spec))
        for value in actions.tolist():
            assert value == -1 or value == BOUNDARY_ACTION or 0 <= value <= 5


def test_no_encoded_transition_rebuilds_the_board():
    """A board rebuild shows up as many cells changing at once.

    A single action moves an agent and at most clears a target. If an
    encoded transition changes a large fraction of the active area, a
    discontinuity got through.
    """
    for spec in _worlds():
        history = _play(spec)
        grids, actions = encode_history(history)
        for i in range(MAX_TRANSITIONS):
            if int(actions[i]) < 0:
                continue
            changed = int(grids[i, :, :, 2].sum())
            assert changed <= 12, f"{spec.world_id} transition {i}: {changed} cells changed"


def test_target_mass_never_rises_in_unordered_worlds():
    """The signal `ordered_targets` actually rests on.

    Under unordered rules a collected target is gone for good, so within a
    level the count of target cells can only fall. If it rises in an encoded
    transition, a level boundary was encoded and the label is being taught
    backwards.
    """
    for spec in _worlds(12):
        if spec.mechanics.ordered_targets:
            continue
        history = _play(spec)
        grids, actions = encode_history(history)
        for i in range(MAX_TRANSITIONS):
            if int(actions[i]) < 0:
                continue
            before = int((grids[i, :, :, 0] == TARGET).sum())
            after = int((grids[i, :, :, 1] == TARGET).sum())
            assert after <= before, (
                f"{spec.world_id} transition {i}: targets {before} -> {after} "
                "in an unordered world"
            )
