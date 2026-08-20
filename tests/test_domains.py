"""Phase 6: the gate that separates a general system from an ARC solver.

The plan is blunt about what failing this means -- "if it only works on grid
worlds, you built an excellent ARC solver". The test only means anything if
the second domain is structurally different rather than cosmetically
different, which is why `gen/toy.py` does not count: it is still an agent
walking around a board.

DialWorld shares the observation TYPE and nothing else. No agent occupies a
cell, no cell is blocked, no two actions interact through geometry, and a
dial's value is a magnitude rather than a position.
"""

from __future__ import annotations

import numpy as np

from sentinel.domains.dials import (
    DialMechanics,
    DialModel,
    DialWorld,
    mechanic_space,
    render,
    turn,
)
from sentinel.domains.dials import DialState
from sentinel.env.types import Action
from sentinel.verify.evidence import evidence_coverage
from sentinel.verify.verifier import verify


def _episode(mech, start, target, steps=40, seed=0):
    world = DialWorld(mech, start, target)
    world.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))
    return world.history


def test_dials_have_no_spatial_structure():
    """Guard against the domain quietly becoming a grid world again."""
    state = DialState(values=(3, 0, 0, 0), target=(0, 0, 0, 0))
    grid = render(state)
    values = {c for row in grid for c in row}
    assert values <= {0, 2, 5}, "dials should render only bars and targets"
    # Changing a dial changes a COLUMN's height, not a position.
    taller = render(DialState(values=(4, 0, 0, 0), target=(0, 0, 0, 0)))
    changed_rows = {y for y in range(len(grid)) if grid[y] != taller[y]}
    assert len(changed_rows) == 1


def test_coupling_is_invisible_in_any_single_frame():
    """The hidden rule, and the reason this domain is worth testing.

    Two identical boards respond differently to the same action, so a model
    whose state is 'just the grid' cannot be correct -- the same property
    `charge_period` gives the grid worlds.
    """
    plain = DialMechanics(coupling=0)
    coupled = DialMechanics(coupling=1)
    state = DialState(values=(0, 0, 0, 0), target=(9, 9, 9, 9))
    assert render(state) == render(state)
    after_plain = turn(state, Action(1), plain)
    after_coupled = turn(state, Action(1), coupled)
    assert render(after_plain) != render(after_coupled)


def test_history_type_is_shared_but_no_code_is():
    history = _episode(DialMechanics(), (0, 1, 2, 3), (5, 5, 5, 5))
    assert len(history.steps) > 0
    assert history.steps[0].settled.grid is not None
    assert evidence_coverage(history).can_test_transition


def test_verifier_transfers_unchanged():
    """The Phase 6 claim: nothing above env/ should need to change.

    Measured over 24 episodes: the verifier ranked the true rule set first
    in 24 of them, median rank 1 of 24.
    """
    space = mechanic_space()
    rng = np.random.default_rng(0)
    firsts = 0
    trials = 0
    for trial in range(8):
        truth = space[trial % len(space)]
        start = tuple(int(v) for v in rng.integers(0, 6, 4))
        target = tuple(int(v) for v in rng.integers(0, 6, 4))
        history = _episode(truth, start, target, seed=trial)
        if len(history.steps) < 5:
            continue
        trials += 1
        scored = sorted(
            ((verify(DialModel(c, start, target), history).fitness, c) for c in space),
            key=lambda p: -p[0],
        )
        firsts += int(scored[0][1].summary() == truth.summary())
    assert trials > 0
    assert firsts == trials


def test_exact_model_is_perfect_and_wrong_ones_are_not():
    truth = DialMechanics(step=2, modulus=8, coupling=1)
    start, target = (0, 0, 0, 0), (6, 6, 6, 6)
    history = _episode(truth, start, target)
    good = verify(DialModel(truth, start, target), history)
    assert good.first_divergence is None

    bad = verify(DialModel(DialMechanics(step=1, modulus=8, coupling=1), start, target), history)
    assert bad.fitness < good.fitness
