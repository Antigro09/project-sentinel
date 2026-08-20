"""Gates for experiment design and for scoring labels only where they mean
something.

Two findings drive this file, and both are about evidence rather than
learning:

- `ordered_targets` is under-determined. Random play distinguishes it in 6%
  of worlds against 100% for the hidden counter, so a core sitting near its
  prior is responding correctly to evidence that does not determine the
  answer.
- Aiming an experiment at it requires knowing the movement rule FIRST.
  Collection happens only at the final cell of a move, so a greedy prober
  overshoots: landings fall 2.81 -> 1.19 -> 0.43 as step_distance goes
  1 -> 3.
"""

from __future__ import annotations

import numpy as np

from sentinel.core.data import exploration_history, probing_history
from sentinel.core.encoding import CROP, HEADS, MAX_TRANSITIONS, MechanicLabels, defined_mask
from sentinel.core.train import load_core, save_core
from sentinel.core.model import CoreConfig, TinyRecursiveCore
from sentinel.explore import staged_exploration
from sentinel.gen.generator import generate
from sentinel.gen.grid import TARGET
from sentinel.gen.spec import Mechanics

import mlx.core as mx


def _worlds(n, start=7000, wide=True):
    out, seed = [], start
    space = None
    if wide:
        from sentinel.gen.generator import mechanic_space
        space = mechanic_space(wide=True)
    while len(out) < n and seed < start + 60 * n:
        mech = space[(seed * 7919) % len(space)] if space else None
        w = generate(seed=seed, mechanics=mech)
        if w is not None:
            out.append(w)
        seed += 1
    return out


def test_defined_mask_marks_unanswerable_labels():
    """A label with no observable consequence must not be scored."""
    names = [n for n, _ in HEADS]
    no_counter = MechanicLabels.from_mechanics(
        Mechanics(charge_period=None, has_switches=False)
    ).as_array()[None]
    assert not defined_mask(no_counter, names.index("wait_advances_charge"))[0]
    assert not defined_mask(no_counter, names.index("gates_start_open"))[0]

    full = MechanicLabels.from_mechanics(
        Mechanics(charge_period=3, has_switches=True)
    ).as_array()[None]
    assert defined_mask(full, names.index("wait_advances_charge"))[0]
    assert defined_mask(full, names.index("gates_start_open"))[0]


def test_unconditional_labels_are_always_defined():
    labels = MechanicLabels.from_mechanics(Mechanics()).as_array()[None]
    names = [n for n, _ in HEADS]
    for name in ("step_distance", "charge_period", "edge_mode", "hazards", "switches"):
        assert defined_mask(labels, names.index(name)).all()


def test_core_weights_roundtrip(tmp_path):
    """Training costs ~20 minutes; reloading must give the same model."""
    cfg = CoreConfig(cycles=2)
    core = TinyRecursiveCore(cfg)
    # Full-length sequences: positional encodings are fixed at
    # MAX_TRANSITIONS, so a shorter batch cannot broadcast.
    grids = mx.zeros((2, MAX_TRANSITIONS, CROP, CROP, 3), dtype=mx.int32)
    actions = mx.zeros((2, MAX_TRANSITIONS), dtype=mx.int32)
    before = [np.array(h) for h in core(grids, actions)]

    path = save_core(core, tmp_path / "core.safetensors")
    again = load_core(path, cfg)
    after = [np.array(h) for h in again(grids, actions)]

    for a, b in zip(before, after):
        assert np.allclose(a, b, atol=1e-5)


def test_explorers_produce_usable_episodes():
    for spec in _worlds(3):
        for make in (exploration_history, probing_history):
            history = make(spec, 0, 40)
            assert len(history.steps) > 0
            assert all(s.action.action_id in (1, 2, 3, 4, 5) for s in history.steps)


def test_staged_exploration_lands_on_targets_more_than_greedy():
    """The point of staging: aim the experiment with the inferred rule.

    Compared on worlds where step_distance > 1, which is exactly where a
    greedy prober sails past its goal.
    """
    specs = [s for s in _worlds(6) if s.mechanics.step_distance > 1][:4]
    if not specs:
        return
    staged_total = greedy_total = 0
    for spec in specs:
        result = staged_exploration(spec, seed=0, free_steps=25, aimed_steps=35)
        staged_total += result.landings

        size = spec.field_size
        history = probing_history(spec, 0, 60)
        previous = history.initial
        for step in history.steps:
            for y in range(size):
                for x in range(size):
                    if previous.grid[y][x] == TARGET and step.settled.grid[y][x] == 4:
                        greedy_total += 1
            previous = step.settled
    assert staged_total >= greedy_total


def test_staged_exploration_reports_whether_it_could_aim():
    """A run that never managed to plan is a finding, not a silent zero."""
    for spec in _worlds(2):
        result = staged_exploration(spec, seed=0, free_steps=20, aimed_steps=20)
        assert isinstance(result.planned, bool)
        assert result.landings >= 0
        assert result.believed is not None
