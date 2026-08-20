"""Gates for test-time adaptation.

The load-bearing test here is `test_true_mechanics_never_beaten`. Three
separate bugs let a WRONG rule set outscore the true one, and each was
invisible until measured: multi-level episodes scored against a one-level
hypothesis, level crossings that `level_index` cannot see, and target order
being treated as observed when it is part of the hypothesis.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.adapt.hypothesis import (
    classes_from_mechanics,
    mechanics_from_classes,
    scorable_segment,
    score_hypothesis,
)
from sentinel.adapt.search import ALL_HYPOTHESES, exhaustive_search
from sentinel.core.agent import read_layout
from sentinel.env.boundary import continuous_runs, is_boundary
from sentinel.env.types import Action
from sentinel.gen.generator import generate
from sentinel.gen.grid import GridWorld
from sentinel.gen.spec import Mechanics


def _explore(spec, steps=60, seed=0):
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))
    return world.history


def _worlds(n=6):
    out = []
    seed = 1000
    while len(out) < n and seed < 1000 + 40 * n:
        world = generate(seed=seed)
        if world is not None:
            out.append(world)
        seed += 1
    return out


def test_class_roundtrip():
    for classes in ALL_HYPOTHESES[:24]:
        assert classes_from_mechanics(mechanics_from_classes(classes)) == classes


def test_hypothesis_space_is_complete():
    from sentinel.adapt.hypothesis import NCLASS

    expected = 1
    for n in NCLASS:
        expected *= n
    assert len(ALL_HYPOTHESES) == expected
    assert len(set(ALL_HYPOTHESES)) == len(ALL_HYPOTHESES)


def test_segments_contain_no_boundary():
    """A scorable segment must be replayable without the board rebuilding."""
    for spec in _worlds():
        history = _explore(spec)
        segment = scorable_segment(history)
        previous = segment.initial
        for step in segment.steps:
            assert not is_boundary(previous, step.settled)
            previous = step.settled


def test_runs_cover_every_step():
    """Splitting an episode must not silently lose steps."""
    for spec in _worlds():
        history = _explore(spec)
        runs = continuous_runs(history)
        covered = sum(len(steps) for _, steps in runs)
        boundaries = sum(
            1
            for i, step in enumerate(history.steps)
            if is_boundary(history.observation_before(i), step.settled)
        )
        assert covered + boundaries == len(history.steps)


def test_true_mechanics_never_beaten():
    """No wrong rule set may explain the evidence better than the truth.

    Sampled rather than exhaustive: the space holds 5,760 rule sets and one
    verifier replay costs ~17.5ms, so checking all of them against every
    world would take the suite past ten minutes. The sample is seeded, so a
    failure is reproducible.
    """
    rng = np.random.default_rng(0)
    for spec in _worlds(4):
        history = _explore(spec)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)
        truth = classes_from_mechanics(spec.mechanics)
        true_fit = score_hypothesis(truth, history, observed, spec.field_size).fitness
        picks = rng.choice(len(ALL_HYPOTHESES), size=60, replace=False)
        for index in picks:
            classes = ALL_HYPOTHESES[int(index)]
            if classes == truth:
                continue
            other = score_hypothesis(classes, history, observed, spec.field_size).fitness
            assert other <= true_fit + 1e-9, (
                f"{spec.world_id}: {mechanics_from_classes(classes).summary()} "
                f"scored {other:.3f} > truth {true_fit:.3f}"
            )


def test_search_finds_identifiable_rules():
    """Search must recover the rules the evidence actually determines."""
    recovered = 0
    total = 0
    for spec in _worlds(3):
        history = _explore(spec)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)
        found = exhaustive_search(history, observed, spec.field_size)
        truth = classes_from_mechanics(spec.mechanics)
        total += 1
        # step_distance and charge_period are the rules random exploration
        # always exercises; ordered_targets is famously not.
        recovered += int(found.best.classes[:2] == truth[:2])
    assert total > 0
    assert recovered / total >= 0.6


def test_search_stops_early_on_exact_explanation():
    spec = _worlds(1)[0]
    history = _explore(spec)
    segment = scorable_segment(history)
    if len(segment.steps) < 5:
        pytest.skip("episode too short to score")
    observed = read_layout(segment.initial.grid, spec.field_size)
    found = exhaustive_search(history, observed, spec.field_size)
    if found.best.fitness >= 1.0:
        assert not found.exhausted
        assert found.replays <= len(ALL_HYPOTHESES)
