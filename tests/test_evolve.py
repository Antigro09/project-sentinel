"""Gates for scaffold self-modification.

Every test here is about the overseer rather than the search. The search is
ordinary; what has to hold is that it cannot quietly buy score with
resources, cannot promote on the set it tuned against, and cannot lose the
record of what it tried.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.evolve.archive import Archive, Version
from sentinel.evolve.genome import ScaffoldGenome
from sentinel.evolve.search import ACTION_PRICE, Evaluation, evolve
from sentinel.gen.generator import generate


def _worlds(n, start):
    out, seed = [], start
    while len(out) < n and seed < start + 40 * n:
        w = generate(seed=seed)
        if w is not None:
            out.append(w)
        seed += 1
    return out


def test_mutation_stays_inside_bounds():
    rng = np.random.default_rng(0)
    genome = ScaffoldGenome()
    for _ in range(200):
        genome = genome.mutate(rng, rate=1.0)
        assert 8 <= genome.explore_steps <= 120
        assert 0 <= genome.simplicity_weight <= 4
        assert 2 <= genome.order_search_cap <= 5
        assert 1 <= genome.library_k <= 32
        assert 0.0 <= genome.library_strength <= 4.0
        assert 1 <= genome.stall_limit <= 8


def test_actions_are_priced_into_score():
    """Solving the same worlds with more actions must score worse."""
    cheap = Evaluation(solve_rate=0.5, mean_actions=40, mechanics_exact=0.5)
    dear = Evaluation(solve_rate=0.5, mean_actions=140, mechanics_exact=0.5)
    assert cheap.score > dear.score
    assert abs((cheap.score - dear.score) - ACTION_PRICE * 100) < 1e-9


def test_archive_ranks_on_guard_never_on_tuning():
    archive = Archive()
    archive.record(Version(0, ScaffoldGenome(), 0.50, 0.50, 40, True, "baseline"))
    # Higher tuning score, worse guard score: must not win.
    archive.record(
        Version(1, ScaffoldGenome(explore_steps=110), 0.95, 0.30, 110, True, "overfit")
    )
    best = archive.best()
    assert best is not None and best.generation == 0
    assert archive.rollback().explore_steps == 60


def test_rejected_versions_are_still_archived():
    archive = Archive()
    archive.record(Version(1, ScaffoldGenome(), 0.9, 0.2, 100, False, "guard rejected"))
    assert len(archive) == 1
    assert archive.promoted == []
    assert archive.rollback() == ScaffoldGenome()


def test_rollback_defaults_to_baseline_when_nothing_survives():
    assert Archive().rollback() == ScaffoldGenome()


def test_guard_overlap_is_refused():
    worlds = _worlds(3, 3000)
    with pytest.raises(ValueError, match="overlap"):
        evolve(train=worlds, guard=worlds, generations=1, population=1)


def test_archive_roundtrips(tmp_path):
    archive = Archive()
    archive.record(Version(0, ScaffoldGenome(), 0.5, 0.5, 40, True, "baseline"))
    path = archive.save(tmp_path / "archive.json")
    again = Archive.load(path)
    assert len(again) == 1
    assert again.versions[0].genome == ScaffoldGenome()
    assert again.versions[0].note == "baseline"
