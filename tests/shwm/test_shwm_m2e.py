"""Regression pins for the M2E closure.

Several of these pin corrections rather than capabilities. `test_detector_nll_is_finite`
exists because `np.clip(x_float32, 1e-9, 1 - 1e-9)` does not clip: 1 - 1e-9 rounds to
exactly 1.0 in float32, so a confident correct prediction produced log(0) and the
development NLL was reported as nan. `test_route_length_is_the_right_parity_exponent`
pins the fact that the specification's event-count form of the independence diagnostic
over-predicts measured parity, because errors happen on steps, not only on events.
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/shwm"))

import m2e_core as core  # noqa: E402
import m2e_event_target as target  # noqa: E402


def test_generic_initialisation_is_orientation_free():
    diagonal = np.array([core.distances(core.generic_antisymmetric(s, 2))
                         ["stay_minus_flip_diagonal"] for s in range(20_000, 20_400)])
    assert abs(diagonal.mean()) < 0.1, "the family must not lean toward stay/flip"
    assert 0.4 < (diagonal > 0).mean() < 0.6
    answer = core.distances(core.ANSWER)["stay_minus_flip_diagonal"]
    assert answer > 0.9, "the M2D perturbation IS oriented; that is the M2D finding"


def test_event_axis_reversal_maps_the_generic_family_onto_itself():
    """If reversing which event is called 0 changed the family, the initialisation
    would encode an event name."""
    forward, backward = [], []
    for s in range(20_000, 20_200):
        p = core.generic_antisymmetric(s, 2)
        forward.append(core.distances(p)["stay_minus_flip_diagonal"])
        backward.append(core.distances(
            np.ascontiguousarray(p[::-1]))["stay_minus_flip_diagonal"])
    assert np.allclose(np.mean(forward), -np.mean(backward))


def test_no_generic_draw_serialises_the_answer():
    answer = core.initialisation_digest(core.ANSWER)
    for s in range(20_000, 20_100):
        assert core.initialisation_digest(core.generic_antisymmetric(s, 2)) != answer


def test_answer_oriented_arm_is_marked_ineligible_in_code():
    arms = core.build_arms(4)
    assert arms["B_answer_oriented_init"].eligible is False
    assert arms["A_exact_xor_accumulator"].eligible is False
    assert arms["E_generic_restarts"].eligible is True


def test_restart_selection_sees_only_training_inputs():
    parameters = set(inspect.signature(core.train_arm).parameters)
    assert parameters <= {"arm", "train", "seed", "updates", "event_transform"}
    # Names from the syntax tree, not a substring scan of the source: the docstring
    # contains the word "validation", and a check that a docstring can fail is not a
    # check of the code.
    tree = ast.parse(inspect.getsource(core.training_log_likelihood))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not any(("valid" in n) or ("test" in n) for n in names), sorted(names)


def test_compute_ledger_charges_every_restart():
    ledger = core.ComputeLedger()
    for _ in range(5):
        one = core.ComputeLedger(training_runs=1, optimizer_updates=1024,
                                 likelihood_evaluations=1)
        ledger.merge(one)
    assert ledger.training_runs == 5
    assert ledger.optimizer_updates == 5 * 1024
    assert ledger.likelihood_evaluations == 5


def test_parity_under_independence_matches_the_closed_form():
    for p in (0.6, 0.8, 0.95):
        for n in (0, 1, 2, 5):
            expected = sum(
                math.comb(n, k) * (1 - p) ** k * p ** (n - k)
                for k in range(0, n + 1, 2))
            assert abs(target.parity_under_independence(p, n) - expected) < 1e-9
    assert target.parity_under_independence(0.9, 0) == 1.0


def test_route_length_is_the_right_parity_exponent():
    """A false positive on a step with no event inverts the parity just as surely as a
    miss on a step with one, so the exponent is the number of steps, not of events."""
    rng = np.random.default_rng(0)
    length, p = 6, 0.84
    events = (rng.random((20_000, length)) < 0.3).astype(int)
    errors = rng.random((20_000, length)) > p
    estimated = np.abs(events - errors.astype(int))
    correct = (estimated.sum(1) % 2) == (events.sum(1) % 2)
    measured = correct.mean()
    by_length = target.parity_under_independence(p, length)
    by_events = float(np.mean([target.parity_under_independence(p, int(n))
                               for n in events.sum(1)]))
    assert abs(by_length - measured) < abs(by_events - measured)
    assert abs(by_length - measured) < 0.02


def test_required_detector_accuracy_is_monotone_and_non_degenerate():
    lengths = np.full(500, 6)
    low = target.required_detector_accuracy(lengths, 0.55)
    high = target.required_detector_accuracy(lengths, 0.75)
    assert 0.5 < low < high < 1.0


def test_detector_nll_is_finite_for_confident_correct_predictions():
    probability = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    truth = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    value = target.detector_nll(truth, probability)
    assert np.isfinite(value) and value >= 0.0


def test_public_event_derivation_is_exact_and_hidden_field_invariant():
    from structured_calibration import collect
    from belief_factorization import public_event
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

    trajectories = collect([61_000, 61_001], 2, 8, CANONICAL_APPEARANCE_SEED, 11)
    for trajectory in trajectories:
        rows = target.with_crossings(trajectory["rows"])
        derived = [target.derive_event_public(rows[t - 1] if t else None, rows[t])
                   for t in range(len(rows))]
        label = [public_event(rows[t - 1] if t else None, rows[t])
                 for t in range(len(rows))]
        assert derived == label
        honest = target.assert_derivation_ignores_hidden_state(
            target.derive_event_public, rows)
        assert honest["invariant"], honest["changed_under"]


def test_the_hidden_field_guard_catches_a_leaky_derivation():
    """A guard that passes both the honest and the leaky derivation is vacuous."""
    from structured_calibration import collect
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

    trajectories = collect([61_000, 61_001, 61_002], 2, 8, CANONICAL_APPEARANCE_SEED, 11)
    caught = 0
    for trajectory in trajectories:
        rows = target.with_crossings(trajectory["rows"])
        leaky = target.assert_derivation_ignores_hidden_state(
            target.derive_event_leaky, rows)
        caught += int(not leaky["invariant"])
    assert caught > 0, "the invariance guard has no detection power"


def test_population_manifest_records_members_not_just_a_size():
    import m2d_core as m2d
    population = m2d.build_population((90_000,), depth=4)
    manifest = core.population_manifest(population, "t", (90_000,))
    assert manifest["member_table"].shape[0] == manifest["rows"]
    assert len(manifest["member_routes"]) == manifest["rows"]
    assert manifest["rows_per_layout"] == {90_000: manifest["rows"]}
    smaller = core.population_manifest(
        m2d.Population(population.states, population.rows[:-1],
                       population.route_rows, population.crossing_steps),
        "t", (90_000,))
    assert smaller["member_digest"] != manifest["member_digest"]
