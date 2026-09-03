"""Regression pins for the M2F closure.

The load-bearing one is `test_relational_detector_is_translation_equivariant`, with its
calibration arm: the current structured detector MUST fail the same test. A detector that
reads absolute position can memorise training layouts, and that is the mechanism behind
the 1.0000/0.80 development/held-out split M2E measured.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/shwm"))

import m2d_core as m2d  # noqa: E402
import m2e_core as m2e  # noqa: E402
import m2f_core as core  # noqa: E402
import m2f_events as events  # noqa: E402


def row(seed, restart, likelihood, alias):
    return core.RestartRow(seed=seed, restart=restart,
                           training_log_likelihood=likelihood, alias_accuracy=alias,
                           phase_accuracy=1.0 if alias > 0.9 else 0.5,
                           transition_entropy=0.15, stay_minus_flip=1.9,
                           state_occupancy=1.0, belief_entropy=0.2,
                           collapsed=False, checkpoint=f"{seed}-{restart}")


def test_certificate_sees_only_training_quantities():
    parameters = list(inspect.signature(core.certify).parameters)
    assert parameters == ["best_training_log_likelihood", "tau"]
    assert core.certify(-0.001, -0.01) == "CERTIFIED"
    assert core.certify(-0.08, -0.01) == "UNRESOLVED_TRANSITION"


def test_fixed_k_is_a_prefix_argmax():
    rows = [row(1, 0, -0.5, 0.5), row(1, 1, -0.001, 1.0), row(1, 2, -0.9, 0.5)]
    assert core.fixed_k(rows, 1).restart == 0
    assert core.fixed_k(rows, 2).restart == 1
    assert core.fixed_k(rows, 3).restart == 1


def test_adaptive_stops_at_the_first_certifying_block_and_respects_k_max():
    solved_late = [row(1, r, -0.5 if r < 12 else -0.001, 0.5 if r < 12 else 1.0)
                   for r in range(32)]
    best, used, certificate = core.adaptive(solved_late, tau=-0.01, start=8, block=8,
                                            k_max=32)
    assert certificate == "CERTIFIED" and used == 16 and best.restart == 12
    never = [row(1, r, -0.5, 0.5) for r in range(32)]
    best, used, certificate = core.adaptive(never, tau=-0.01, start=8, block=8, k_max=32)
    assert certificate == "UNRESOLVED_TRANSITION" and used == 32


def test_choose_tau_separates_and_reports_non_separability_honestly():
    rows = [row(1, r, -0.001, 1.0) for r in range(5)]
    rows += [row(1, 5 + r, -0.08, 0.5) for r in range(5)]
    block = core.choose_tau(rows)
    assert block["separable"] is True
    assert -0.08 < block["tau"] < -0.001
    entangled = [row(1, 0, -0.001, 0.5), row(1, 1, -0.08, 1.0)]
    assert core.choose_tau(entangled)["separable"] is False


def test_no_generic_restart_reproduces_the_answer_initialisation():
    answer = m2e.initialisation_digest(m2e.ANSWER)
    for seed in range(21_000, 21_020):
        for restart in range(8):
            spec = core.generic_spec(seed, restart)
            assert m2e.initialisation_digest(spec.perturbation) != answer


def _sequence_with_offset(offset: int) -> np.ndarray:
    from structured_calibration import collect
    from belief_factorization import build_dataset
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

    items = build_dataset(collect([61_000], 1, 6, CANONICAL_APPEARANCE_SEED, 11), 5)
    x, _, _, _, _ = m2d.pad(items)
    moved = np.array(x, copy=True)
    moved[..., m2d.POSITION] += offset / m2d.GRID
    return x, moved


def test_relational_detector_is_translation_equivariant():
    x, moved = _sequence_with_offset(3)
    detector = events.RelationalDetector("full")
    assert np.allclose(detector.featurise(x), detector.featurise(moved)), (
        "the relational detector must not see absolute position")


def test_the_equivariance_test_has_power_against_the_structured_detector():
    """The calibration arm. If the current detector also passed, the test would be
    measuring nothing."""
    from m2d_coupling import EventDetector

    x, moved = _sequence_with_offset(3)
    detector = EventDetector("full")
    assert not np.allclose(detector.featurise(x), detector.featurise(moved))


def test_exact_derivation_detector_reproduces_the_public_event():
    from structured_calibration import collect
    from belief_factorization import build_dataset, public_event
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

    items = build_dataset(collect([61_000, 61_001], 2, 8, CANONICAL_APPEARANCE_SEED, 11), 5)
    x, _, e, m, _ = m2d.pad(items)
    probability = events.ExactDerivationDetector().probabilities(x)
    valid = m.astype(bool)
    valid[:, 0] = False
    assert np.array_equal((probability[valid] >= 0.5).astype(int),
                          e[valid].astype(int))


def test_non_identical_parity_formula_matches_simulation():
    """P(correct parity) = [1 + prod(1 - 2 e_t)] / 2, with per-step error rates that
    differ from each other -- which is why balanced accuracy must not be plugged in."""
    rng = np.random.default_rng(0)
    errors = np.array([0.05, 0.30, 0.12, 0.22, 0.08])
    draws = rng.random((200_000, len(errors))) < errors
    measured = float((draws.sum(axis=1) % 2 == 0).mean())
    predicted = float((1.0 + np.prod(1.0 - 2.0 * errors)) / 2.0)
    assert abs(measured - predicted) < 0.005
    naive = float((1.0 + (1.0 - 2.0 * errors.mean()) ** len(errors)) / 2.0)
    assert abs(naive - measured) > abs(predicted - measured)


def test_sequence_metrics_parity_and_burst():
    probability = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    truth = np.array([[0.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    block = events.sequence_metrics(probability, truth, np.array([5]))
    assert block["exact_route_sequence_accuracy"] == 0.0
    assert block["final_event_parity_accuracy"] == 0.0     # one error flips parity
    assert block["max_error_burst_length"] == 1
    assert block["mean_first_error_position"] == 2.0


def test_optional_dependency_marker_is_registered():
    import importlib.util
    conftest = REPO / "tests/conftest.py"
    assert conftest.exists()
    text = conftest.read_text()
    assert "optional_dependency" in text and "importlib.util.find_spec" in text
