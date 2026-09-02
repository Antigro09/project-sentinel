"""Regression pins for the M2D closure.

Two of these pin defects rather than capabilities, which is deliberate. The M2C
coupling imported a filter and never called it, and nothing in the suite noticed;
`test_m2c_coupling_never_called_the_selected_filter` fails the day someone quietly
edits that file, which is the only way a withdrawn result stays withdrawn.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/shwm"))

import m2d_core as core  # noqa: E402
from m2d_arm_identity import (  # noqa: E402
    SOURCE, called_names, function_named, imported_names, uses_xor_accumulation)


def test_feature_column_constants_match_the_encoder():
    core.check_feature_layout()


def test_m2c_coupling_never_called_the_selected_filter():
    """The finding that withdrew U7, pinned so it cannot be edited away silently."""
    tree = ast.parse(SOURCE.read_text())
    imports = imported_names(tree)
    aliases = {a for a, origin in imports.items() if origin.startswith("filter_stability.")}
    assert aliases, "the M2C coupling is expected to import from filter_stability"
    assert not (aliases & called_names(tree)), (
        "learned_event_coupling.py now calls the filter it imports; the M2C U7 "
        "withdrawal was based on it not doing so, so this needs re-deciding")
    assert uses_xor_accumulation(function_named(tree, "phase_from_learned_events"))


def test_symmetry_breaking_perturbation_is_the_xor_automaton():
    """The M2C docstring claimed this matrix encodes no XOR structure. It does."""
    anti = core.antisymmetric_two_state()

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    stay, flip = softmax(anti[0]), softmax(anti[1])
    assert stay[0, 0] > 0.7 and stay[1, 1] > 0.7, "event 0 must initialise as STAY"
    assert flip[0, 1] > 0.7 and flip[1, 0] > 0.7, "event 1 must initialise as FLIP"


def test_perturbation_is_invariant_under_simultaneous_state_relabelling():
    anti = core.antisymmetric_two_state()
    swap = np.array([[0, 1], [1, 0]])
    for event in range(2):
        assert np.allclose(swap.T @ anti[event] @ swap, anti[event])


def test_soft_parity_recursion_equals_cumsum_parity_on_hard_events():
    rng = np.random.default_rng(0)
    events = rng.integers(0, 2, (32, 7)).astype(np.float32)
    reset = rng.integers(0, 2, (32, 1)).astype(np.float32)
    shifted = np.concatenate([np.zeros((32, 1), np.float32), events[:, 1:]], axis=1)
    cumulative = np.remainder(reset + np.remainder(np.cumsum(shifted, axis=1), 2), 2)
    phase = reset[:, 0].copy()
    columns = [phase.copy()]
    for t in range(1, 7):
        phase = phase * (1 - events[:, t]) + (1 - phase) * events[:, t]
        columns.append(phase.copy())
    assert np.allclose(cumulative, np.stack(columns, axis=1))


def test_nested_bootstrap_is_centred_on_zero_for_identical_arms():
    rng = np.random.default_rng(1)
    n = 4000
    a = rng.random(n)
    result = core.hierarchical_paired_interval(
        a, a.copy(), np.repeat(np.arange(4), n // 4), rng.integers(0, 5, n),
        rng.integers(0, 30, n), resamples=200)
    assert result["delta"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_nested_bootstrap_detects_a_real_shift():
    rng = np.random.default_rng(2)
    n = 4000
    a = rng.random(n) + 0.5
    b = rng.random(n)
    result = core.hierarchical_paired_interval(
        a, b, np.repeat(np.arange(4), n // 4), rng.integers(0, 5, n),
        rng.integers(0, 30, n), resamples=200)
    assert result["excludes_zero"] and result["ci_low"] > 0


def test_pair_minimum_stratifier_keeps_both_directions_together():
    """Aliases share a layout and so share an initial polarity; their crossing counts
    always differ in parity. Stratifying on a row's own count would split the two
    directions of a pair and move the memoryless baseline off 0.5."""
    from m2d_core import AliasRow
    rows = [AliasRow(0, "k", 1, 0, 0, 1, 1, 2, 3, 4, 5, -1, 0),
            AliasRow(0, "k", 1, 0, 1, 0, 2, 1, 4, 3, 5, -1, 1)]
    population = core.Population(states=[], rows=rows, route_rows=[], crossing_steps=[])
    strata = core.stratify(population)
    assert strata["changes"][0] == strata["changes"][1] == 3
    assert strata["changes_self"][0] != strata["changes_self"][1]


def test_event_corruptions_preserve_the_marginal_they_claim_to():
    from m2d_coupling import corrupt
    rng = np.random.default_rng(3)
    base = (rng.random((40, 7)) < 0.3).astype(np.float32)
    base[:, 0] = 0.0
    lengths = np.full(40, 7)
    shuffled = corrupt("6_cross_episode_shuffle", base, lengths, seed=7)
    assert shuffled.sum() == base.sum()
    permuted = corrupt("7_positionwise_permutation", base, lengths, seed=7)
    assert permuted.sum() == base.sum()
    assert corrupt("8_constant", base, lengths, seed=7).sum() == 0.0


@pytest.mark.parametrize("channel", core.__dict__.get("_", []) or [
    "future_observation", "future_outcome", "target_displacement",
    "future_action_result", "evaluator_phase", "simulator_step", "provenance_digest"])
def test_each_forbidden_channel_guard_catches_its_own_leak(channel):
    """A guard that passes both the honest and the mutated pipeline is vacuous."""
    import m2d_dataflow as flow
    from structured_calibration import collect
    from belief_factorization import build_dataset
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

    items = build_dataset(collect([61_000, 61_001], 1, 6, CANONICAL_APPEARANCE_SEED, 11), 5)
    records = flow.build_records(items)
    width = 2 * core.FEATURE_WIDTH + 1 + len(flow.CHANNELS) + 3
    detector = flow.Detector(width)
    guard = flow.channel_guard(channel)
    assert guard(flow.Pipeline(), records, detector), "guard must pass the honest pipeline"
    leaked = flow.Pipeline(leaks=frozenset({channel}))
    assert not guard(leaked, records, detector), "guard must fail its own planted leak"
