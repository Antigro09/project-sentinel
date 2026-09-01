"""Scale-1A-1 F: six arms whose differences are attributable.

Each ablation must remove exactly one thing and hold everything else fixed. The
failure this guards against is subtle and common: an ablation that also shrinks
the model, so "recurrence matters" and "this model is smaller" become the same
measurement.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from sentinel.wm.arms import (
    ABLATION_ARMS,
    ARMS,
    ActionMode,
    ArmSpec,
    MASKED_ACTION_INDEX,
    apply_action_mode,
    assert_shuffle_is_trajectory_safe,
)
from sentinel.wm.latent_contract import ContractViolation, RepresentationKind
from sentinel.wm.models import build_model
from sentinel.wm.sizing import solve_config


def model():
    sized = solve_config(
        RepresentationKind.CONTINUOUS, 50_000_000,
        encoder_dimension=256, latent_width=512, action_count=4,
    )
    return build_model(sized.config, seed=6600)


def batch(shape=(2, 6, 256)):
    mx.random.seed(0)
    return (
        mx.random.normal(shape).astype(mx.bfloat16),
        mx.zeros(shape[:2], dtype=mx.int32),
        mx.zeros((*shape[:2], 1), dtype=mx.bfloat16),
    )


# ---- the arm set -------------------------------------------------------------------


def test_the_six_arms_are_exactly_the_frozen_set():
    assert len(ARMS) == 6
    assert [a.name for a in ARMS] == [
        "continuous_action_recurrent",
        "continuous_no_action_recurrent",
        "continuous_shuffled_action_recurrent",
        "continuous_action_no_recurrence",
        "discrete_action_recurrent",
        "hybrid_action_recurrent",
    ]


def test_the_continuous_arm_supplies_every_ablation():
    """Action conditioning and recurrence must be attributable somewhere."""
    continuous = [a for a in ARMS if a.representation is RepresentationKind.CONTINUOUS]
    modes = {a.action_mode for a in continuous}
    assert modes == {ActionMode.CONDITIONED, ActionMode.MASKED, ActionMode.SHUFFLED}
    assert {a.recurrent for a in continuous} == {True, False}
    assert ABLATION_ARMS == {a.name for a in continuous if a.name.count("_") > 2} - {
        "continuous_action_recurrent"
    } or ABLATION_ARMS <= {a.name for a in continuous}


def test_the_other_representations_ship_only_their_headline_arm():
    """Their ablations are owed only if they turn out to be the winner."""
    for representation in (RepresentationKind.DISCRETE, RepresentationKind.HYBRID):
        arms = [a for a in ARMS if a.representation is representation]
        assert len(arms) == 1
        assert arms[0].action_mode is ActionMode.CONDITIONED
        assert arms[0].recurrent is True


def test_arm_digests_separate_every_arm():
    assert len({a.digest for a in ARMS}) == len(ARMS)


# ---- action modes -------------------------------------------------------------------


def test_masking_removes_action_information_and_keeps_the_module():
    actions = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
    masked, report = apply_action_mode(actions, ActionMode.MASKED, seed=1)
    assert (masked == MASKED_ACTION_INDEX).all()
    assert report["information_removed"] is True
    assert report["marginal_preserved"] is False  # removing information changes it


def test_shuffling_preserves_the_action_marginal_exactly():
    actions = np.array([[0, 1, 1, 3], [3, 2, 2, 0]], dtype=np.int32)
    shuffled, report = apply_action_mode(actions, ActionMode.SHUFFLED, seed=7)
    assert report["marginal_preserved"] is True
    assert report["alignment_destroyed"] is True
    assert not np.array_equal(actions, shuffled) or actions.shape[1] < 2


def test_the_shuffle_stays_inside_a_trajectory():
    actions = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    shuffled, _ = apply_action_mode(actions, ActionMode.SHUFFLED, seed=3)
    assert_shuffle_is_trajectory_safe(actions, shuffled)
    # A cross-trajectory shuffle changes each row's multiset and must be caught.
    crossed = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.int32)
    with pytest.raises(ContractViolation, match="crossed a trajectory boundary"):
        assert_shuffle_is_trajectory_safe(actions, crossed)


def test_conditioned_actions_are_untouched():
    actions = np.array([[0, 1, 2, 3]], dtype=np.int32)
    same, report = apply_action_mode(actions, ActionMode.CONDITIONED, seed=1)
    assert np.array_equal(actions, same)
    assert report["marginal_preserved"] is True


# ---- recurrence ---------------------------------------------------------------------


def test_removing_recurrence_keeps_every_parameter():
    """The ablation must not double as a capacity reduction."""
    network = model()
    before = network.actual_trainable_parameters()
    features, actions, rewards = batch()
    network(features, actions, rewards, key=mx.random.key(1), recurrent=False)
    assert network.actual_trainable_parameters() == before


def test_the_non_recurrent_arm_cannot_see_earlier_steps():
    network = model()
    features, actions, rewards = batch()
    mx.random.seed(1)
    altered = mx.concatenate(
        [mx.random.normal((2, 1, 256)).astype(mx.bfloat16), features[:, 1:]], axis=1
    )
    key = mx.random.key(1)

    late_a = network(features, actions, rewards, key=key, recurrent=False).belief[:, 3]
    late_b = network(altered, actions, rewards, key=key, recurrent=False).belief[:, 3]
    mx.eval(late_a, late_b)
    assert bool(mx.allclose(late_a, late_b).item()), (
        "a change at step 0 moved the non-recurrent belief at step 3; prior state is "
        "still reachable and the ablation is not doing what it claims"
    )


def test_the_recurrent_arm_does_see_earlier_steps():
    """Calibration arm for the test above: if recurrence changed nothing, the
    ablation would pass for the wrong reason."""
    network = model()
    features, actions, rewards = batch()
    mx.random.seed(1)
    altered = mx.concatenate(
        [mx.random.normal((2, 1, 256)).astype(mx.bfloat16), features[:, 1:]], axis=1
    )
    key = mx.random.key(1)
    late_a = network(features, actions, rewards, key=key, recurrent=True).belief[:, 3]
    late_b = network(altered, actions, rewards, key=key, recurrent=True).belief[:, 3]
    mx.eval(late_a, late_b)
    assert not bool(mx.allclose(late_a, late_b).item())


def test_the_two_recurrence_modes_produce_the_same_shapes():
    network = model()
    features, actions, rewards = batch()
    key = mx.random.key(1)
    a = network(features, actions, rewards, key=key, recurrent=True)
    b = network(features, actions, rewards, key=key, recurrent=False)
    for field in ("belief", "core", "next_latent", "event_logits", "reward", "termination"):
        assert getattr(a, field).shape == getattr(b, field).shape


def test_every_arm_solves_to_the_same_parameter_target():
    """Matched budget across representations is the precondition for comparing
    them at all."""
    from sentinel.wm.matrix import parameters_within_tolerance

    counts = {}
    for arm in ARMS:
        sized = solve_config(
            arm.representation, 50_000_000,
            encoder_dimension=256, latent_width=512, action_count=4,
        )
        counts[arm.name] = sized.counted
        assert parameters_within_tolerance(sized.counted, 50_000_000)
    assert max(counts.values()) - min(counts.values()) < 50_000_000 * 1e-3
