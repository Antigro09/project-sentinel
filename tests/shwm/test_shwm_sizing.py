"""Scale-0 gate: a 50M model has 50M actual trainable parameters.

"Do not call nominal hidden width an actual parameter count" is the rule these
tests enforce. The authoritative number is counted off the built MLX model; the
closed form exists only to make the search cheap, and it has to agree exactly or
the search is solving a different problem than the one being run.
"""

from __future__ import annotations

import pytest

from sentinel.wm.latent_contract import ContractViolation, RepresentationKind
from sentinel.wm.matrix import (
    DIMENSION_CONTROL_WIDTHS,
    PARAMETER_TOLERANCE,
    PRIMARY_WIDTH,
    TRAINABLE_TARGETS,
    parameters_within_tolerance,
)
from sentinel.wm.models import build_model
from sentinel.wm.sizing import (
    WorldModelConfig,
    count_parameters,
    parameter_breakdown,
    solve_config,
)

ARMS = (RepresentationKind.CONTINUOUS, RepresentationKind.DISCRETE, RepresentationKind.HYBRID)
ENCODER_DIMENSION = 512
ACTION_COUNT = 4


def solved(arm: RepresentationKind, target: int, width: int):
    return solve_config(
        arm,
        target,
        encoder_dimension=ENCODER_DIMENSION,
        latent_width=width,
        action_count=ACTION_COUNT,
    )


@pytest.mark.parametrize("target", TRAINABLE_TARGETS)
@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_solves_inside_the_one_percent_tolerance(arm, target):
    sized = solved(arm, target, PRIMARY_WIDTH)
    assert parameters_within_tolerance(sized.counted, target)
    assert abs(sized.drift) < PARAMETER_TOLERANCE


@pytest.mark.parametrize("target", TRAINABLE_TARGETS)
@pytest.mark.parametrize("arm", ARMS)
def test_the_closed_form_equals_the_built_model_exactly(arm, target):
    sized = solved(arm, target, PRIMARY_WIDTH)
    model = build_model(sized.config, seed=6600)
    report = model.parameter_report()
    assert report["trainable_parameters"] == report["closed_form_parameters"] == sized.counted
    assert report["closed_form_agrees"] is True


@pytest.mark.parametrize("target", TRAINABLE_TARGETS)
def test_the_three_arms_are_matched_to_each_other_not_only_to_the_target(target):
    counts = {arm: solved(arm, target, PRIMARY_WIDTH).counted for arm in ARMS}
    smallest, largest = min(counts.values()), max(counts.values())
    assert largest - smallest <= target * PARAMETER_TOLERANCE, counts
    # In practice the solver lands far inside that: assert the tighter fact too,
    # so a regression that merely stays legal still shows up.
    assert (largest - smallest) / target < 1e-3, counts


@pytest.mark.parametrize("width", (PRIMARY_WIDTH,) + DIMENSION_CONTROL_WIDTHS)
def test_the_dimension_control_holds_parameters_fixed_while_width_moves(width):
    sized = solved(RepresentationKind.HYBRID, 50_000_000, width)
    assert sized.config.latent_width == width
    assert parameters_within_tolerance(sized.counted, 50_000_000)
    model = build_model(sized.config, seed=6600)
    assert model.actual_trainable_parameters() == sized.counted


def test_widths_that_differ_produce_configurations_that_differ():
    configs = {
        width: solved(RepresentationKind.HYBRID, 50_000_000, width).config
        for width in (256, 512, 1024)
    }
    assert len({c.digest for c in configs.values()}) == 3
    # Compensation is real: a wider latent must buy a smaller belief dimension.
    assert configs[256].belief_dimension > configs[1024].belief_dimension


def test_the_breakdown_sums_to_the_total_and_names_every_component():
    sized = solved(RepresentationKind.HYBRID, 50_000_000, PRIMARY_WIDTH)
    breakdown = parameter_breakdown(sized.config)
    assert sum(breakdown.values()) == count_parameters(sized.config)
    expected = {
        "projector_in",
        "projector_norm",
        "representation_continuous",
        "representation_codebook",
        "representation_readout",
        "action_embedding",
        "belief_gru",
        "dynamics_core",
        "head_next_latent",
        "head_event",
        "head_reward",
        "head_termination",
        "head_uncertainty",
        "head_inverse_action",
        "head_consistency",
        "head_boundary",
    }
    assert set(breakdown) == expected


def test_the_continuous_arm_has_no_codebook_and_the_discrete_arm_has_no_gaussian():
    continuous = parameter_breakdown(solved(RepresentationKind.CONTINUOUS, 50_000_000, 512).config)
    discrete = parameter_breakdown(solved(RepresentationKind.DISCRETE, 50_000_000, 512).config)
    assert "representation_codebook" not in continuous
    assert "representation_continuous" not in discrete


def test_a_target_that_cannot_be_reached_raises_instead_of_being_relabelled():
    with pytest.raises(ContractViolation):
        solve_config(
            RepresentationKind.CONTINUOUS,
            1_000,
            encoder_dimension=ENCODER_DIMENSION,
            latent_width=4096,
            action_count=ACTION_COUNT,
        )


def test_malformed_configurations_are_rejected():
    common = dict(
        representation=RepresentationKind.CONTINUOUS,
        encoder_dimension=8,
        latent_width=8,
        belief_dimension=8,
        core_width=8,
        core_depth=1,
        action_count=2,
    )
    WorldModelConfig(**common)
    with pytest.raises(ContractViolation):
        WorldModelConfig(**{**common, "latent_width": 0})
    with pytest.raises(ContractViolation):
        WorldModelConfig(**{**common, "core_depth": 0})
    with pytest.raises(ContractViolation):
        WorldModelConfig(**{**common, "representation": RepresentationKind.HYBRID, "latent_width": 7})
    with pytest.raises(ContractViolation):
        WorldModelConfig(**{**common, "representation": RepresentationKind.DISCRETE, "code_groups": 0})


def test_frozen_encoder_parameters_are_never_counted_as_trainable():
    """The 4B backbones are inherited capability and are reported apart.

    Nothing in the sizing path can see them, which is the structural version of
    that rule: a frozen backbone cannot accidentally inflate a trainable budget
    it was never part of.
    """
    sized = solved(RepresentationKind.HYBRID, 50_000_000, PRIMARY_WIDTH)
    assert "encoder" not in " ".join(parameter_breakdown(sized.config)).replace("encoder_dimension", "")
    assert sized.counted < 60_000_000  # not 4B + 50M
