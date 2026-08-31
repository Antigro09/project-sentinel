"""Scale-0 gate: nine components, one total, and no silent zeros.

The plumbing requirements are exactly three: the total is the weighted sum,
disabling a component removes only that component, and a non-finite value stops
the run. The fourth thing tested here is coverage -- a boundary term with no
branch pairs reports zero *and* says it measured nothing, because a zero that
means "nothing to score" and a zero that means "scored and satisfied" are
different facts.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from sentinel.wm.latent_contract import ContractViolation, RepresentationKind
from sentinel.wm.metrics import (
    action_effect_discrimination,
    code_utilisation,
    event_accuracy_and_coverage,
    fitted_sensitivity,
    gradient_global_norm,
    rollout_divergence_by_horizon,
)
from sentinel.wm.models import build_model
from sentinel.wm.objective import (
    COMPONENT_NAMES,
    NONNEGATIVE_COMPONENTS,
    NonFiniteLoss,
    ObjectiveBatch,
    ObjectiveConfig,
    compute_objective,
    finalise,
)
from sentinel.wm.sizing import solve_config

BATCH, TIME, ENCODER = 4, 8, 128


def fixture(arm=RepresentationKind.HYBRID, *, pairs=((0, 1, 1, 1),)):
    sized = solve_config(
        arm, 50_000_000, encoder_dimension=ENCODER, latent_width=512, action_count=4
    )
    model = build_model(sized.config, seed=6600)
    mx.random.seed(11)
    batch = ObjectiveBatch(
        features=mx.random.normal((BATCH, TIME, ENCODER)).astype(mx.bfloat16),
        actions=mx.random.randint(0, 4, (BATCH, TIME)),
        previous_rewards=mx.zeros((BATCH, TIME, 1), dtype=mx.bfloat16),
        rewards=mx.random.normal((BATCH, TIME)),
        terminations=mx.zeros((BATCH, TIME)),
        event_targets=mx.random.randint(0, 12, (BATCH, TIME)),
        boundary_pairs=pairs,
    )
    output = model(batch.features, batch.actions, batch.previous_rewards)
    return model, batch, output


def evaluate(model, output, batch, config):
    total, components, coverage, extra = compute_objective(model, output, batch, config)
    return finalise(total, components, coverage, config, extra)


def test_all_nine_declared_components_are_present():
    assert len(COMPONENT_NAMES) == 9
    model, batch, output = fixture()
    result = evaluate(model, output, batch, ObjectiveConfig())
    assert set(result.components) == set(COMPONENT_NAMES)


def test_the_total_is_exactly_the_weighted_sum():
    model, batch, output = fixture()
    weights = {name: float(i + 1) for i, name in enumerate(COMPONENT_NAMES)}
    config = ObjectiveConfig(weights=weights)
    result = evaluate(model, output, batch, config)
    expected = sum(weights[name] * result.components[name] for name in COMPONENT_NAMES)
    assert result.metrics["loss/total"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    for name in COMPONENT_NAMES:
        assert result.weighted[name] == pytest.approx(weights[name] * result.components[name])


@pytest.mark.parametrize("dropped", COMPONENT_NAMES)
def test_disabling_a_component_removes_only_that_component(dropped):
    model, batch, output = fixture()
    full = ObjectiveConfig()
    reduced = full.without(dropped)
    before = evaluate(model, output, batch, full)
    after = evaluate(model, output, batch, reduced)
    assert dropped not in after.components
    assert f"loss/{dropped}" not in after.metrics
    assert f"coverage/{dropped}" not in after.metrics
    assert set(before.components) - set(after.components) == {dropped}
    for name in after.components:
        assert after.components[name] == pytest.approx(before.components[name], rel=1e-5, abs=1e-6)
    assert after.metrics["loss/total"] == pytest.approx(
        before.metrics["loss/total"] - before.weighted[dropped], rel=1e-5, abs=1e-5
    )


def test_a_non_finite_component_stops_the_run():
    model, batch, output = fixture()
    total, components, coverage, _ = compute_objective(model, output, batch, ObjectiveConfig())
    components["next"] = mx.array(float("nan"))
    with pytest.raises(NonFiniteLoss):
        finalise(total, components, coverage, ObjectiveConfig())
    components["next"] = mx.array(float("inf"))
    with pytest.raises(NonFiniteLoss):
        finalise(total, components, coverage, ObjectiveConfig())


def test_a_total_that_disagrees_with_its_parts_is_rejected():
    model, batch, output = fixture()
    total, components, coverage, _ = compute_objective(model, output, batch, ObjectiveConfig())
    with pytest.raises(ContractViolation, match="does not equal"):
        finalise(total + 10.0, components, coverage, ObjectiveConfig())


def test_the_boundary_term_distinguishes_three_different_zeros():
    """No pairs, pairs that are already separated, and a genuine penalty."""
    model, batch, output = fixture(pairs=())
    empty = evaluate(model, output, batch, ObjectiveConfig())
    assert empty.components["boundary"] == 0.0
    assert empty.coverage["boundary"] == 0
    assert math.isnan(empty.metrics["boundary/mean_pair_distance"])

    model, batch, output = fixture(pairs=((0, 1, 1, 1), (2, 3, 3, 3)))
    separated = evaluate(model, output, batch, ObjectiveConfig())
    assert separated.coverage["boundary"] == 2
    assert separated.metrics["boundary/mean_pair_distance"] > 0.0
    assert separated.metrics["boundary/active_pairs"] >= 0.0


def test_the_boundary_hinge_actually_fires_on_a_collapsed_pair():
    """Calibration arm: a term that can never fire is measuring nothing.

    A pair compared against itself has distance zero, which is exactly the
    collapse the term exists to penalise, so the hinge must reach the margin.
    """
    from sentinel.wm.objective import BOUNDARY_MARGIN, boundary_separation

    model, batch, output = fixture(pairs=())
    collapsed = ((0, 1, 0, 1),)
    value, count, active, distance = boundary_separation(output, collapsed)
    mx.eval(value)
    assert count == 1 and active == 1
    assert distance == pytest.approx(0.0, abs=1e-3)
    assert float(value.item()) == pytest.approx(BOUNDARY_MARGIN, abs=1e-3)


def test_the_boundary_metric_space_is_normalised_so_the_margin_means_something():
    """On the unit sphere every pair distance lies in [0, 2]."""
    from sentinel.wm.objective import boundary_separation

    model, batch, output = fixture(pairs=())
    pairs = tuple((0, t, 1, t) for t in range(TIME))
    _, count, _, distance = boundary_separation(output, pairs)
    assert count == TIME
    assert 0.0 <= distance <= 2.0


def test_the_nonnegativity_precondition_is_measured_not_assumed():
    """Two components are Gaussian NLLs, so the Lean lemma may not apply."""
    assert set(COMPONENT_NAMES) - NONNEGATIVE_COMPONENTS == {"reward", "calibration"}
    model, batch, output = fixture()
    result = evaluate(model, output, batch, ObjectiveConfig())
    for name in NONNEGATIVE_COMPONENTS & set(result.components):
        assert result.components[name] >= 0.0, name
    assert "loss/nonnegativity_precondition_holds" in result.metrics
    assert result.metrics["loss/nonnegativity_precondition_holds"] in (0.0, 1.0)


def test_negative_weights_are_refused():
    with pytest.raises(ContractViolation):
        ObjectiveConfig(weights={**{n: 1.0 for n in COMPONENT_NAMES}, "next": -1.0})


def test_unknown_components_are_refused():
    with pytest.raises(ContractViolation):
        ObjectiveConfig(weights={"not_a_component": 1.0})
    with pytest.raises(ContractViolation):
        ObjectiveConfig(enabled=frozenset({"not_a_component"}))


def test_the_objective_config_digest_changes_with_weights_and_membership():
    base = ObjectiveConfig()
    assert base.digest != base.without("boundary").digest
    assert base.digest != ObjectiveConfig(weights={**base.weights, "next": 2.0}).digest


# ---- metrics -----------------------------------------------------------------


def test_rollout_divergence_is_reported_per_horizon_and_fits_a_sensitivity():
    model, batch, output = fixture()
    divergence = rollout_divergence_by_horizon(model, output, batch.actions, (1, 2, 3))
    assert set(divergence) == {1, 2, 3}
    assert all(v >= 0.0 for v in divergence.values())
    sensitivity = fitted_sensitivity(divergence)
    assert math.isfinite(sensitivity) and sensitivity > 0.0


def test_event_accuracy_and_coverage_are_separate_numbers():
    model, batch, output = fixture()
    always_abstain = mx.zeros_like(output.event_logits)
    always_abstain = always_abstain + mx.eye(12)[10] * 100.0  # UNKNOWN_EVENT
    stats = event_accuracy_and_coverage(always_abstain, batch.event_targets, abstain_index=10)
    assert stats["event_coverage"] == 0.0
    assert math.isnan(stats["event_accuracy"]), "an abstaining head must not score perfect accuracy"


def test_code_utilisation_is_empty_for_the_continuous_arm_and_bounded_otherwise():
    _, _, continuous = fixture(RepresentationKind.CONTINUOUS)
    assert code_utilisation(continuous.code_logits, 32, 32) == {}
    model, batch, discrete = fixture(RepresentationKind.DISCRETE)
    stats = code_utilisation(discrete.code_logits, 32, 32)
    assert 0.0 < stats["code_utilisation"] <= 1.0
    assert stats["code_entropy_nats"] >= 0.0


def test_action_effect_discrimination_returns_a_bounded_score():
    model, batch, output = fixture()
    score = action_effect_discrimination(model, output, batch.actions, 4)
    assert 0.0 <= score <= 1.0


def test_gradient_norm_is_computable_and_finite():
    import mlx.nn as nn

    model, batch, output = fixture()

    def loss_fn(model):
        out = model(batch.features, batch.actions, batch.previous_rewards)
        total, components, coverage, _ = compute_objective(model, out, batch, ObjectiveConfig())
        return total

    value, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(value, grads)
    norm = gradient_global_norm(grads)
    assert math.isfinite(norm) and norm > 0.0


# ---- no parameter is dead ------------------------------------------------------


@pytest.mark.parametrize(
    "arm",
    [RepresentationKind.CONTINUOUS, RepresentationKind.DISCRETE, RepresentationKind.HYBRID],
)
def test_every_trainable_tensor_receives_gradient_under_the_full_objective(arm):
    """Matched parameter counts mean nothing if some of the parameters cannot learn.

    The one documented exception is the boundary head: its hinge is inactive
    while nothing is collapsed, which is the term's intended semantics rather
    than a wiring fault. The next test checks that head separately.
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    model, batch, _ = fixture(arm)

    def loss_fn(model):
        out = model(batch.features, batch.actions, batch.previous_rewards)
        total, _, _, _ = compute_objective(model, out, batch, ObjectiveConfig())
        return total

    _, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(grads)
    dead = [
        name
        for name, tensor in tree_flatten(grads)
        if float(mx.sum(mx.abs(tensor.astype(mx.float32))).item()) == 0.0
    ]
    assert set(dead) <= {"head_boundary.weight", "head_boundary.bias"}, dead


def test_the_boundary_head_is_wired_even_though_its_hinge_starts_inactive():
    """Calibration arm for the exception above.

    Raising the margin past the initialisation distance activates the hinge; if
    the head then still received no gradient, the exception in the previous test
    would be hiding a genuinely disconnected tensor.
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    import sentinel.wm.objective as objective_module

    model, batch, output = fixture(pairs=((0, 1, 1, 1), (2, 3, 3, 3)))
    _, _, active, distance = objective_module.boundary_separation(output, batch.boundary_pairs)
    assert active == 0 and distance > objective_module.BOUNDARY_MARGIN

    original = objective_module.BOUNDARY_MARGIN
    objective_module.BOUNDARY_MARGIN = distance + 0.25
    try:

        def loss_fn(model):
            out = model(batch.features, batch.actions, batch.previous_rewards)
            value, _, _, _ = objective_module.boundary_separation(out, batch.boundary_pairs)
            return value

        value, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(value, grads)
        boundary_grads = {
            name: float(mx.sum(mx.abs(tensor.astype(mx.float32))).item())
            for name, tensor in tree_flatten(grads)
            if name.startswith("head_boundary")
        }
    finally:
        objective_module.BOUNDARY_MARGIN = original

    assert float(value.item()) > 0.0
    assert set(boundary_grads) == {"head_boundary.weight", "head_boundary.bias"}
    assert all(v > 0.0 for v in boundary_grads.values()), boundary_grads
