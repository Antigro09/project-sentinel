"""Scale-0 gate: planning is counted, and no action reaches the world uncounted.

Scale 0 does not ask whether planning works. It asks whether the accounting can
be trusted, because giving one arm more planner compute is the cheapest way to
manufacture a world-model advantage. Every planner here spends exactly the
matrix's candidate budget, and the numbers in the report come from the object
that did the spending.

The verifier half tests the asymmetry that matters: required probes are the
evaluator's and are always executed; requested probes are the model's and can
only add coverage. A model that could shrink its own evidence set could
authorise itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.env.adapters.base import ProbeSet
from sentinel.wm import matrix as M
from sentinel.wm.authority import ActionAuthority, AuthorityDenied, AuthorityGate
from sentinel.wm.latent_contract import (
    ContractViolation,
    TransitionPrediction,
    UncertaintyTriple,
)
from sentinel.wm.planner_bridge import (
    BeamPlanner,
    CEMPlanner,
    CountingRollout,
    FakeDynamicsRollout,
    MCTSPlanner,
    planner_registry,
)
from sentinel.wm.verifier_bridge import (
    ControlThresholds,
    Decision,
    DecisionController,
    ObservableVerifierBridge,
    VerificationContext,
    authorize_if_verified,
)
from sentinel.wm.versioning import digest_array, digest_of

PLANNERS = [BeamPlanner(), CEMPlanner(), MCTSPlanner()]


# ---- planner accounting --------------------------------------------------------


@pytest.mark.parametrize("planner", PLANNERS, ids=lambda p: p.name)
def test_every_planner_spends_exactly_the_candidate_budget(planner):
    fake = FakeDynamicsRollout()
    rollout = CountingRollout(fake)
    planner.plan(rollout, fake.root(), horizon=5, candidates=64)
    account = rollout.account
    assert account.candidate_sequences == 64
    assert account.score_calls == 64
    assert account.invocations == 1
    assert account.model_calls > 0


@pytest.mark.parametrize("planner", PLANNERS, ids=lambda p: p.name)
def test_planning_is_deterministic(planner):
    fake = FakeDynamicsRollout()
    first = planner.plan(CountingRollout(fake), fake.root(), 5, 64)
    second = planner.plan(CountingRollout(fake), fake.root(), 5, 64)
    assert first.digest == second.digest
    assert first.actions == second.actions


@pytest.mark.parametrize("planner", PLANNERS, ids=lambda p: p.name)
def test_the_planned_sequence_has_the_requested_horizon(planner):
    fake = FakeDynamicsRollout()
    for horizon in M.PLANNER_HORIZONS:
        plan = planner.plan(CountingRollout(fake), fake.root(), horizon, 8)
        assert len(plan.actions) == horizon


def test_the_frozen_planner_workload_totals_the_matrix_figure():
    fake = FakeDynamicsRollout()
    planner = BeamPlanner()
    rollout = CountingRollout(fake)
    for horizon in M.PLANNER_HORIZONS:
        for _ in range(2):  # two of the hundred, to keep the unit test quick
            planner.plan(rollout, fake.root(), horizon, M.PLANNER_CANDIDATES_PER_INVOCATION)
    assert rollout.account.invocations == len(M.PLANNER_HORIZONS) * 2
    assert rollout.account.candidate_sequences == (
        len(M.PLANNER_HORIZONS) * 2 * M.PLANNER_CANDIDATES_PER_INVOCATION
    )
    # The full frozen workload scales from here by exactly 50x.
    assert (
        len(M.PLANNER_HORIZONS)
        * M.PLANNER_INVOCATIONS_PER_HORIZON
        * M.PLANNER_CANDIDATES_PER_INVOCATION
        == M.PLANNER_CANDIDATES_TOTAL
    )


def test_the_uncertainty_penalty_actually_changes_the_chosen_plan():
    """Otherwise the penalty is decoration and the controller has no lever."""
    indifferent = FakeDynamicsRollout(uncertainty_growth=0.0, constraint_action=None)
    penalised = FakeDynamicsRollout(uncertainty_growth=5.0, constraint_action=None)
    planner = BeamPlanner()
    a = planner.plan(CountingRollout(indifferent), indifferent.root(), 5, 64)
    b = planner.plan(CountingRollout(penalised), penalised.root(), 5, 64)
    assert a.utility.uncertainty_penalty == 0.0
    assert b.utility.uncertainty_penalty > 0.0
    assert b.utility.total < a.utility.total


def test_the_constraint_penalty_pushes_the_plan_away_from_the_costly_action():
    free = FakeDynamicsRollout(constraint_action=None)
    costly = FakeDynamicsRollout(constraint_action=3)
    planner = BeamPlanner()
    unconstrained = planner.plan(CountingRollout(free), free.root(), 8, 64)
    constrained = planner.plan(CountingRollout(costly), costly.root(), 8, 64)
    assert constrained.actions.count(3) <= unconstrained.actions.count(3)


def test_the_registry_names_all_three_planners():
    assert set(planner_registry()) == {"beam", "cem", "mcts"}


# ---- verifier bridge -------------------------------------------------------------


def prediction(events=None, action=1) -> TransitionPrediction:
    return TransitionPrediction(
        next_latent=digest_array(np.zeros(4, dtype=np.float32)),
        event_distribution=events if events is not None else {"ACTION_SUCCEEDED": 1.0},
        reward_mean=0.0,
        reward_variance=0.0,
        termination_probability=0.0,
        uncertainty=UncertaintyTriple(0.0, 0.0, 0.0),
        rollout_support_scope="development",
        model_version=digest_of("model"),
        action=action,
    )


def context(extra=("diagnostic_probe",)) -> VerificationContext:
    return VerificationContext(
        episode_id="ep",
        step=3,
        available_probes=tuple(M.REQUIRED_PROBES) + tuple(extra),
        required_probes=M.REQUIRED_PROBES,
    )


def actual_probes(**overrides) -> ProbeSet:
    values = {
        "reward": 0.0,
        "termination": False,
        "goal_progress": 0.5,
        "constraint_violation": False,
        "action_succeeded": True,
        "observable_signature": 5,
        "diagnostic_probe": 1,
    }
    values.update(overrides)
    return ProbeSet(values)


def bridge() -> ObservableVerifierBridge:
    return ObservableVerifierBridge(required=M.REQUIRED_PROBES)


def full_prediction() -> dict:
    return {
        "reward": 0.0,
        "termination": False,
        "goal_progress": 0.5,
        "constraint_violation": False,
        "action_succeeded": True,
        "observable_signature": 5,
    }


def test_required_probes_are_always_executed_even_when_the_model_asks_for_none():
    result = bridge().verify(
        prediction(events={"UNKNOWN_EVENT": 1.0}), full_prediction(), actual_probes(), context()
    )
    assert set(M.REQUIRED_PROBES) <= set(result.accepted_observables) | set(result.rejected_observables)
    assert result.required_probe_names == tuple(sorted(M.REQUIRED_PROBES))


def test_model_requested_probes_can_only_add_coverage():
    verifier = bridge()
    quiet = verifier.probe_plan(prediction(events={"UNKNOWN_EVENT": 1.0}), context())
    noisy = verifier.probe_plan(
        prediction(events={"CONSTRAINT_VIOLATED": 0.5, "GOAL_PROGRESS_CHANGED": 0.5}), context()
    )
    assert set(quiet[0]) <= set(noisy[0])
    assert set(M.REQUIRED_PROBES) <= set(quiet[0])


def test_a_model_cannot_shrink_the_required_set_by_predicting_nothing():
    verifier = bridge()
    executed, _ = verifier.probe_plan(prediction(events={}), context())
    assert set(M.REQUIRED_PROBES) <= set(executed)


def test_an_observable_mismatch_becomes_a_counterexample():
    result = bridge().verify(
        prediction(),
        {**full_prediction(), "observable_signature": 99},
        actual_probes(),
        context(),
    )
    assert "observable_signature" in result.rejected_observables
    names = {c.probe_name for c in result.counterexamples}
    assert names == {"observable_signature"}
    counterexample = result.counterexamples[0]
    assert counterexample.predicted == 99 and counterexample.actual == 5


def test_a_required_probe_with_no_prediction_counts_against_the_model():
    partial = {k: v for k, v in full_prediction().items() if k != "reward"}
    result = bridge().verify(prediction(), partial, actual_probes(), context())
    assert "reward" in result.rejected_observables
    assert any(c.predicted is None for c in result.counterexamples)


def test_accuracy_and_coverage_move_independently():
    perfect_narrow = bridge().verify(
        prediction(events={"UNKNOWN_EVENT": 1.0}), full_prediction(), actual_probes(), context()
    )
    assert perfect_narrow.accuracy == pytest.approx(1.0)
    assert perfect_narrow.coverage < 1.0  # the diagnostic probe was not run


def test_a_non_injective_probe_hides_a_latent_mismatch():
    """Lemma 4's boundary, made concrete.

    Two different underlying states share a probe value, so an exact verifier
    accepts a prediction that is wrong about the state. Verifier correctness and
    probe coverage are therefore two numbers, not one.
    """
    constant_probe = ProbeSet({**actual_probes().values, "observable_signature": 0})
    verifier = bridge()
    first = verifier.verify(
        prediction(), {**full_prediction(), "observable_signature": 0}, constant_probe, context()
    )
    assert first.rejected_observables == ()
    assert "diagnostic_probe" in first.unprobed_observables


def test_constraint_violations_are_reported_separately_from_rejections():
    result = bridge().verify(
        prediction(),
        {**full_prediction(), "constraint_violation": True},
        actual_probes(constraint_violation=True),
        context(),
    )
    assert result.constraint_violations == ("constraint_violation",)
    assert "constraint_violation" in result.accepted_observables


def test_a_context_missing_a_required_probe_is_refused():
    with pytest.raises(ContractViolation):
        VerificationContext(
            episode_id="ep",
            step=0,
            available_probes=("reward",),
            required_probes=M.REQUIRED_PROBES,
        )


# ---- controller and authority -----------------------------------------------------


def test_the_controller_separates_inadequacy_from_ordinary_uncertainty():
    controller = DecisionController()
    assert controller.decide(UncertaintyTriple(0.0, 0.0, 0.0), 0.0) is Decision.ACT
    assert controller.decide(UncertaintyTriple(0.0, 0.9, 0.0), 0.0) is Decision.ASK
    assert controller.decide(UncertaintyTriple(0.0, 0.9, 0.9), 0.0) is Decision.EXPAND_REPRESENTATION
    assert controller.decide(UncertaintyTriple(0.0, 0.0, 0.0), 0.9) is Decision.RUN_TEST
    assert (
        controller.decide(UncertaintyTriple(0.0, 0.0, 0.0), 0.9, can_test=False) is Decision.ABSTAIN
    )
    assert controller.decide(UncertaintyTriple(0.0, 0.9, 0.0), 0.0, can_ask=False) is Decision.OBSERVE


def test_the_controller_ledger_reports_act_and_abstain_rates():
    controller = DecisionController()
    for _ in range(3):
        controller.decide(UncertaintyTriple(0.0, 0.0, 0.0), 0.0)
    controller.decide(UncertaintyTriple(0.0, 0.0, 0.0), 0.9, can_test=False)
    ledger = controller.ledger()
    assert ledger["act_rate"] == pytest.approx(0.75)
    assert ledger["abstain_rate"] == pytest.approx(0.25)
    assert ledger["thresholds"] == ControlThresholds().canonical_dict()


def test_the_only_route_to_an_action_is_a_verified_plan():
    gate = AuthorityGate(required_probes=M.REQUIRED_PROBES)
    result = bridge().verify(prediction(), full_prediction(), actual_probes(), context())
    token = authorize_if_verified(gate, Decision.ACT, 1, result)
    assert token.authority is ActionAuthority.VERIFIED_PLAN
    assert gate.consume(token, 1) is ActionAuthority.VERIFIED_PLAN


@pytest.mark.parametrize(
    "decision",
    [d for d in Decision if d is not Decision.ACT],
)
def test_every_non_act_decision_is_denied_a_token(decision):
    gate = AuthorityGate(required_probes=M.REQUIRED_PROBES)
    result = bridge().verify(prediction(), full_prediction(), actual_probes(), context())
    with pytest.raises(AuthorityDenied):
        authorize_if_verified(gate, decision, 1, result)
    assert gate.denied == 1


def test_a_plan_that_skipped_a_required_probe_is_denied():
    gate = AuthorityGate(required_probes=M.REQUIRED_PROBES)
    with pytest.raises(AuthorityDenied, match="evaluator-required probe"):
        gate.authorize_plan(
            1, executed_probes=("reward",), verifier_version=digest_of("verifier")
        )


def test_the_verifier_ledger_counts_what_it_did():
    verifier = bridge()
    verifier.verify(prediction(), full_prediction(), actual_probes(), context())
    verifier.verify(
        prediction(), {**full_prediction(), "reward": 9.0}, actual_probes(), context()
    )
    ledger = verifier.ledger()
    assert ledger["verifications"] == 2
    assert ledger["rejections"] == 1
    assert ledger["counterexamples"] == 1
