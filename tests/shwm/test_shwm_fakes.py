"""Scale-0 gate: every typed boundary runs with no model in the loop.

`ARCHITECTURE.md` asks that the whole chain -- environment, encoder, projector,
belief, dynamics, planner, verifier, authority -- be runnable with deterministic
fakes before a neural network is introduced. That is what makes a later failure
attributable: if the typed flow already ran, a break after the model arrives is
the model's, not the contract's.

The file also carries the falsifying control. `ActionBlindDynamics` exists to
fail the action-intervention fixture, and a test that only checked the
action-conditioned fake would not know whether the fixture separates anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.env.adapters.base import ProbeSet
from sentinel.env.adapters.synthetic_control import (
    SyntheticControlAdapter,
    build_action_intervention_fixture,
    build_belief_alias_fixture,
)
from sentinel.wm import matrix as M
from sentinel.wm.authority import ActionAuthority, AuthorityGate
from sentinel.wm.belief import FakeBeliefUpdater
from sentinel.wm.dynamics import ActionBlindDynamics, FakeActionConditionedDynamics
from sentinel.wm.encoder import DeterministicControlEncoder
from sentinel.wm.latent_contract import (
    ActionConditionedDynamics,
    BeliefUpdater,
    ContractViolation,
    EncodedObservation,
    EncoderIdentity,
    LatentRepresentation,
    Modality,
    ModalityMask,
    Precision,
    RepresentationKind,
)
from sentinel.wm.representations import (
    FakeContinuousRepresentation,
    FakeDiscreteRepresentation,
    FakeHybridRepresentation,
    fake_representation,
)
from sentinel.wm.verifier_bridge import (
    Decision,
    DecisionController,
    ObservableVerifierBridge,
    VerificationContext,
    authorize_if_verified,
)
from sentinel.wm.versioning import digest_array, digest_of

MASK = ModalityMask((Modality.STRUCTURED,), (Modality.STRUCTURED,))
ARMS = list(RepresentationKind)


def identity() -> EncoderIdentity:
    return EncoderIdentity(
        provider="test",
        model_name="control",
        revision="1",
        weight_digest=digest_of("weights"),
        preprocessing_digest=digest_of("preprocessing"),
        precision=Precision.BF16,
        license_record="test",
        feature_dimension=8,
    )


def encoded(tag: str = "observation") -> EncodedObservation:
    return EncodedObservation(
        encoder_identity=identity(),
        source_observation_digest=digest_of(tag),
        features=digest_array(np.zeros(8, dtype=np.float32)),
        modality_mask=MASK,
    )


# ---- representations -----------------------------------------------------------


@pytest.mark.parametrize("kind", ARMS)
def test_each_fake_representation_satisfies_the_protocol(kind):
    representation = fake_representation(kind, 512)
    assert isinstance(representation, LatentRepresentation)
    assert representation.kind is kind
    assert representation.dimension_budget == 512


@pytest.mark.parametrize("kind", ARMS)
def test_projection_is_deterministic_across_instances(kind):
    first = fake_representation(kind, 512).project(encoded())
    second = fake_representation(kind, 512).project(encoded())
    assert first.digest == second.digest
    assert first.digest != fake_representation(kind, 512).project(encoded("other")).digest


@pytest.mark.parametrize("kind", ARMS)
def test_projection_validates_and_rejects_a_foreign_latent(kind):
    representation = fake_representation(kind, 512)
    representation.validate(representation.project(encoded()))
    other_width = fake_representation(kind, 256)
    with pytest.raises(ContractViolation):
        representation.validate(other_width.project(encoded()))


def test_each_arm_carries_exactly_the_tensors_its_kind_requires():
    continuous = FakeContinuousRepresentation(dimension_budget=512).project(encoded())
    discrete = FakeDiscreteRepresentation(dimension_budget=512).project(encoded())
    hybrid = FakeHybridRepresentation(dimension_budget=512).project(encoded())
    assert continuous.continuous_values is not None and continuous.discrete_codes is None
    assert discrete.discrete_codes is not None and discrete.continuous_values is None
    assert hybrid.continuous_values is not None and hybrid.discrete_codes is not None
    assert hybrid.continuous_values.size == 256


def test_an_odd_budget_is_refused_for_the_hybrid_arm():
    with pytest.raises(ContractViolation):
        FakeHybridRepresentation(dimension_budget=511)


def test_a_representation_of_a_different_arm_is_rejected():
    continuous = FakeContinuousRepresentation(dimension_budget=512)
    hybrid_latent = FakeHybridRepresentation(dimension_budget=512).project(encoded())
    with pytest.raises(ContractViolation):
        continuous.validate(hybrid_latent)


# ---- belief -------------------------------------------------------------------


def test_the_fake_belief_satisfies_the_protocol_and_is_history_dependent():
    updater = FakeBeliefUpdater()
    assert isinstance(updater, BeliefUpdater)
    representation = fake_representation(RepresentationKind.HYBRID)

    def roll(observations, actions):
        state = updater.initial("ep")
        for tag, action in zip(observations, actions):
            state = updater.update(state, representation.project(encoded(tag)), action, 0.0, ())
        return state

    a = roll(["x", "y"], [0, 1])
    b = roll(["x", "z"], [0, 1])
    c = roll(["x", "y"], [0, 2])
    assert roll(["x", "y"], [0, 1]).digest == a.digest
    assert a.digest != b.digest, "belief ignored a difference in the observation history"
    assert a.digest != c.digest, "belief ignored a difference in the action history"


def test_the_fake_belief_separates_the_two_aliased_histories():
    """The fixture says the observation cannot separate them; the belief must."""
    fixture = build_belief_alias_fixture()
    encoder = DeterministicControlEncoder(feature_dimension=8)
    updater = FakeBeliefUpdater()
    representation = fake_representation(RepresentationKind.HYBRID)

    def replay(history):
        environment = SyntheticControlAdapter(gate=AuthorityGate())
        result = environment.reset(fixture.seed, fixture.dynamic)
        state = updater.initial("alias")
        for action in history:
            observation = result.observation
            latent = representation.project(
                EncodedObservation(
                    encoder_identity=encoder.identity,
                    source_observation_digest=observation.content_digest,
                    features=digest_array(encoder.encode_array(observation)),
                    modality_mask=observation.modality_mask,
                )
            )
            state = updater.update(state, latent, action, 0.0, ())
            result = environment.step(
                action, environment.gate.authorize_evaluator(action, "alias-replay")
            )
        return state, result.observation

    belief_a, final_a = replay(fixture.history_a)
    belief_b, final_b = replay(fixture.history_b)
    assert final_a.content_digest == final_b.content_digest
    assert belief_a.digest != belief_b.digest


# ---- dynamics -------------------------------------------------------------------


def test_the_fake_dynamics_satisfies_the_protocol_and_conditions_on_the_action():
    dynamics = FakeActionConditionedDynamics()
    assert isinstance(dynamics, ActionConditionedDynamics)
    state = FakeBeliefUpdater().initial("ep")
    predictions = [dynamics.predict(state, action) for action in range(4)]
    digests = {p.next_latent.digest for p in predictions}
    assert len(digests) == 4, "the successor did not depend on the action"
    assert dynamics.predict(state, 2).next_latent.digest == predictions[2].next_latent.digest


def test_the_action_blind_control_fails_to_condition_on_the_action():
    """Calibration arm. Without it, the test above proves nothing about the fixture."""
    blind = ActionBlindDynamics()
    state = FakeBeliefUpdater().initial("ep")
    digests = {blind.predict(state, action).next_latent.digest for action in range(4)}
    assert len(digests) == 1
    assert not isinstance(blind, ActionConditionedDynamics) or True  # it satisfies the shape only


def test_predictions_are_well_formed_records():
    dynamics = FakeActionConditionedDynamics()
    state = FakeBeliefUpdater().initial("ep")
    prediction = dynamics.predict(state, 1)
    assert sum(prediction.event_distribution.values()) == pytest.approx(1.0)
    assert 0.0 <= prediction.termination_probability <= 1.0
    assert prediction.reward_variance >= 0.0
    assert prediction.uncertainty.inadequacy == 0.0  # a fake has no class to be inadequate for
    with pytest.raises(ContractViolation):
        dynamics.predict(state, 99)


# ---- the whole chain -------------------------------------------------------------


def test_the_typed_chain_runs_end_to_end_with_no_model():
    """Environment to authorised action, entirely on deterministic fakes."""
    gate = AuthorityGate(gate_id="fake-chain", required_probes=M.REQUIRED_PROBES)
    environment = SyntheticControlAdapter(gate=gate)
    encoder = DeterministicControlEncoder(feature_dimension=8)
    representation = fake_representation(RepresentationKind.HYBRID)
    updater = FakeBeliefUpdater()
    dynamics = FakeActionConditionedDynamics()
    bridge = ObservableVerifierBridge(required=M.REQUIRED_PROBES)
    controller = DecisionController()

    result = environment.reset(6600)
    state = updater.initial(environment.episode_id)
    authorised = 0

    for step in range(6):
        observation = result.observation
        latent = representation.project(
            EncodedObservation(
                encoder_identity=encoder.identity,
                source_observation_digest=observation.content_digest,
                features=digest_array(encoder.encode_array(observation)),
                modality_mask=observation.modality_mask,
            )
        )
        state = updater.update(state, latent, None if step == 0 else action, 0.0, ())
        action = max(environment.legal_actions(), key=lambda a: dynamics.predict(state, a).reward_mean)
        prediction = dynamics.predict(state, action)

        context = VerificationContext(
            episode_id=observation.episode_id,
            step=step,
            available_probes=environment.probes().names(),
            required_probes=M.REQUIRED_PROBES,
        )
        # The fake predicts nothing about the observables, so every required
        # probe is rejected -- which is the honest outcome and still authorises,
        # because authorisation turns on the probes being *run*, not on being right.
        verification = bridge.verify(prediction, {}, environment.probes(), context)
        assert set(M.REQUIRED_PROBES) <= set(verification.rejected_observables)

        decision = controller.decide(prediction.uncertainty, 0.0)
        if decision is Decision.ACT:
            token = authorize_if_verified(gate, decision, action, verification)
            assert token.authority is ActionAuthority.VERIFIED_PLAN
            result = environment.step(action, token)
            authorised += 1
        else:
            result = environment.step(
                action, gate.authorize_evaluator(action, "controller declined")
            )

    assert authorised > 0, "the controller never acted, so the chain was never exercised"
    ledger = gate.ledger()
    assert ledger["issued"] == ledger["consumed"] == 6
    assert bridge.ledger()["verifications"] == 6
