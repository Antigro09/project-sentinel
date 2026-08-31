"""Scale-0 gate: the typed boundary fails closed.

Every record in the SHWM contract is a place where a later capability claim can
be quietly invalidated -- by a malformed mask, a payload with no provenance, a
representation arm carrying the wrong tensors, or an evaluator-only field
riding along in a training record. These tests are the calibration arm for the
schema: each one supplies a known-bad input that construction must reject.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.wm.events import (
    EVENT_ORDER,
    SCHEMA_DIGEST,
    EventKind,
    StructuredEvent,
    event_distribution_from_logits,
)
from sentinel.wm.latent_contract import (
    ContractViolation,
    Counterexample,
    EncodedObservation,
    EncoderIdentity,
    LatentObservation,
    Modality,
    ModalityMask,
    ObservationEnvelope,
    Precision,
    RepresentationKind,
    Taint,
    TransitionPrediction,
    UncertaintyTriple,
    VerificationResult,
)
from sentinel.wm.uncertainty import build_calibration_table, ensemble_disagreement
from sentinel.wm.versioning import (
    ArrayDigest,
    CanonicalisationError,
    canonical_json,
    digest_array,
    digest_of,
)

MASK = ModalityMask(declared=(Modality.IMAGE, Modality.STRUCTURED), present=(Modality.STRUCTURED,))


def identity(**overrides) -> EncoderIdentity:
    base = dict(
        provider="test",
        model_name="model",
        revision="rev-1",
        weight_digest=digest_of("weights"),
        preprocessing_digest=digest_of("preprocessing"),
        precision=Precision.BF16,
        license_record="test-licence",
        feature_dimension=8,
    )
    base.update(overrides)
    return EncoderIdentity(**base)


def envelope(**overrides) -> ObservationEnvelope:
    base = dict(
        episode_id="ep-1",
        step=0,
        timestamp_ns=123,
        modality_payloads={},
        structured_observation={"visible": 3},
        modality_mask=MASK,
        available_action_digest=digest_of([0, 1, 2, 3]),
        environment_version=digest_of("env-v1"),
    )
    base.update(overrides)
    return ObservationEnvelope(**base)


# ---- canonical serialisation ------------------------------------------------


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_rejects_non_finite_floats():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CanonicalisationError):
            digest_of({"x": bad})


def test_digest_is_stable_across_calls_and_sensitive_to_content():
    assert digest_of({"a": 1}) == digest_of({"a": 1})
    assert digest_of({"a": 1}) != digest_of({"a": 2})


def test_array_digest_distinguishes_dtype_and_shape():
    values = np.arange(6, dtype=np.float32)
    same = digest_array(values.copy())
    reshaped = digest_array(values.reshape(2, 3))
    recast = digest_array(values.astype(np.float64))
    assert digest_array(values) == same
    assert reshaped.digest != same.digest
    assert recast.digest != same.digest


def test_array_digest_rejects_a_bad_digest_string():
    with pytest.raises(CanonicalisationError):
        ArrayDigest(dtype="float32", shape=(2,), digest="deadbeef")


# ---- round trips -------------------------------------------------------------


def test_records_round_trip_through_canonical_form():
    latent = LatentObservation(
        episode_id="ep-1",
        step=2,
        encoder_identity=identity(),
        projector_digest=digest_of("projector"),
        representation_kind=RepresentationKind.CONTINUOUS,
        modality_mask=MASK,
        source_observation_digest=digest_of("obs"),
        continuous_values=digest_array(np.zeros(8, dtype=np.float32)),
    )
    text = canonical_json(latent.canonical_dict())
    assert canonical_json(latent.canonical_dict()) == text
    assert latent.digest == digest_of(latent.canonical_dict())


def test_observation_digest_ignores_wall_clock_but_not_content():
    a = envelope(timestamp_ns=1)
    b = envelope(timestamp_ns=999_999)
    c = envelope(structured_observation={"visible": 4})
    assert a.digest == b.digest
    assert a.digest != c.digest


# ---- malformed-state rejection ----------------------------------------------


def test_mask_rejects_present_modality_that_was_not_declared():
    with pytest.raises(ContractViolation):
        ModalityMask(declared=(Modality.IMAGE,), present=(Modality.AUDIO,))


def test_envelope_rejects_payload_for_an_absent_modality():
    with pytest.raises(ContractViolation):
        envelope(modality_payloads={"image": digest_array(np.zeros(2, dtype=np.uint8))})


def test_envelope_rejects_raw_array_payload_without_provenance():
    with pytest.raises(ContractViolation):
        envelope(
            modality_mask=ModalityMask((Modality.IMAGE,), (Modality.IMAGE,)),
            modality_payloads={"image": np.zeros(2)},
        )


def test_envelope_rejects_hidden_simulator_fields():
    for field_name in ("hidden_state", "simulator_state", "target_program", "evaluator_answer"):
        with pytest.raises(ContractViolation):
            envelope(structured_observation={field_name: 1})


def test_encoder_identity_requires_frozen_weights_and_positive_width():
    with pytest.raises(ContractViolation):
        identity(frozen=False)
    with pytest.raises(ContractViolation):
        identity(feature_dimension=0)
    with pytest.raises(CanonicalisationError):
        identity(weight_digest="not-a-digest")


def test_encoded_observation_rejects_width_disagreeing_with_identity():
    with pytest.raises(ContractViolation):
        EncodedObservation(
            encoder_identity=identity(feature_dimension=8),
            source_observation_digest=digest_of("obs"),
            features=digest_array(np.zeros(16, dtype=np.float32)),
            modality_mask=MASK,
        )


@pytest.mark.parametrize(
    "kind,continuous,discrete",
    [
        (RepresentationKind.CONTINUOUS, False, True),
        (RepresentationKind.CONTINUOUS, False, False),
        (RepresentationKind.DISCRETE, True, True),
        (RepresentationKind.DISCRETE, False, False),
        (RepresentationKind.HYBRID, True, False),
        (RepresentationKind.HYBRID, False, True),
    ],
)
def test_latent_rejects_tensors_that_disagree_with_its_arm(kind, continuous, discrete):
    with pytest.raises(ContractViolation):
        LatentObservation(
            episode_id="ep",
            step=0,
            encoder_identity=identity(),
            projector_digest=digest_of("p"),
            representation_kind=kind,
            modality_mask=MASK,
            source_observation_digest=digest_of("o"),
            continuous_values=digest_array(np.zeros(4, dtype=np.float32)) if continuous else None,
            discrete_codes=digest_array(np.zeros(4, dtype=np.int32)) if discrete else None,
        )


def test_discrete_codes_must_be_integral():
    with pytest.raises(ContractViolation):
        LatentObservation(
            episode_id="ep",
            step=0,
            encoder_identity=identity(),
            projector_digest=digest_of("p"),
            representation_kind=RepresentationKind.DISCRETE,
            modality_mask=MASK,
            source_observation_digest=digest_of("o"),
            discrete_codes=digest_array(np.zeros(4, dtype=np.float32)),
        )


def test_uncertainty_components_stay_separate_and_non_negative():
    triple = UncertaintyTriple(aleatoric=0.25, epistemic=0.5, inadequacy=0.125)
    assert triple.canonical_dict() == {"aleatoric": 0.25, "epistemic": 0.5, "inadequacy": 0.125}
    assert triple.scalar((1.0, 2.0, 4.0)) == pytest.approx(0.25 + 1.0 + 0.5)
    with pytest.raises(ContractViolation):
        UncertaintyTriple(aleatoric=-0.1, epistemic=0.0, inadequacy=0.0)


def test_prediction_rejects_an_unnormalised_event_distribution():
    with pytest.raises(ContractViolation):
        TransitionPrediction(
            next_latent=digest_array(np.zeros(4, dtype=np.float32)),
            event_distribution={"ACTION_SUCCEEDED": 0.4, "ACTION_FAILED": 0.4},
            reward_mean=0.0,
            reward_variance=0.0,
            termination_probability=0.0,
            uncertainty=UncertaintyTriple(0.0, 0.0, 0.0),
            rollout_support_scope="dev",
            model_version=digest_of("m"),
            action=0,
        )


def test_prediction_rejects_out_of_range_termination_and_negative_variance():
    common = dict(
        next_latent=digest_array(np.zeros(4, dtype=np.float32)),
        event_distribution={},
        reward_mean=0.0,
        uncertainty=UncertaintyTriple(0.0, 0.0, 0.0),
        rollout_support_scope="dev",
        model_version=digest_of("m"),
        action=0,
    )
    with pytest.raises(ContractViolation):
        TransitionPrediction(termination_probability=1.5, reward_variance=0.0, **common)
    with pytest.raises(ContractViolation):
        TransitionPrediction(termination_probability=0.5, reward_variance=-1.0, **common)


# ---- accuracy and coverage stay independent ---------------------------------


def test_verification_keeps_accuracy_and_coverage_apart():
    result = VerificationResult(
        accepted_observables=("reward",),
        rejected_observables=("termination",),
        unprobed_observables=("goal_progress", "constraint_violation"),
        counterexamples=(
            Counterexample("termination", predicted=True, actual=False, step=3, episode_id="ep"),
        ),
        constraint_violations=(),
        verifier_version=digest_of("verifier"),
        required_probe_names=("reward", "termination"),
    )
    assert result.accuracy == pytest.approx(0.5)
    assert result.coverage == pytest.approx(0.5)


def test_verification_refuses_to_report_when_a_required_probe_was_skipped():
    with pytest.raises(ContractViolation):
        VerificationResult(
            accepted_observables=("reward",),
            rejected_observables=(),
            unprobed_observables=("termination",),
            counterexamples=(),
            constraint_violations=(),
            verifier_version=digest_of("verifier"),
            required_probe_names=("reward", "termination"),
        )


def test_total_abstention_reports_zero_coverage_not_perfect_accuracy():
    result = VerificationResult(
        accepted_observables=(),
        rejected_observables=(),
        unprobed_observables=("reward", "termination"),
        counterexamples=(),
        constraint_violations=(),
        verifier_version=digest_of("verifier"),
    )
    assert result.coverage == 0.0
    assert result.accuracy != result.accuracy  # NaN: undefined, not 1.0


# ---- events ------------------------------------------------------------------


def test_event_order_is_frozen_and_hashed():
    assert len(EVENT_ORDER) == 12
    assert EVENT_ORDER[0] is EventKind.OBJECT_APPEARED
    assert EVENT_ORDER[-1] is EventKind.MISSING_EVENT_REPRESENTATION
    assert SCHEMA_DIGEST == digest_of([k.value for k in EVENT_ORDER])


def test_event_distribution_normalises_over_the_frozen_order():
    distribution = event_distribution_from_logits(np.zeros(len(EVENT_ORDER)))
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert set(distribution) == {k.value for k in EVENT_ORDER}
    with pytest.raises(ContractViolation):
        event_distribution_from_logits(np.zeros(len(EVENT_ORDER) - 1))


def test_an_event_without_a_witnessing_probe_is_rejected():
    with pytest.raises(ContractViolation):
        StructuredEvent(kind=EventKind.GOAL_PROGRESS_CHANGED, witness="")
    # The one exception: the schema itself being inadequate has no witness.
    StructuredEvent(kind=EventKind.MISSING_EVENT_REPRESENTATION, witness="")


# ---- uncertainty diagnostics -------------------------------------------------


def test_calibration_table_counts_every_sample_including_confidence_one():
    table = build_calibration_table([0.0, 0.55, 1.0], [False, True, True], n_bins=10)
    assert table.total == 3
    assert sum(b.count for b in table.bins) == 3


def test_ensemble_disagreement_is_zero_for_identical_members():
    assert ensemble_disagreement([[1.0, 2.0], [1.0, 2.0]]) == 0.0
    assert ensemble_disagreement([[0.0, 0.0], [1.0, 1.0]]) > 0.0
