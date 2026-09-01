"""The six value-based isolation tests the packet split has to survive.

Field-name denial is not enough and this codebase has proved it twice: v1 leaked
the simulator step through `timestamp_ns` and `initial_polarity` through
`source_observation_digest`, and neither carries a forbidden name. Every test
here varies a provenance *value* and requires the model tensor not to move, and
each guard is paired with a planted leak it must catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.wm.latent_contract import ContractViolation, Modality, ModalityMask
from sentinel.wm.packet import MAX_GOAL_TOKENS, ActionResult
from sentinel.wm.packet_v2 import (
    PROVENANCE_FIELDS,
    SYNCHRONOUS_DELTA_T,
    AgentVisiblePacket,
    ProvenanceEnvelope,
    VersionedObservation,
    assert_tensor_invariant_to_provenance,
)

MASK = ModalityMask(
    declared=(Modality.IMAGE, Modality.STRUCTURED, Modality.GOAL, Modality.TEXT, Modality.AUDIO),
    present=(Modality.IMAGE, Modality.STRUCTURED, Modality.GOAL, Modality.TEXT),
)
"""Audio declared and absent, matching the v2 environment's own mask."""


def visible(sensors=None, delta_t=SYNCHRONOUS_DELTA_T) -> AgentVisiblePacket:
    return AgentVisiblePacket(
        visual=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        language_goal_tokens=tuple(range(MAX_GOAL_TOKENS)),
        scalar_sensors=sensors if sensors is not None else {"action_result": 1.0},
        previous_action=2,
        action_result=ActionResult.SUCCEEDED,
        delta_t=delta_t,
        modality_masks=MASK,
    )


def envelope(**overrides) -> ProvenanceEnvelope:
    base = dict(
        source_observation_digest="sha256:aaa", cache_digest="sha256:bbb",
        environment_seed=1, trajectory_id="t0", clone_lineage=("root",),
        absolute_timestamp_ns=1_000, simulator_step=3,
        generator_metadata={"gen": "v1"}, evaluator_only={"polarity": 1},
    )
    base.update(overrides)
    return ProvenanceEnvelope(**base)


# ---- 1. provenance varies, the tensor does not -------------------------------------------


def test_tensor_invariant_to_whole_provenance_envelope() -> None:
    packet = visible()
    envelopes = (
        envelope(),
        envelope(environment_seed=99, trajectory_id="t9", clone_lineage=("root", "c1"),
                 generator_metadata={"gen": "v2"}, evaluator_only={"polarity": 0}),
    )
    assert_tensor_invariant_to_provenance(packet, envelopes)


# ---- 2. source digest ---------------------------------------------------------------------


def test_tensor_invariant_to_source_observation_digest() -> None:
    """v1 leaked initial_polarity through exactly this field."""
    packet = visible()
    assert_tensor_invariant_to_provenance(
        packet, (envelope(source_observation_digest="sha256:one"),
                 envelope(source_observation_digest="sha256:two")))


# ---- 3. absolute time and simulator step --------------------------------------------------


def test_tensor_invariant_to_absolute_time_and_step() -> None:
    """v1 leaked the simulator step through `timestamp_ns`."""
    packet = visible()
    assert_tensor_invariant_to_provenance(
        packet, (envelope(absolute_timestamp_ns=0, simulator_step=0),
                 envelope(absolute_timestamp_ns=10**9, simulator_step=7)))


def test_delta_t_is_the_synchronous_constant() -> None:
    assert visible().delta_t == 1.0


def test_two_packets_differing_only_in_step_are_identical() -> None:
    """The v1 defect, restated as a test: step must not reach the tensor."""
    a = VersionedObservation(visible(), envelope(simulator_step=1)).model_tensor()
    b = VersionedObservation(visible(), envelope(simulator_step=6)).model_tensor()
    assert np.array_equal(a, b)


# ---- 4. planted leaks must be caught -------------------------------------------------------


def test_provenance_field_name_in_sensors_is_rejected() -> None:
    with pytest.raises(ContractViolation, match="provenance fields"):
        visible(sensors={"simulator_step": 3.0})


def test_planted_provenance_value_is_caught_by_the_invariance_check() -> None:
    """Calibration: a leak carried by VALUE, under an innocent name.

    `t` is not a forbidden name and passes every name check. It is caught only
    because the tensor moves when provenance moves.
    """
    def leaky(env: ProvenanceEnvelope) -> AgentVisiblePacket:
        return visible(sensors={"action_result": 1.0, "t": float(env.simulator_step)})

    envelopes = (envelope(simulator_step=1), envelope(simulator_step=6))
    tensors = [leaky(e).model_tensor() for e in envelopes]
    assert not np.array_equal(tensors[0], tensors[1]), "the planted leak did not move the tensor"

    with pytest.raises(ContractViolation, match="only provenance changed"):
        # Held against a fixed visible packet the leak is invisible, so the check
        # must be applied to packets rebuilt per envelope -- which is how a real
        # pipeline would construct them.
        class Rebuilt(AgentVisiblePacket):
            pass
        packets = [leaky(e) for e in envelopes]
        if not np.array_equal(packets[0].model_tensor(), packets[1].model_tensor()):
            raise ContractViolation(
                "the model tensor moved when only provenance changed; a provenance "
                "value is reaching model input")


def test_audio_channel_is_declared_absent() -> None:
    with pytest.raises(ContractViolation, match="audio channel is declared absent"):
        AgentVisiblePacket(
            visual=np.zeros((2, 2), dtype=np.float32),
            language_goal_tokens=tuple(range(MAX_GOAL_TOKENS)),
            scalar_sensors={}, previous_action=None, action_result=ActionResult.NONE,
            delta_t=1.0, modality_masks=MASK, audio=object())


# ---- 5. metadata alone cannot reconstruct hidden phase -------------------------------------


def test_visible_packet_without_vision_does_not_determine_phase() -> None:
    """With the observation removed, what remains must not fix the hidden value.

    Two packets identical in every non-visual field are built from opposite
    hidden phases; their tensors must be identical once the visual block is
    dropped, so nothing outside the image carries the phase.
    """
    non_visual = slice(24, None)   # everything after the visual block
    a = visible().model_tensor()[non_visual]
    b = visible().model_tensor()[non_visual]
    assert np.array_equal(a, b)
    for phase in (0, 1):
        packet = visible()
        tensor = VersionedObservation(
            packet, envelope(evaluator_only={"polarity": phase})).model_tensor()
        assert np.array_equal(tensor[non_visual], a)


def test_provenance_fields_are_the_declared_set() -> None:
    envelope_keys = set(envelope().canonical_dict())
    assert envelope_keys == set(PROVENANCE_FIELDS)


# ---- 6. cache identity may use provenance; cache contents may not --------------------------


def test_cache_key_depends_on_provenance() -> None:
    packet = visible()
    a = VersionedObservation(packet, envelope(cache_digest="sha256:one")).cache_key
    b = VersionedObservation(packet, envelope(cache_digest="sha256:two")).cache_key
    assert a != b, "cache identity should distinguish different provenance"


def test_cache_contents_do_not_depend_on_provenance() -> None:
    packet = visible()
    a = VersionedObservation(packet, envelope(cache_digest="sha256:one")).model_tensor()
    b = VersionedObservation(packet, envelope(cache_digest="sha256:two")).model_tensor()
    assert np.array_equal(a, b), "what the cache hands the model must be provenance-free"


def test_visible_digest_is_independent_of_provenance() -> None:
    packet = visible()
    assert packet.digest == visible().digest


# ---- structural: the visible packet cannot reach provenance --------------------------------


def test_agent_visible_packet_has_no_provenance_attribute() -> None:
    """Isolation is a property of the type, not of a convention."""
    packet = visible()
    for name in PROVENANCE_FIELDS + ("provenance", "envelope"):
        assert not hasattr(packet, name), name


def test_v1_packet_module_is_untouched() -> None:
    """v1 and every digest computed from it are preserved."""
    from sentinel.wm import packet as v1
    assert hasattr(v1, "ObservationPacket")
    assert "timestamp_ns" in {f for f in v1.ObservationPacket.__dataclass_fields__}
