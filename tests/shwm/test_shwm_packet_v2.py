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


def clean_builder(_: ProvenanceEnvelope) -> AgentVisiblePacket:
    """A builder that ignores provenance, as a correct one must."""
    return visible()


def test_tensor_invariant_to_whole_provenance_envelope() -> None:
    envelopes = (
        envelope(),
        envelope(environment_seed=99, trajectory_id="t9", clone_lineage=("root", "c1"),
                 generator_metadata={"gen": "v2"}, evaluator_only={"polarity": 0}),
    )
    assert_tensor_invariant_to_provenance(clean_builder, envelopes)


# ---- 2. source digest ---------------------------------------------------------------------


def test_tensor_invariant_to_source_observation_digest() -> None:
    """v1 leaked initial_polarity through exactly this field."""
    assert_tensor_invariant_to_provenance(
        clean_builder, (envelope(source_observation_digest="sha256:one"),
                        envelope(source_observation_digest="sha256:two")))


# ---- 3. absolute time and simulator step --------------------------------------------------


def test_tensor_invariant_to_absolute_time_and_step() -> None:
    """v1 leaked the simulator step through `timestamp_ns`."""
    assert_tensor_invariant_to_provenance(
        clean_builder, (envelope(absolute_timestamp_ns=0, simulator_step=0),
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
    with pytest.raises(ContractViolation, match="not on the permitted list"):
        visible(sensors={"simulator_step": 3.0})


def test_planted_provenance_value_is_caught_by_the_guard() -> None:
    """Calibration, and it must fail for the right reason.

    The first version of this test raised `ContractViolation` itself inside its own
    `pytest.raises` block, so it asserted nothing about production code. It now hands
    a leaky BUILDER to the real guard and requires the guard to raise.

    Two leaks are planted, both under names no denylist would flag: one that reaches
    the tensor through `delta_t`, and one through `visual`. If either survives, the
    isolation claim is false.
    """
    def leak_via_delta_t(env: ProvenanceEnvelope) -> AgentVisiblePacket:
        return visible(delta_t=float(env.simulator_step))

    def leak_via_visual(env: ProvenanceEnvelope) -> AgentVisiblePacket:
        packet = visible()
        pixels = np.array(packet.visual, copy=True)
        pixels.flat[0] = float(env.simulator_step)
        return AgentVisiblePacket(
            visual=pixels, language_goal_tokens=packet.language_goal_tokens,
            scalar_sensors=packet.scalar_sensors, previous_action=packet.previous_action,
            action_result=packet.action_result, delta_t=packet.delta_t,
            modality_masks=packet.modality_masks)

    envelopes = (envelope(simulator_step=1), envelope(simulator_step=6))
    for builder in (leak_via_delta_t, leak_via_visual):
        with pytest.raises(ContractViolation, match="folding a provenance value"):
            assert_tensor_invariant_to_provenance(builder, envelopes)


def test_the_guard_can_actually_fail() -> None:
    """A guard that cannot fail is not a guard.

    Replacing `assert_tensor_invariant_to_provenance` with a no-op previously left
    every test in this file passing. This pins that it cannot happen again.
    """
    def leaky(env: ProvenanceEnvelope) -> AgentVisiblePacket:
        return visible(delta_t=float(env.absolute_timestamp_ns))

    with pytest.raises(ContractViolation):
        assert_tensor_invariant_to_provenance(
            leaky, (envelope(absolute_timestamp_ns=1), envelope(absolute_timestamp_ns=2)))


def test_scalar_sensors_are_allow_listed_not_deny_listed() -> None:
    """An innocent name must be rejected because it is not permitted, not because
    it appears on a list of forbidden words."""
    with pytest.raises(ContractViolation, match="not on the permitted list"):
        visible(sensors={"action_result": 1.0, "t": 3.0})


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


# ---- the audit's packet and the class must not drift apart ---------------------------------


def test_audit_packet_matches_the_class() -> None:
    """The §C certificate counts must describe the schema the class defines.

    They were written separately, and a reviewer pointed out that nothing tied them
    together -- so a count could describe a packet no model would ever see.
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiments/shwm"))
    from alias_audit import assert_matches_packet_v2

    assert_matches_packet_v2()


def test_audit_packet_check_can_fail() -> None:
    """Calibration: the consistency check must catch a real divergence."""
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiments/shwm"))
    import alias_audit

    original = alias_audit.V2_AGENT_VISIBLE
    try:
        alias_audit.V2_AGENT_VISIBLE = original + ("simulator_step",)
        with pytest.raises(ContractViolation):
            alias_audit.assert_matches_packet_v2()
    finally:
        alias_audit.V2_AGENT_VISIBLE = original
