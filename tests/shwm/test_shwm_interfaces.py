"""Scale-1A-0 E and F: the packet, and eight interfaces at one matched shape.

The matched shape is the whole comparison. An arm with more slots or a wider slot
has more capacity, and more capacity cannot be told apart from a better
representation, so the shape is enforced at construction rather than checked in
a report.

The other thing enforced here is naming: a fixed random projection is a frozen
matrix drawn once and must not be counted or described as learned.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.env.adapters.procedural_visual_v2 import (
    GOAL_PHRASES,
    ProceduralVisualV2Adapter,
)
from sentinel.wm.authority import AuthorityGate
from sentinel.wm.interfaces import (
    BackboneMeanPool,
    BackboneSpatialSlots,
    FixedRandomSpatialProjection,
    InterfaceContext,
    OracleStructuredState,
    RawLowResSpatial,
    SmallLearnedSpatialEncoder,
    build_interfaces,
    interface_report,
)
from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.packet import (
    MAX_GOAL_TOKENS,
    SLOT_COUNT,
    SLOT_WIDTH,
    ActionResult,
    build_packet,
    build_vocabulary,
    tokenise_goal,
)

VOCABULARY = build_vocabulary(GOAL_PHRASES.values())


def context(seed: int = 9000, tokens: dict | None = None) -> InterfaceContext:
    gate = AuthorityGate()
    adapter = ProceduralVisualV2Adapter(gate=gate)
    result = adapter.reset(seed)
    truth = {
        k: v
        for k, v in adapter.snapshot().reveal("evaluator").items()
        if isinstance(v, (int, float, bool))
    }
    return InterfaceContext(
        observation=result.observation,
        frame=adapter.frame(),
        visual_tokens=tokens or {},
        truth=truth,
    )


# ---- packet -------------------------------------------------------------------------


def test_the_packet_enforces_the_matched_slot_shape():
    good = np.zeros((SLOT_COUNT, SLOT_WIDTH), dtype=np.float32)
    build_packet(
        context().observation, good, vocabulary=VOCABULARY, interface_name="t",
        previous_action=None, action_result=ActionResult.NONE,
    )
    for bad_shape in ((SLOT_COUNT + 1, SLOT_WIDTH), (SLOT_COUNT, SLOT_WIDTH * 2)):
        with pytest.raises(ContractViolation, match="expected"):
            build_packet(
                context().observation, np.zeros(bad_shape, dtype=np.float32),
                vocabulary=VOCABULARY, interface_name="t",
                previous_action=None, action_result=ActionResult.NONE,
            )


def test_audio_is_a_declared_and_absent_channel():
    packet = build_packet(
        context().observation, np.zeros((SLOT_COUNT, SLOT_WIDTH), dtype=np.float32),
        vocabulary=VOCABULARY, interface_name="t",
        previous_action=None, action_result=ActionResult.NONE,
    )
    declared = {m.value for m in packet.modality_masks.declared}
    present = {m.value for m in packet.modality_masks.present}
    assert "audio" in declared and "audio" not in present
    assert packet.audio_slots is None


def test_the_goal_is_tokenised_to_a_fixed_length_with_a_shared_vocabulary():
    a = tokenise_goal(GOAL_PHRASES["alpha"], VOCABULARY)
    b = tokenise_goal(GOAL_PHRASES["beta"], VOCABULARY)
    assert len(a) == len(b) == MAX_GOAL_TOKENS
    assert a != b, "the two instructions tokenise identically"
    assert tokenise_goal("completely unseen words here", VOCABULARY).count(VOCABULARY["<unk>"]) >= 3


def test_the_packet_refuses_evaluator_fields_in_its_sensors():
    with pytest.raises(ContractViolation):
        build_packet(
            context().observation, np.zeros((SLOT_COUNT, SLOT_WIDTH), dtype=np.float32),
            vocabulary=VOCABULARY, interface_name="t",
            previous_action=None, action_result=ActionResult.NONE,
            scalar_sensors={"hidden_state": 1.0},
        )


def test_an_unknown_action_result_is_refused():
    with pytest.raises(ContractViolation, match="action_result"):
        build_packet(
            context().observation, np.zeros((SLOT_COUNT, SLOT_WIDTH), dtype=np.float32),
            vocabulary=VOCABULARY, interface_name="t",
            previous_action=None, action_result="maybe",
        )


# ---- interfaces ----------------------------------------------------------------------


def fake_tokens(count: int, width: int = 2560) -> np.ndarray:
    generator = np.random.default_rng(count)
    return generator.normal(size=(count, width)).astype(np.float32)


def test_every_interface_emits_the_same_shape():
    tokens = {"qwen3_vl_4b": fake_tokens(64), "gemma3_4b": fake_tokens(256)}
    ctx = context(tokens=tokens)
    for interface in build_interfaces():
        slots = interface.slots(ctx)
        assert slots.shape == (SLOT_COUNT, SLOT_WIDTH), interface.name
        assert slots.dtype == np.float32


def test_only_the_cnn_is_learned_and_only_it_has_parameters():
    interfaces = build_interfaces()
    learned = [i.name for i in interfaces if i.learned]
    assert learned == ["small_learned_cnn"]
    for interface in interfaces:
        if interface.learned:
            assert interface.trainable_parameters > 0
        else:
            assert interface.trainable_parameters == 0, (
                f"{interface.name} reports trainable parameters but is not learned"
            )


def test_a_fixed_random_projection_is_not_described_as_learned():
    projection = FixedRandomSpatialProjection()
    assert projection.learned is False
    assert projection.trainable_parameters == 0
    report = interface_report(build_interfaces())
    assert "not a learned representation" in report["note"]
    assert report["total_trainable_adapter_parameters"] == sum(
        i.trainable_parameters for i in build_interfaces() if not i.evaluator_only
    )


def test_mean_pooling_makes_every_slot_identical():
    """The ablation reads correctly only if the pooled arm really is the spatial
    arm with position deleted."""
    tokens = {"qwen3_vl_4b": fake_tokens(64)}
    ctx = context(tokens=tokens)
    pooled = BackboneMeanPool(encoder_id="qwen3_vl_4b").slots(ctx)
    assert np.allclose(pooled, pooled[0][None, :]), "pooled slots are not all equal"
    spatial = BackboneSpatialSlots(encoder_id="qwen3_vl_4b").slots(ctx)
    assert not np.allclose(spatial, spatial[0][None, :]), "spatial slots collapsed"


def test_spatial_slots_preserve_the_coordinate_layout():
    """A change confined to one corner of the token grid must move one slot."""
    base = fake_tokens(64)
    moved = base.copy()
    moved[0] += 50.0  # top-left token of the 8x8 grid
    interface = BackboneSpatialSlots(encoder_id="qwen3_vl_4b")
    a = interface.slots(context(tokens={"qwen3_vl_4b": base}))
    b = interface.slots(context(tokens={"qwen3_vl_4b": moved}))
    differing = {i for i in range(SLOT_COUNT) if not np.allclose(a[i], b[i])}
    assert differing == {0}, f"a corner change moved slots {sorted(differing)}"


def test_a_non_square_token_count_is_refused():
    interface = BackboneSpatialSlots(encoder_id="qwen3_vl_4b")
    with pytest.raises(ContractViolation, match="square grid"):
        interface.slots(context(tokens={"qwen3_vl_4b": fake_tokens(63)}))


def test_the_oracle_is_marked_evaluator_only_and_needs_truth():
    oracle = OracleStructuredState()
    assert oracle.evaluator_only is True
    empty = context()
    empty.truth = {}
    with pytest.raises(ContractViolation, match="evaluator-only truth"):
        oracle.slots(empty)


def test_the_oracle_is_excluded_from_the_admissible_adapter_budget():
    interfaces = build_interfaces()
    admissible = [i for i in interfaces if not i.evaluator_only]
    assert len(admissible) == 7
    assert all(not i.evaluator_only for i in admissible)


def test_raw_and_random_interfaces_differ_from_each_other():
    ctx = context()
    raw = RawLowResSpatial().slots(ctx)
    projected = FixedRandomSpatialProjection().slots(ctx)
    assert not np.allclose(raw, projected)
    # The raw interface is genuinely raw: the tail of each slot is untouched.
    assert np.allclose(raw[:, 200:], 0.0)


def test_the_learned_encoder_is_deterministic_at_a_fixed_seed():
    ctx = context()
    a = SmallLearnedSpatialEncoder(seed=6600).slots(ctx)
    b = SmallLearnedSpatialEncoder(seed=6600).slots(ctx)
    assert np.allclose(a, b)
    assert SmallLearnedSpatialEncoder(seed=6600).trainable_parameters > 0
