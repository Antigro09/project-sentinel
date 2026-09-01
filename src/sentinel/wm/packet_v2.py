"""B. Packet v2: what the model sees, split from what the run records.

`packet.py` v1 is preserved unchanged, and so is every digest computed from it.
This module is a new version, and it exists because two hidden values were
reaching the agent-visible surface through v1's schema.

`ObservationPacket.timestamp_ns` was set to the simulator step, so the packet
carried the step. And `canonical_dict` included `source_observation_digest`,
which is `ObservationEnvelope.content_digest`, which hashes
`environment_version`, a digest of `LevelV2.digest`, which contains
`initial_polarity`. Two levels identical in layout and appearance but opposite in
initial polarity therefore produced different packet digests. Neither field has a
forbidden name, so v1's name-based rejection could not see either.

The fix is structural rather than a longer denylist. `AgentVisiblePacket` holds
what a model may read and has no reference to provenance at all;
`ProvenanceEnvelope` holds identity, seeds, lineage, timing and evaluator state.
A model tensor is built from the visible half by a function that cannot reach the
other, so isolation is a property of the type graph and not of a convention that
a later edit could quietly break.

Timing is the one field that needed a judgement rather than a move. The v2
environment is synchronous -- one step is one transition -- so an absolute
timestamp carries the step and nothing else. `delta_t` is therefore the constant
1.0, present because the schema wants a time value, carrying no information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sentinel.wm.latent_contract import (
    ArrayDigest,
    ContractViolation,
    ModalityMask,
    reject_hidden_fields,
)
from sentinel.wm.packet import MAX_GOAL_TOKENS, ActionResult, tokenise_goal
from sentinel.wm.versioning import digest_of

PACKET_VERSION = "agent-visible-v2"

SYNCHRONOUS_DELTA_T = 1.0
"""One step is one transition, so relative timing is a constant.

Keeping the field and fixing its value is deliberate: a schema that omits timing
entirely cannot express an asynchronous environment later, while a schema that
carries an absolute timestamp in a synchronous one is carrying the step number.
"""

PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_observation_digest",
    "cache_digest",
    "environment_seed",
    "trajectory_id",
    "clone_lineage",
    "absolute_timestamp_ns",
    "simulator_step",
    "generator_metadata",
    "evaluator_only",
)


@dataclass(frozen=True, slots=True)
class AgentVisiblePacket:
    """Everything the model may read, and nothing else.

    It holds no provenance field and no reference to a `ProvenanceEnvelope`, so
    `model_tensor` cannot reach one however it is called.
    """

    visual: np.ndarray
    language_goal_tokens: tuple[int, ...]
    scalar_sensors: Mapping[str, float]
    previous_action: int | None
    action_result: str
    delta_t: float
    modality_masks: ModalityMask
    audio: None = None

    def __post_init__(self) -> None:
        if len(self.language_goal_tokens) != MAX_GOAL_TOKENS:
            raise ContractViolation(
                f"goal tokens are length {len(self.language_goal_tokens)}, expected "
                f"{MAX_GOAL_TOKENS} after padding"
            )
        if self.action_result not in (
            ActionResult.NONE, ActionResult.SUCCEEDED, ActionResult.FAILED
        ):
            raise ContractViolation(f"unknown action_result {self.action_result!r}")
        reject_hidden_fields(self.scalar_sensors, "AgentVisiblePacket.scalar_sensors")
        leaked = sorted(set(self.scalar_sensors) & set(PROVENANCE_FIELDS))
        if leaked:
            raise ContractViolation(
                f"provenance fields {leaked} appear in scalar_sensors; provenance may "
                f"key a cache but may never be a model input"
            )
        if self.audio is not None:
            raise ContractViolation("the audio channel is declared absent in v2")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "version": PACKET_VERSION,
            "visual": digest_of(np.ascontiguousarray(self.visual).tobytes().hex()),
            "language_goal_tokens": list(self.language_goal_tokens),
            "scalar_sensors": {k: float(v) for k, v in sorted(self.scalar_sensors.items())},
            "previous_action": self.previous_action,
            "action_result": self.action_result,
            "delta_t": float(self.delta_t),
            "modality_masks": self.modality_masks.canonical_dict(),
            "audio": None,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())

    def model_tensor(self) -> np.ndarray:
        """The numbers a model actually consumes.

        Built only from this object. There is no provenance argument and no
        attribute on `self` that could supply one, which is what makes the
        invariance tests in `tests/shwm/test_shwm_packet_v2.py` structural rather
        than a spot check.
        """
        action = np.zeros(5, dtype=np.float32)
        action[0 if self.previous_action is None else self.previous_action + 1] = 1.0
        result = np.zeros(3, dtype=np.float32)
        result[(ActionResult.NONE, ActionResult.SUCCEEDED,
                ActionResult.FAILED).index(self.action_result)] = 1.0
        sensors = np.array(
            [float(v) for _, v in sorted(self.scalar_sensors.items())], dtype=np.float32
        )
        goal = np.asarray(self.language_goal_tokens, dtype=np.float32)
        return np.concatenate([
            np.asarray(self.visual, dtype=np.float32).reshape(-1),
            goal, sensors, action, result,
            np.array([float(self.delta_t)], dtype=np.float32),
        ])


@dataclass(frozen=True, slots=True)
class ProvenanceEnvelope:
    """Run bookkeeping. May key a cache; may never reach a model.

    `evaluator_only` is deliberately typed as a plain mapping rather than being
    flattened into named fields, so adding an evaluator quantity later cannot
    accidentally widen the visible packet.
    """

    source_observation_digest: str
    cache_digest: str
    environment_seed: int
    trajectory_id: str
    clone_lineage: tuple[str, ...]
    absolute_timestamp_ns: int
    simulator_step: int
    generator_metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluator_only: Mapping[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "source_observation_digest": self.source_observation_digest,
            "cache_digest": self.cache_digest,
            "environment_seed": self.environment_seed,
            "trajectory_id": self.trajectory_id,
            "clone_lineage": list(self.clone_lineage),
            "absolute_timestamp_ns": self.absolute_timestamp_ns,
            "simulator_step": self.simulator_step,
            "generator_metadata": dict(sorted(self.generator_metadata.items())),
            "evaluator_only": dict(sorted(self.evaluator_only.items())),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class VersionedObservation:
    """The pair. `cache_key` may use provenance; `model_tensor` provably cannot."""

    visible: AgentVisiblePacket
    provenance: ProvenanceEnvelope

    @property
    def cache_key(self) -> str:
        """Cache identity may depend on provenance -- two runs of the same content
        are legitimately different cache entries -- but what the cache hands the
        model is `visible` alone."""
        return digest_of({"visible": self.visible.digest,
                          "provenance": self.provenance.digest})

    def model_tensor(self) -> np.ndarray:
        return self.visible.model_tensor()


def assert_tensor_invariant_to_provenance(
    visible: AgentVisiblePacket, envelopes: tuple[ProvenanceEnvelope, ...]
) -> None:
    """Vary provenance, hold the visible packet fixed, require tensor identity.

    A value-based check. Field-name denial cannot catch a hidden value folded
    into a digest, which is exactly how both v1 leaks travelled.
    """
    tensors = [VersionedObservation(visible, e).model_tensor() for e in envelopes]
    for other in tensors[1:]:
        if not np.array_equal(tensors[0], other):
            raise ContractViolation(
                "the model tensor moved when only provenance changed; a provenance "
                "value is reaching model input"
            )
