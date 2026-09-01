"""E. The multimodal observation packet.

One typed container for everything a model is allowed to see at a step, so that
adding a modality later is a mask bit rather than a schema change. Audio is
declared and empty from the start for exactly that reason: a channel introduced
after a gate has passed changes the interface every earlier result was measured
against.

The packet carries no hidden simulator state and no evaluator field. It is built
from an `ObservationEnvelope`, which already rejects those by name, plus the
interface's own visual slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sentinel.wm.latent_contract import (
    ArrayDigest,
    ContractViolation,
    ModalityMask,
    ObservationEnvelope,
    reject_hidden_fields,
)
from sentinel.wm.versioning import digest_array, digest_of

SLOT_COUNT = 16
SLOT_WIDTH = 256
"""Every non-oracle interface emits this shape.

A matched slot shape is what makes the interfaces comparable at all: an arm with
more slots or a wider slot would be a different capacity, and a difference in
capacity would be indistinguishable from a difference in representation.
"""

MAX_GOAL_TOKENS = 16


class ActionResult(str):
    NONE = "none"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def tokenise_goal(text: str, vocabulary: Mapping[str, int]) -> tuple[int, ...]:
    """A fixed word-level vocabulary, padded to a constant length.

    Deliberately not the backbone's tokenizer: the language channel has to be
    identical across all eight interfaces, or a comparison between them would be
    partly a comparison of tokenizers.
    """
    tokens = [vocabulary.get(word, vocabulary["<unk>"]) for word in text.lower().split()]
    tokens = tokens[:MAX_GOAL_TOKENS]
    return tuple(tokens + [vocabulary["<pad>"]] * (MAX_GOAL_TOKENS - len(tokens)))


def build_vocabulary(phrases: Any) -> dict[str, int]:
    words = sorted({word for phrase in phrases for word in phrase.lower().split()})
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for index, word in enumerate(words):
        vocabulary[word] = index + 2
    return vocabulary


@dataclass(frozen=True, slots=True)
class ObservationPacket:
    """What one interface hands the model for one step."""

    visual_slots: ArrayDigest
    language_goal_tokens: tuple[int, ...]
    scalar_sensors: Mapping[str, float]
    previous_action: int | None
    action_result: str
    timestamp_ns: int
    modality_masks: ModalityMask
    audio_slots: ArrayDigest | None = None
    interface_name: str = ""
    source_observation_digest: str = ""

    def __post_init__(self) -> None:
        if self.visual_slots.shape != (SLOT_COUNT, SLOT_WIDTH):
            raise ContractViolation(
                f"visual slots are {self.visual_slots.shape}, every interface must emit "
                f"({SLOT_COUNT}, {SLOT_WIDTH}) so the arms are comparable"
            )
        if len(self.language_goal_tokens) != MAX_GOAL_TOKENS:
            raise ContractViolation(
                f"goal tokens are length {len(self.language_goal_tokens)}, expected "
                f"{MAX_GOAL_TOKENS} after padding"
            )
        if self.action_result not in (
            ActionResult.NONE,
            ActionResult.SUCCEEDED,
            ActionResult.FAILED,
        ):
            raise ContractViolation(f"unknown action_result {self.action_result!r}")
        reject_hidden_fields(self.scalar_sensors, "ObservationPacket.scalar_sensors")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "visual_slots": self.visual_slots.canonical_dict(),
            "language_goal_tokens": list(self.language_goal_tokens),
            "scalar_sensors": {k: float(v) for k, v in sorted(self.scalar_sensors.items())},
            "previous_action": self.previous_action,
            "action_result": self.action_result,
            "modality_masks": self.modality_masks.canonical_dict(),
            "audio_slots": self.audio_slots.canonical_dict() if self.audio_slots else None,
            "interface_name": self.interface_name,
            "source_observation_digest": self.source_observation_digest,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


def build_packet(
    observation: ObservationEnvelope,
    slots: np.ndarray,
    *,
    vocabulary: Mapping[str, int],
    interface_name: str,
    previous_action: int | None,
    action_result: str,
    scalar_sensors: Mapping[str, float] | None = None,
) -> ObservationPacket:
    if slots.shape != (SLOT_COUNT, SLOT_WIDTH):
        raise ContractViolation(
            f"interface {interface_name} emitted {slots.shape}, expected "
            f"({SLOT_COUNT}, {SLOT_WIDTH})"
        )
    goal_text = str(observation.structured_observation.get("goal_text", ""))
    return ObservationPacket(
        visual_slots=digest_array(slots),
        language_goal_tokens=tokenise_goal(goal_text, vocabulary),
        scalar_sensors=dict(scalar_sensors or {}),
        previous_action=previous_action,
        action_result=action_result,
        timestamp_ns=observation.timestamp_ns,
        modality_masks=observation.modality_mask,
        audio_slots=None,
        interface_name=interface_name,
        source_observation_digest=observation.content_digest,
    )
