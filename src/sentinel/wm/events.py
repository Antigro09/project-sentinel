"""Verifier-facing event schema.

Events are the channel through which a learned model says something an exact
probe can check. They are deliberately observable and deliberately small: this
is a probe interface, not a claim to have discovered the ontology of the world.

The two unknown members carry different meanings and different consequences.
`UNKNOWN_EVENT` is an in-schema event whose value the model could not resolve --
ordinary epistemic uncertainty. `MISSING_EVENT_REPRESENTATION` says the schema
itself cannot express what happened, which is a representation obligation and
must never be auto-promoted from one embedding anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.versioning import digest_of


class EventKind(str, Enum):
    OBJECT_APPEARED = "OBJECT_APPEARED"
    OBJECT_DISAPPEARED = "OBJECT_DISAPPEARED"
    INVENTORY_CHANGED = "INVENTORY_CHANGED"
    FOCUS_MOVED = "FOCUS_MOVED"
    FILE_STATE_CHANGED = "FILE_STATE_CHANGED"
    UI_STATE_CHANGED = "UI_STATE_CHANGED"
    ACTION_SUCCEEDED = "ACTION_SUCCEEDED"
    ACTION_FAILED = "ACTION_FAILED"
    CONSTRAINT_VIOLATED = "CONSTRAINT_VIOLATED"
    GOAL_PROGRESS_CHANGED = "GOAL_PROGRESS_CHANGED"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"
    MISSING_EVENT_REPRESENTATION = "MISSING_EVENT_REPRESENTATION"


EVENT_ORDER: tuple[EventKind, ...] = tuple(EventKind)
"""Fixed index order. Head logits map to this order and nothing else.

Frozen so that a checkpoint trained today can be scored tomorrow without a
silent permutation turning `ACTION_FAILED` into `ACTION_SUCCEEDED`.
"""

EVENT_INDEX: Mapping[EventKind, int] = {kind: i for i, kind in enumerate(EVENT_ORDER)}

SCHEMA_DIGEST: str = digest_of([k.value for k in EVENT_ORDER])
"""Identity of the event vocabulary. Expanding it changes every downstream hash."""


@dataclass(frozen=True, slots=True)
class StructuredEvent:
    """One observed event, with the observable that witnesses it.

    `witness` names the exact probe whose value changed. An event with no
    witness cannot be verified and is rejected, which is what stops the event
    head from quietly becoming a reward-label channel.
    """

    kind: EventKind
    witness: str
    detail: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            object.__setattr__(self, "detail", {})
        if self.kind is not EventKind.MISSING_EVENT_REPRESENTATION and not self.witness:
            raise ContractViolation(
                f"event {self.kind.value} has no witnessing probe; "
                "an unverifiable event is a label channel, not an observation"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "witness": self.witness, "detail": dict(self.detail)}


def event_distribution_from_logits(logits) -> dict[str, float]:
    """Softmax over the frozen event order, returned as a name-keyed map."""
    import numpy as np

    array = np.asarray(logits, dtype=np.float64).reshape(-1)
    if array.shape[0] != len(EVENT_ORDER):
        raise ContractViolation(
            f"event logits have width {array.shape[0]}, expected {len(EVENT_ORDER)}"
        )
    shifted = array - array.max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    return {kind.value: float(weights[i]) for i, kind in enumerate(EVENT_ORDER)}
