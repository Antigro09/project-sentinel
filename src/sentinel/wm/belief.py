"""The recurrent belief boundary, with a deterministic fake behind it.

`BeliefState` is the object the planner, the verifier bridge, and the memory
retriever all read. This module gives that boundary an implementation with no
learned parameters, so the typed flow -- latent in, belief out, digest stable
across a restart -- can be exercised before a recurrence with weights exists.

The fake is a hash chain rather than a random walk, and that is the point. The
belief at step *t* depends on the whole history of latents, actions, and
outcomes that produced it, so two histories that differ anywhere produce
different beliefs, and the same history in a fresh process produces the same
one. Those are exactly the two properties the aliasing fixture and the restart
gate ask of the real recurrence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from sentinel.wm.latent_contract import (
    BeliefState,
    ContractViolation,
    LatentObservation,
    RepresentationKind,
    UncertaintyTriple,
)
from sentinel.wm.versioning import digest_array, digest_of


def _expand(material: bytes, count: int) -> np.ndarray:
    blocks: list[bytes] = []
    counter = 0
    while sum(len(b) for b in blocks) < count * 4:
        blocks.append(hashlib.sha256(material + counter.to_bytes(4, "big")).digest())
        counter += 1
    integers = np.frombuffer(b"".join(blocks)[: count * 4], dtype=np.uint32)
    return ((integers.astype(np.float64) / np.float64(2**32)) * 2.0 - 1.0).astype(np.float32)


@dataclass
class FakeBeliefUpdater:
    """A history-dependent belief with no parameters.

    `epistemic` falls as the history lengthens and `inadequacy` stays at zero:
    a fake has no model class to be inadequate for, and reporting a nonzero
    inadequacy would put a number into the report that means nothing.
    """

    deterministic_width: int = 256
    stochastic_width: int = 64
    representation: RepresentationKind = RepresentationKind.HYBRID
    model_version: str = field(default="")
    _chain: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_version:
            self.model_version = digest_of(
                {
                    "belief": "deterministic-fake",
                    "deterministic_width": self.deterministic_width,
                    "stochastic_width": self.stochastic_width,
                    "representation": self.representation.value,
                    "version": 1,
                }
            )

    def _state_from(self, episode_id: str, step: int, chain: str) -> BeliefState:
        deterministic = _expand(chain.encode() + b"det", self.deterministic_width)
        stochastic = _expand(chain.encode() + b"sto", self.stochastic_width)
        return BeliefState(
            episode_id=episode_id,
            step=step,
            deterministic_state=digest_array(deterministic),
            stochastic_state=digest_array(stochastic),
            representation_kind=self.representation,
            retrieved_memory_ids=(),
            uncertainty=UncertaintyTriple(
                aleatoric=0.5,
                epistemic=1.0 / (1.0 + step),
                inadequacy=0.0,
            ),
            model_version=self.model_version,
        )

    def initial(self, episode_id: str) -> BeliefState:
        chain = digest_of({"episode": episode_id, "model": self.model_version})
        self._chain[episode_id] = chain
        return self._state_from(episode_id, 0, chain)

    def update(
        self,
        previous: BeliefState,
        latent: LatentObservation,
        previous_action: int | None,
        previous_reward: float | None,
        retrieved_memory: Sequence[str] = (),
    ) -> BeliefState:
        if latent.projector_digest is None:  # pragma: no cover - guarded by the record
            raise ContractViolation("latent has no projector identity")
        chain = digest_of(
            {
                "previous": previous.digest,
                "latent": latent.digest,
                "previous_action": previous_action,
                "previous_reward": previous_reward,
                "retrieved": sorted(retrieved_memory),
            }
        )
        state = self._state_from(previous.episode_id, previous.step + 1, chain)
        return BeliefState(
            episode_id=state.episode_id,
            step=state.step,
            deterministic_state=state.deterministic_state,
            stochastic_state=state.stochastic_state,
            representation_kind=state.representation_kind,
            retrieved_memory_ids=tuple(retrieved_memory),
            uncertainty=state.uncertainty,
            model_version=state.model_version,
        )
