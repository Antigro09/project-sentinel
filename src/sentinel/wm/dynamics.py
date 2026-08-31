"""The action-conditioned prediction boundary, with a deterministic fake behind it.

The action argument is mandatory in the protocol and it is mandatory here. The
fake's successor depends on the action, so the action-intervention fixture has
something to separate; an implementation that ignored it would be an
action-blind control, which is a different object with a different name.

Everything the fake returns is a valid `TransitionPrediction`: a normalised
event distribution over the frozen vocabulary, a reward mean and variance, a
termination probability in range, and the three uncertainty components kept
apart. That is what lets the planner, the verifier bridge, and the authority gate
be exercised end to end before any of them has ever seen a trained model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from sentinel.wm.events import EVENT_ORDER, EventKind
from sentinel.wm.latent_contract import (
    BeliefState,
    ContractViolation,
    TransitionPrediction,
    UncertaintyTriple,
)
from sentinel.wm.versioning import digest_array, digest_of


def _unit(material: bytes) -> float:
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / float(1 << 64)


def _expand(material: bytes, count: int) -> np.ndarray:
    blocks: list[bytes] = []
    counter = 0
    while sum(len(b) for b in blocks) < count * 4:
        blocks.append(hashlib.sha256(material + counter.to_bytes(4, "big")).digest())
        counter += 1
    integers = np.frombuffer(b"".join(blocks)[: count * 4], dtype=np.uint32)
    return ((integers.astype(np.float64) / np.float64(2**32)) * 2.0 - 1.0).astype(np.float32)


@dataclass
class FakeActionConditionedDynamics:
    """A parameterless predictor whose output genuinely depends on the action."""

    latent_width: int = 512
    action_count: int = 4
    support_scope: str = "development"
    model_version: str = field(default="")

    def __post_init__(self) -> None:
        if not self.model_version:
            self.model_version = digest_of(
                {
                    "dynamics": "deterministic-fake",
                    "latent_width": self.latent_width,
                    "action_count": self.action_count,
                    "version": 1,
                }
            )

    def _events(self, material: bytes) -> Mapping[str, float]:
        logits = _expand(material + b"events", len(EVENT_ORDER)).astype(np.float64)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        return {kind.value: float(weights[i]) for i, kind in enumerate(EVENT_ORDER)}

    def predict(self, belief: BeliefState, action: int) -> TransitionPrediction:
        if not 0 <= int(action) < self.action_count:
            raise ContractViolation(
                f"action {action} is outside the declared set of {self.action_count}"
            )
        material = (belief.digest + f":{int(action)}").encode()
        return TransitionPrediction(
            next_latent=digest_array(_expand(material + b"latent", self.latent_width)),
            event_distribution=self._events(material),
            reward_mean=float(_unit(material + b"reward") * 2.0 - 1.0),
            reward_variance=float(_unit(material + b"variance")),
            termination_probability=float(_unit(material + b"terminate")),
            uncertainty=UncertaintyTriple(
                aleatoric=float(_unit(material + b"aleatoric")),
                epistemic=float(_unit(material + b"epistemic")),
                inadequacy=0.0,
            ),
            rollout_support_scope=self.support_scope,
            model_version=self.model_version,
            action=int(action),
        )


@dataclass
class ActionBlindDynamics(FakeActionConditionedDynamics):
    """The falsifying control. It exists to fail the intervention fixture.

    Deliberately not an `ActionConditionedDynamics`: it drops the action, so its
    successor is a function of the belief alone. Theorem 1's construction says a
    passive dataset cannot distinguish it from the real thing, and the
    action-intervention fixture is the experiment that can.
    """

    def predict(self, belief: BeliefState, action: int) -> TransitionPrediction:
        prediction = super().predict(belief, 0)
        return TransitionPrediction(
            next_latent=prediction.next_latent,
            event_distribution=prediction.event_distribution,
            reward_mean=prediction.reward_mean,
            reward_variance=prediction.reward_variance,
            termination_probability=prediction.termination_probability,
            uncertainty=prediction.uncertainty,
            rollout_support_scope=prediction.rollout_support_scope,
            model_version=self.model_version,
            action=int(action),
        )
