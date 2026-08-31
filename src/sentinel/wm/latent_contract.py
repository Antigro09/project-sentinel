"""Typed records and Protocols for the learned action-conditioned world model.

This is the Scale-0 interface layer. It is deliberately all types and no
learning: the point of Scale 0 is that every boundary between perception,
belief, dynamics, planning, verification, and memory can be crossed by a
deterministic fake before a neural network is allowed anywhere near it.

Three invariants are enforced by construction rather than by convention.

**Taint travels with data.** A record knows whether it came from development
collection, validation, a final split, an oracle, or an inherited pretrained
backbone. Training code refuses records carrying a taint it is not allowed to
see, so a leak becomes an exception instead of an unexplained score.

**Masks are not zeros.** A missing modality is represented by a mask bit, never
by a zero vector, because a zero vector is a perfectly good observation and the
model cannot tell the two apart.

**Uncertainty stays decomposed.** Aleatoric spread, epistemic ignorance, and
model inadequacy have different correct responses -- act, gather evidence,
expand the representation -- so they are never collapsed into one scalar in
storage, even when one scalar is what a particular decision consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from sentinel.wm.versioning import (
    ArrayDigest,
    CanonicalisationError,
    digest_of,
    require_digest,
)


class ContractViolation(ValueError):
    """A record is malformed. Fail closed; never repair silently."""


class Taint(str, Enum):
    """Where a value came from, and therefore who is allowed to look at it."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    FINAL = "final"
    ORACLE = "oracle"
    INHERITED_PRETRAINED = "inherited_pretrained"
    EVALUATOR_ONLY = "evaluator_only"


TRAIN_FORBIDDEN_TAINTS: frozenset[Taint] = frozenset(
    {Taint.VALIDATION, Taint.FINAL, Taint.ORACLE, Taint.EVALUATOR_ONLY}
)
"""Taints that may never appear on a record fed to a Scale-0 training step.

`ORACLE` is on the list even though oracle-assisted *trajectories* are 25% of
the collection mixture: the trajectory's observations and actions are ordinary
development data, while the oracle's own answer is not. The collector strips
the latter and labels the former `DEVELOPMENT`.
"""

INHERITED_TAINTS: frozenset[Taint] = frozenset({Taint.INHERITED_PRETRAINED})
"""Not forbidden -- frozen backbone features are the point -- but reported apart.

Capability that arrives through this channel is inherited, not Sentinel-learned,
and the resource report keeps the two columns separate.
"""


class RepresentationKind(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"


class Precision(str, Enum):
    BF16 = "bfloat16"
    FP16 = "float16"
    FP32 = "float32"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    GUI = "gui"
    STRUCTURED = "structured"
    GOAL = "goal"


@dataclass(frozen=True, slots=True)
class ModalityMask:
    """Which modalities are actually present in an observation.

    Stored as an explicit present/absent map over a declared modality set. An
    absent modality is absent; it is not a zero tensor. The mask is part of the
    latent cache key, because the same raw bytes encoded with a different mask
    are different features.
    """

    declared: tuple[Modality, ...]
    present: tuple[Modality, ...]

    def __post_init__(self) -> None:
        if len(set(self.declared)) != len(self.declared):
            raise ContractViolation(f"duplicate modality in declared set: {self.declared}")
        missing = set(self.present) - set(self.declared)
        if missing:
            raise ContractViolation(f"present modalities not declared: {sorted(m.value for m in missing)}")

    def is_present(self, modality: Modality) -> bool:
        return modality in self.present

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "declared": [m.value for m in self.declared],
            "present": sorted(m.value for m in self.present),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class EncoderIdentity:
    """Complete identity of a frozen perceptual backbone.

    Every field participates in the digest. That is the whole point: two runs
    that used "Qwen3-VL 4B" are not comparable unless they used the same
    revision, the same preprocessing, and the same numeric precision, and a
    cache that cannot tell those apart will silently serve one run's features
    to another.
    """

    provider: str
    model_name: str
    revision: str
    weight_digest: str
    preprocessing_digest: str
    precision: Precision
    license_record: str
    frozen: bool = True
    feature_dimension: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("provider", "model_name", "revision", "license_record"):
            if not getattr(self, name):
                raise ContractViolation(f"EncoderIdentity.{name} is empty")
        require_digest(self.weight_digest, "EncoderIdentity.weight_digest")
        require_digest(self.preprocessing_digest, "EncoderIdentity.preprocessing_digest")
        if not self.frozen:
            raise ContractViolation("Scale 0 through Scale 2 require frozen encoders")
        if self.feature_dimension <= 0:
            raise ContractViolation(
                f"EncoderIdentity.feature_dimension must be positive, got {self.feature_dimension}"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "revision": self.revision,
            "weight_digest": self.weight_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "precision": self.precision.value,
            "license_record": self.license_record,
            "frozen": self.frozen,
            "feature_dimension": self.feature_dimension,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    """One raw multimodal observation, before any encoder touches it.

    `structured_observation` holds only values the agent is allowed to see. The
    simulator's hidden state lives in a separate evaluator-only snapshot type
    and is rejected here by name and by taint.
    """

    episode_id: str
    step: int
    timestamp_ns: int
    modality_payloads: Mapping[str, ArrayDigest]
    structured_observation: Mapping[str, Any]
    modality_mask: ModalityMask
    available_action_digest: str
    environment_version: str
    taint: frozenset[Taint] = frozenset({Taint.DEVELOPMENT})

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ContractViolation(f"step must be non-negative, got {self.step}")
        if not self.episode_id:
            raise ContractViolation("ObservationEnvelope.episode_id is empty")
        require_digest(self.available_action_digest, "available_action_digest")
        require_digest(self.environment_version, "environment_version")
        for key, ref in self.modality_payloads.items():
            if not isinstance(ref, ArrayDigest):
                raise ContractViolation(
                    f"modality payload {key!r} is {type(ref).__name__}; "
                    "raw arrays must be wrapped in ArrayDigest so provenance survives"
                )
        declared = {m.value for m in self.modality_mask.declared}
        unexpected = set(self.modality_payloads) - declared
        if unexpected:
            raise ContractViolation(f"payloads for undeclared modalities: {sorted(unexpected)}")
        absent_with_payload = set(self.modality_payloads) - {m.value for m in self.modality_mask.present}
        if absent_with_payload:
            raise ContractViolation(
                f"modalities marked absent but carrying payload: {sorted(absent_with_payload)}"
            )
        reject_hidden_fields(self.structured_observation, "structured_observation")

    def content_dict(self) -> dict[str, Any]:
        """What the agent can actually see, with no positional metadata."""
        return {
            "modality_payloads": {k: v.canonical_dict() for k, v in sorted(self.modality_payloads.items())},
            "structured_observation": dict(self.structured_observation),
            "modality_mask": self.modality_mask.canonical_dict(),
            "available_action_digest": self.available_action_digest,
            "environment_version": self.environment_version,
        }

    def canonical_dict(self) -> dict[str, Any]:
        return {
            **self.content_dict(),
            "episode_id": self.episode_id,
            "step": self.step,
            "taint": sorted(t.value for t in self.taint),
        }

    @property
    def content_digest(self) -> str:
        """Identity of the observation *content* alone.

        Episode id, step, and wall-clock time are excluded, and that exclusion
        is load-bearing twice over. It is what lets the encoder cache dedupe two
        identical observations reached by different routes -- with the episode id
        hashed in, the cache never hits and its hit ratio measures nothing. And
        it is what makes the duplicate-frame leak check able to see anything at
        all: a frame that appears in train and in a held-out split is only
        detectable if the two hash the same.
        """
        return digest_of(self.content_dict())

    @property
    def digest(self) -> str:
        """Positional identity: this content, in this episode, at this step."""
        return digest_of(self.canonical_dict())


HIDDEN_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "hidden_state",
        "hidden_phase",
        "hidden_history",
        "simulator_state",
        "full_state",
        "snapshot",
        "target_program",
        "target_action_sequence",
        "expected_observation",
        "expected_reward",
        "evaluator_answer",
        "final_label",
        "solution",
        "oracle_answer",
        "held_out_mechanic",
    }
)
"""Names that must never appear in anything routed to model input.

The list is a tripwire, not a security boundary -- a determined leak can rename
a field. Its job is to catch the accident that actually happens: an adapter
author putting the simulator's own state into the observation dict because it
was convenient for debugging.
"""


def reject_hidden_fields(mapping: Mapping[str, Any], where: str) -> None:
    """Raise if a model-visible mapping carries an evaluator-only field name."""
    offending = sorted(set(mapping) & HIDDEN_FIELD_NAMES)
    if offending:
        raise ContractViolation(
            f"{where} contains evaluator-only field(s) {offending}; "
            "hidden simulator state must never reach model input"
        )


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """Frozen-backbone features for one observation.

    Carries the encoder identity that produced it so that a feature vector can
    never be silently reused across a precision change or a revision bump.
    """

    encoder_identity: EncoderIdentity
    source_observation_digest: str
    features: ArrayDigest
    modality_mask: ModalityMask
    taint: frozenset[Taint] = frozenset({Taint.DEVELOPMENT, Taint.INHERITED_PRETRAINED})

    def __post_init__(self) -> None:
        require_digest(self.source_observation_digest, "source_observation_digest")
        if self.features.shape and self.features.shape[-1] != self.encoder_identity.feature_dimension:
            raise ContractViolation(
                f"feature width {self.features.shape[-1]} disagrees with declared encoder "
                f"dimension {self.encoder_identity.feature_dimension}"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "encoder_identity": self.encoder_identity.canonical_dict(),
            "source_observation_digest": self.source_observation_digest,
            "features": self.features.canonical_dict(),
            "modality_mask": self.modality_mask.canonical_dict(),
            "taint": sorted(t.value for t in self.taint),
        }


@dataclass(frozen=True, slots=True)
class LatentObservation:
    """Projected latent for one observation, in one representation arm."""

    episode_id: str
    step: int
    encoder_identity: EncoderIdentity
    projector_digest: str
    representation_kind: RepresentationKind
    modality_mask: ModalityMask
    source_observation_digest: str
    continuous_values: ArrayDigest | None = None
    discrete_codes: ArrayDigest | None = None

    def __post_init__(self) -> None:
        require_digest(self.projector_digest, "projector_digest")
        require_digest(self.source_observation_digest, "source_observation_digest")
        kind = self.representation_kind
        has_continuous = self.continuous_values is not None
        has_discrete = self.discrete_codes is not None
        if kind is RepresentationKind.CONTINUOUS and not (has_continuous and not has_discrete):
            raise ContractViolation("continuous arm requires continuous_values and no discrete_codes")
        if kind is RepresentationKind.DISCRETE and not (has_discrete and not has_continuous):
            raise ContractViolation("discrete arm requires discrete_codes and no continuous_values")
        if kind is RepresentationKind.HYBRID and not (has_continuous and has_discrete):
            raise ContractViolation("hybrid arm requires both continuous_values and discrete_codes")
        if has_discrete and "int" not in self.discrete_codes.dtype:
            raise ContractViolation(
                f"discrete_codes dtype must be integral, got {self.discrete_codes.dtype}"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "encoder_identity": self.encoder_identity.canonical_dict(),
            "projector_digest": self.projector_digest,
            "representation_kind": self.representation_kind.value,
            "modality_mask": self.modality_mask.canonical_dict(),
            "source_observation_digest": self.source_observation_digest,
            "continuous_values": self.continuous_values.canonical_dict() if self.continuous_values else None,
            "discrete_codes": self.discrete_codes.canonical_dict() if self.discrete_codes else None,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class UncertaintyTriple:
    """The three failure sources, kept apart.

    Collapsing these is the standard way a model-based agent ends up confidently
    wrong: `aleatoric` says the world is noisy and acting is fine, `epistemic`
    says gather evidence, and `inadequacy` says no model in the class explains
    what was seen and the representation itself is the problem.
    """

    aleatoric: float
    epistemic: float
    inadequacy: float

    def __post_init__(self) -> None:
        for name in ("aleatoric", "epistemic", "inadequacy"):
            value = float(getattr(self, name))
            if value < 0.0 or value != value:
                raise ContractViolation(f"UncertaintyTriple.{name} must be finite and non-negative, got {value}")

    def canonical_dict(self) -> dict[str, float]:
        return {
            "aleatoric": float(self.aleatoric),
            "epistemic": float(self.epistemic),
            "inadequacy": float(self.inadequacy),
        }

    def scalar(self, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """One number for a decision that needs one, with the parts still stored."""
        wa, we, wi = weights
        return wa * self.aleatoric + we * self.epistemic + wi * self.inadequacy


@dataclass(frozen=True, slots=True)
class BeliefState:
    """Recurrent belief at one step, with its provenance."""

    episode_id: str
    step: int
    deterministic_state: ArrayDigest
    stochastic_state: ArrayDigest
    representation_kind: RepresentationKind
    retrieved_memory_ids: tuple[str, ...]
    uncertainty: UncertaintyTriple
    model_version: str

    def __post_init__(self) -> None:
        require_digest(self.model_version, "BeliefState.model_version")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "deterministic_state": self.deterministic_state.canonical_dict(),
            "stochastic_state": self.stochastic_state.canonical_dict(),
            "representation_kind": self.representation_kind.value,
            "retrieved_memory_ids": list(self.retrieved_memory_ids),
            "uncertainty": self.uncertainty.canonical_dict(),
            "model_version": self.model_version,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class TransitionPrediction:
    """What the dynamics model says will happen if an action is taken."""

    next_latent: ArrayDigest
    event_distribution: Mapping[str, float]
    reward_mean: float
    reward_variance: float
    termination_probability: float
    uncertainty: UncertaintyTriple
    rollout_support_scope: str
    model_version: str
    action: int

    def __post_init__(self) -> None:
        require_digest(self.model_version, "TransitionPrediction.model_version")
        total = sum(self.event_distribution.values())
        if self.event_distribution and abs(total - 1.0) > 1e-4:
            raise ContractViolation(f"event distribution sums to {total}, not 1")
        if not 0.0 <= self.termination_probability <= 1.0:
            raise ContractViolation(
                f"termination_probability {self.termination_probability} outside [0,1]"
            )
        if self.reward_variance < 0.0:
            raise ContractViolation(f"reward_variance {self.reward_variance} is negative")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "next_latent": self.next_latent.canonical_dict(),
            "event_distribution": {k: float(v) for k, v in sorted(self.event_distribution.items())},
            "reward_mean": float(self.reward_mean),
            "reward_variance": float(self.reward_variance),
            "termination_probability": float(self.termination_probability),
            "uncertainty": self.uncertainty.canonical_dict(),
            "rollout_support_scope": self.rollout_support_scope,
            "model_version": self.model_version,
            "action": int(self.action),
        }


@dataclass(frozen=True, slots=True)
class Counterexample:
    """An exact observable that contradicted a prediction."""

    probe_name: str
    predicted: Any
    actual: Any
    step: int
    episode_id: str

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "probe_name": self.probe_name,
            "predicted": self.predicted,
            "actual": self.actual,
            "step": self.step,
            "episode_id": self.episode_id,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of comparing a prediction against exact observables.

    Accuracy and coverage are separate fields for the reason `contract.py` gives
    about abstention: a predictor that says nothing is never wrong, and a scheme
    that reports one number rates it as perfect.
    """

    accepted_observables: tuple[str, ...]
    rejected_observables: tuple[str, ...]
    unprobed_observables: tuple[str, ...]
    counterexamples: tuple[Counterexample, ...]
    constraint_violations: tuple[str, ...]
    verifier_version: str
    required_probe_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_digest(self.verifier_version, "verifier_version")
        missing_required = set(self.required_probe_names) - (
            set(self.accepted_observables) | set(self.rejected_observables)
        )
        if missing_required:
            raise ContractViolation(
                f"evaluator-required probes were not executed: {sorted(missing_required)}"
            )

    @property
    def accuracy(self) -> float:
        """Share of *probed* observables that matched. Undefined with no probes."""
        probed = len(self.accepted_observables) + len(self.rejected_observables)
        if probed == 0:
            return float("nan")
        return len(self.accepted_observables) / probed

    @property
    def coverage(self) -> float:
        """Share of *available* observables that were probed at all."""
        total = (
            len(self.accepted_observables)
            + len(self.rejected_observables)
            + len(self.unprobed_observables)
        )
        if total == 0:
            return 0.0
        return (total - len(self.unprobed_observables)) / total

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "accepted_observables": sorted(self.accepted_observables),
            "rejected_observables": sorted(self.rejected_observables),
            "unprobed_observables": sorted(self.unprobed_observables),
            "counterexamples": [c.canonical_dict() for c in self.counterexamples],
            "constraint_violations": sorted(self.constraint_violations),
            "verifier_version": self.verifier_version,
            "required_probe_names": sorted(self.required_probe_names),
        }


@runtime_checkable
class FrozenEncoderAdapter(Protocol):
    """A frozen perceptual backbone behind one interface.

    Scale 0 requires at least two independent implementations to satisfy this
    protocol with no evaluator change, which is what makes "the backbone did it"
    a testable attribution rather than a worry.
    """

    @property
    def identity(self) -> EncoderIdentity: ...

    def encode(self, observation: ObservationEnvelope) -> EncodedObservation: ...

    def health_check(self) -> Mapping[str, Any]: ...


@runtime_checkable
class LatentRepresentation(Protocol):
    kind: RepresentationKind
    dimension_budget: int

    def project(self, encoded: EncodedObservation) -> LatentObservation: ...

    def validate(self, latent: LatentObservation) -> None: ...


@runtime_checkable
class BeliefUpdater(Protocol):
    def initial(self, episode_id: str) -> BeliefState: ...

    def update(
        self,
        previous: BeliefState,
        latent: LatentObservation,
        previous_action: int | None,
        previous_reward: float | None,
        retrieved_memory: Sequence[str],
    ) -> BeliefState: ...


@runtime_checkable
class ActionConditionedDynamics(Protocol):
    """The action argument is mandatory.

    An action-blind predictor is expressible only as an explicit falsifying
    control, never as an instance of this protocol, because Theorem 1's
    construction shows passive data cannot identify what an untried action does.
    """

    def predict(self, belief: BeliefState, action: int) -> TransitionPrediction: ...


__all__ = [
    "ActionConditionedDynamics",
    "ArrayDigest",
    "BeliefState",
    "BeliefUpdater",
    "CanonicalisationError",
    "ContractViolation",
    "Counterexample",
    "EncodedObservation",
    "EncoderIdentity",
    "FrozenEncoderAdapter",
    "HIDDEN_FIELD_NAMES",
    "INHERITED_TAINTS",
    "LatentObservation",
    "LatentRepresentation",
    "Modality",
    "ModalityMask",
    "ObservationEnvelope",
    "Precision",
    "RepresentationKind",
    "TRAIN_FORBIDDEN_TAINTS",
    "Taint",
    "TransitionPrediction",
    "UncertaintyTriple",
    "VerificationResult",
    "reject_hidden_fields",
]
