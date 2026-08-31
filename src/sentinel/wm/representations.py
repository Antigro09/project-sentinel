"""Deterministic fake representations for the three arms.

`ARCHITECTURE.md` asks that every boundary be runnable with deterministic fake
components *before* a neural model is introduced. These are those components for
the projection boundary: they satisfy the `LatentRepresentation` protocol, they
produce real `LatentObservation` records with real digests, and they contain no
learned parameters at all.

Their value is not that they are cheap. It is that they let the typed contract
be exercised, restarted, hashed, and leak-checked with no model in the loop, so
a failure at that layer is unambiguously a contract failure. When the MLX arms
in `models.py` arrive, they are answering a specification that already ran.

The features are derived from the encoder's output by a fixed hash, so the same
observation gives the same latent in any process, and a different observation
gives a different one. That is enough for the schema tests to have teeth: a
downstream component that ignores its input still fails them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from sentinel.wm.latent_contract import (
    ContractViolation,
    EncodedObservation,
    LatentObservation,
    RepresentationKind,
)
from sentinel.wm.versioning import digest_array, digest_of


def _stream(material: bytes, count: int, dtype: str = "f4") -> np.ndarray:
    blocks: list[bytes] = []
    counter = 0
    width = 4
    while sum(len(b) for b in blocks) < count * width:
        blocks.append(hashlib.sha256(material + counter.to_bytes(4, "big")).digest())
        counter += 1
    raw = b"".join(blocks)[: count * width]
    integers = np.frombuffer(raw, dtype=np.uint32)
    if dtype == "f4":
        return ((integers.astype(np.float64) / np.float64(2**32)) * 2.0 - 1.0).astype(np.float32)
    return integers


@dataclass
class _FakeRepresentation:
    """Shared machinery. Not used directly; the three arms below are the API."""

    kind: RepresentationKind
    dimension_budget: int
    code_groups: int = 32
    code_categories: int = 32
    version: int = 1

    def __post_init__(self) -> None:
        if self.dimension_budget <= 0:
            raise ContractViolation("dimension_budget must be positive")
        if self.kind is RepresentationKind.HYBRID and self.dimension_budget % 2:
            raise ContractViolation("the hybrid arm splits its budget in two; it must be even")

    @property
    def projector_digest(self) -> str:
        """Identity of this projection. Part of every cache key it touches."""
        return digest_of(
            {
                "projector": "deterministic-fake",
                "kind": self.kind.value,
                "dimension_budget": self.dimension_budget,
                "code_groups": self.code_groups,
                "code_categories": self.code_categories,
                "version": self.version,
            }
        )

    def _material(self, encoded: EncodedObservation) -> bytes:
        return (
            encoded.source_observation_digest.encode()
            + encoded.encoder_identity.digest.encode()
            + self.projector_digest.encode()
        )

    def _continuous(self, encoded: EncodedObservation, width: int) -> np.ndarray:
        return _stream(self._material(encoded) + b"continuous", width)

    def _codes(self, encoded: EncodedObservation) -> np.ndarray:
        draws = _stream(self._material(encoded) + b"codes", self.code_groups, dtype="u4")
        return (draws % self.code_categories).astype(np.int32)

    def _build(
        self,
        encoded: EncodedObservation,
        continuous: np.ndarray | None,
        codes: np.ndarray | None,
    ) -> LatentObservation:
        return LatentObservation(
            episode_id="",
            step=0,
            encoder_identity=encoded.encoder_identity,
            projector_digest=self.projector_digest,
            representation_kind=self.kind,
            modality_mask=encoded.modality_mask,
            source_observation_digest=encoded.source_observation_digest,
            continuous_values=digest_array(continuous) if continuous is not None else None,
            discrete_codes=digest_array(codes) if codes is not None else None,
        )

    def validate(self, latent: LatentObservation) -> None:
        """Fail closed on a latent that does not belong to this projection."""
        if latent.representation_kind is not self.kind:
            raise ContractViolation(
                f"latent is {latent.representation_kind.value}, this arm is {self.kind.value}"
            )
        if latent.projector_digest != self.projector_digest:
            raise ContractViolation(
                "latent was produced by a different projector; its cache entries and this "
                "arm's are not interchangeable"
            )
        expected = self.dimension_budget
        if self.kind is RepresentationKind.HYBRID:
            expected = self.dimension_budget // 2
        if latent.continuous_values is not None and latent.continuous_values.size != expected:
            raise ContractViolation(
                f"continuous part has width {latent.continuous_values.size}, expected {expected}"
            )
        if latent.discrete_codes is not None and latent.discrete_codes.size != self.code_groups:
            raise ContractViolation(
                f"discrete part has {latent.discrete_codes.size} groups, expected {self.code_groups}"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "dimension_budget": self.dimension_budget,
            "code_groups": self.code_groups,
            "code_categories": self.code_categories,
            "projector_digest": self.projector_digest,
        }


@dataclass
class FakeContinuousRepresentation(_FakeRepresentation):
    """Real-valued latent, no codes."""

    kind: RepresentationKind = RepresentationKind.CONTINUOUS
    dimension_budget: int = 512

    def project(self, encoded: EncodedObservation) -> LatentObservation:
        return self._build(encoded, self._continuous(encoded, self.dimension_budget), None)

    def values(self, encoded: EncodedObservation) -> np.ndarray:
        return self._continuous(encoded, self.dimension_budget)


@dataclass
class FakeDiscreteRepresentation(_FakeRepresentation):
    """Categorical latent, no real-valued part."""

    kind: RepresentationKind = RepresentationKind.DISCRETE
    dimension_budget: int = 512

    def project(self, encoded: EncodedObservation) -> LatentObservation:
        return self._build(encoded, None, self._codes(encoded))

    def values(self, encoded: EncodedObservation) -> np.ndarray:
        return self._codes(encoded)


@dataclass
class FakeHybridRepresentation(_FakeRepresentation):
    """Continuous perception plus discrete event/mechanism variables."""

    kind: RepresentationKind = RepresentationKind.HYBRID
    dimension_budget: int = 512

    def project(self, encoded: EncodedObservation) -> LatentObservation:
        half = self.dimension_budget // 2
        return self._build(encoded, self._continuous(encoded, half), self._codes(encoded))


def fake_representation(kind: RepresentationKind, dimension_budget: int = 512):
    return {
        RepresentationKind.CONTINUOUS: FakeContinuousRepresentation,
        RepresentationKind.DISCRETE: FakeDiscreteRepresentation,
        RepresentationKind.HYBRID: FakeHybridRepresentation,
    }[kind](dimension_budget=dimension_budget)
