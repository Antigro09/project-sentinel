"""Frozen-encoder adapters behind one interface.

Scale 0's encoder gate is a systems gate, not a capability gate: two independent
backbones must satisfy the same adapter with no change to the evaluator. What
the adapter is *for* is attribution -- if the frozen backbone already solves the
task, the Sentinel addition has not earned its place, and that can only be
measured when swapping the backbone is a configuration change.

Two implementations live here.

`DeterministicControlEncoder` is the **random frozen encoder control** from the
required-arms list, made reproducible. Its features are a fixed pseudo-random
projection of the observation digest, so they are deterministic across processes
and carry no pretrained knowledge whatsoever. It is a control and a test double.
It is never a substitute for a named matrix backbone, and its identity says so
in its provider field.

`CachedEncoder` wraps any adapter with the content-addressed cache, so the
throughput report can separate cold encode cost from warm cache cost.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sentinel.wm.cache import CacheIdentity, LatentCache
from sentinel.wm.latent_contract import (
    ContractViolation,
    EncodedObservation,
    EncoderIdentity,
    ModalityMask,
    ObservationEnvelope,
    Precision,
    Taint,
)
from sentinel.wm.versioning import digest_of

CONTROL_PROVIDER = "sentinel-control"
"""Provider string that marks an encoder as a control rather than a backbone.

Anything carrying it is excluded from matrix cells by
`sentinel.wm.matrix.assert_matrix_encoder`, so a control can never be quietly
promoted into a frozen cell.
"""

_PRECISION_DTYPE: Mapping[Precision, Any] = {
    Precision.BF16: np.float32,  # numpy has no bfloat16; see `_quantise_bf16`
    Precision.FP16: np.float16,
    Precision.FP32: np.float32,
}


def _quantise_bf16(values: np.ndarray) -> np.ndarray:
    """Round a float32 array to bfloat16 precision, keeping the float32 dtype.

    numpy has no bfloat16, and silently storing full float32 while *claiming*
    bfloat16 would make the precision field of the encoder identity a lie -- the
    exact failure the cache key exists to catch. Truncating the mantissa gives a
    value that is bit-identical to what a bf16 pipeline would produce, in a
    container numpy can hold.
    """
    as_int = values.astype(np.float32).view(np.uint32)
    # Round-to-nearest-even on the low 16 bits before truncating them.
    rounded = (as_int + 0x7FFF + ((as_int >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def pack_bf16(values: np.ndarray) -> np.ndarray:
    """Store bf16-quantised float32 as the two bytes it actually occupies.

    A bf16 value *is* the high half of its float32 bit pattern, so keeping the
    low half costs twice the disk for no information. This matters beyond
    tidiness: the resource plan budgets the latent cache at two bytes per
    coordinate, and a cache silently storing four would make the measured
    footprint disagree with the arithmetic for a reason nobody could see.
    """
    return _quantise_bf16(values).view(np.uint32).astype(np.uint32).__rshift__(16).astype(np.uint16)


def unpack_bf16(packed: np.ndarray) -> np.ndarray:
    """Inverse of `pack_bf16`. Exact, because nothing was lost in packing."""
    widened = packed.astype(np.uint32) << 16
    return widened.view(np.float32)


def apply_precision(values: np.ndarray, precision: Precision) -> np.ndarray:
    if precision is Precision.BF16:
        return _quantise_bf16(values)
    return values.astype(_PRECISION_DTYPE[precision])


@dataclass(frozen=True, slots=True)
class EncoderHealth:
    ok: bool
    detail: str
    cold_load_seconds: float
    feature_dimension: int
    deterministic: bool

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "detail": self.detail,
            "cold_load_seconds": float(self.cold_load_seconds),
            "feature_dimension": int(self.feature_dimension),
            "deterministic": self.deterministic,
        }


@dataclass
class DeterministicControlEncoder:
    """Reproducible random frozen encoder. A control, never a backbone.

    Features come from a keyed hash of the observation *content* digest expanded
    into a fixed-width vector. That gives three properties the fake-model dry run
    needs: identical output in any process, no inherited capability, and a
    genuine dependence on the observation content so that a downstream model that
    ignores the observation still fails.

    Content rather than positional identity, because a frozen encoder is a
    function of what it was shown. Seeding on the positional digest would make
    the same frame encode differently in two episodes, and the cache -- which
    keys on content -- would then serve one of the two answers to both.
    """

    feature_dimension: int = 512
    precision: Precision = Precision.BF16
    variant: str = "control-a"
    _identity: EncoderIdentity = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.feature_dimension <= 0:
            raise ContractViolation("feature_dimension must be positive")
        self._identity = EncoderIdentity(
            provider=CONTROL_PROVIDER,
            model_name=f"deterministic-hash-projection/{self.variant}",
            revision="1",
            weight_digest=digest_of({"variant": self.variant, "dimension": self.feature_dimension}),
            preprocessing_digest=digest_of({"preprocessing": "digest-expansion", "version": 1}),
            precision=self.precision,
            license_record="internal-control-no-pretrained-weights",
            frozen=True,
            feature_dimension=self.feature_dimension,
            notes="random frozen encoder control; carries no pretrained knowledge",
        )

    @property
    def identity(self) -> EncoderIdentity:
        return self._identity

    def _expand(self, seed_material: bytes) -> np.ndarray:
        needed = self.feature_dimension * 4
        blocks: list[bytes] = []
        counter = 0
        while sum(len(b) for b in blocks) < needed:
            blocks.append(hashlib.sha256(seed_material + counter.to_bytes(4, "big")).digest())
            counter += 1
        raw = b"".join(blocks)[:needed]
        integers = np.frombuffer(raw, dtype=np.uint32).astype(np.float64)
        # Map to a symmetric, unit-ish range so downstream layer norms behave.
        values = (integers / np.float64(2**32)) * 2.0 - 1.0
        return values.astype(np.float32)

    def encode(self, observation: ObservationEnvelope) -> EncodedObservation:
        material = observation.content_digest.encode() + self.variant.encode()
        values = apply_precision(self._expand(material), self.precision)
        from sentinel.wm.versioning import digest_array

        return EncodedObservation(
            encoder_identity=self._identity,
            source_observation_digest=observation.digest,
            features=digest_array(values),
            modality_mask=observation.modality_mask,
            taint=frozenset({Taint.DEVELOPMENT}),
        )

    def encode_array(self, observation: ObservationEnvelope) -> np.ndarray:
        material = observation.content_digest.encode() + self.variant.encode()
        return apply_precision(self._expand(material), self.precision)

    def health_check(self) -> EncoderHealth:
        start = time.perf_counter()
        probe = self._expand(b"health-probe")
        again = self._expand(b"health-probe")
        return EncoderHealth(
            ok=bool(np.array_equal(probe, again)),
            detail="deterministic hash projection; no pretrained weights",
            cold_load_seconds=time.perf_counter() - start,
            feature_dimension=self.feature_dimension,
            deterministic=bool(np.array_equal(probe, again)),
        )


@dataclass
class CachedEncoder:
    """Any adapter, plus the content-addressed cache and its statistics."""

    inner: Any
    cache: LatentCache
    projector_digest: str

    def _identity_for(self, mask: ModalityMask) -> CacheIdentity:
        return CacheIdentity(
            encoder_identity=self.inner.identity,
            projector_digest=self.projector_digest,
            modality_mask=mask,
        )

    @property
    def identity(self) -> EncoderIdentity:
        return self.inner.identity

    @property
    def _packs_bf16(self) -> bool:
        return self.inner.identity.precision is Precision.BF16

    def encode_array(self, observation: ObservationEnvelope) -> np.ndarray:
        identity = self._identity_for(observation.modality_mask)
        cached = self.cache.get(observation.content_digest, identity)
        if cached is not None:
            return unpack_bf16(cached) if self._packs_bf16 else cached
        values = self.inner.encode_array(observation)
        self.cache.put(
            observation.content_digest,
            identity,
            pack_bf16(values) if self._packs_bf16 else values,
            metadata={"packed": "bf16" if self._packs_bf16 else "raw"},
        )
        return values

    def encode(self, observation: ObservationEnvelope) -> EncodedObservation:
        from sentinel.wm.versioning import digest_array

        values = self.encode_array(observation)  # populates the cache on a miss
        return EncodedObservation(
            encoder_identity=self.inner.identity,
            source_observation_digest=observation.digest,
            features=digest_array(values),
            modality_mask=observation.modality_mask,
            taint=frozenset({Taint.DEVELOPMENT}),
        )

    def health_check(self) -> EncoderHealth:
        return self.inner.health_check()
