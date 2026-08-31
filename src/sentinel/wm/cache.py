"""Content-addressed cache for frozen-encoder features.

Re-encoding 100,000 observations through a 4B backbone on every run is the
single largest avoidable cost in Scale 0, so the features are cached. That makes
the cache a correctness surface, not just a speed device: a cache that serves
one encoder's features to another run silently invalidates the comparison the
whole matrix exists to make.

The key is therefore the entire identity, not the observation:

    sha256(raw observation)
    + encoder revision and weight digest
    + preprocessing digest
    + precision
    + projector digest
    + modality mask

Two distinct failures are distinguished on read. A **miss** is a key that was
never stored, which is ordinary. A **stale hit** is the dangerous one: the same
observation under the same encoder *name* but a different identity. Returning a
miss there would be safe but silent, and the run would re-encode and carry on
without anyone learning that the identity moved. It raises instead.

Payload bytes and index/metadata bytes are reported separately because the
matrix budgets them separately, and because the 1.024 GB estimate for a million
512-wide fp16 latents covers only the first of the two.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sentinel.wm.latent_contract import (
    ContractViolation,
    EncoderIdentity,
    ModalityMask,
)
from sentinel.wm.versioning import digest_of, digest_of_bytes, require_digest


class StaleCacheError(RuntimeError):
    """The stored entry for this observation was produced by a different identity."""


class CacheCorruptionError(RuntimeError):
    """Payload bytes do not match the digest recorded when they were written."""


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """Everything other than the observation that determines a feature vector."""

    encoder_identity: EncoderIdentity
    projector_digest: str
    modality_mask: ModalityMask

    def __post_init__(self) -> None:
        require_digest(self.projector_digest, "CacheIdentity.projector_digest")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "encoder_revision": self.encoder_identity.revision,
            "encoder_weight_digest": self.encoder_identity.weight_digest,
            "preprocessing_digest": self.encoder_identity.preprocessing_digest,
            "precision": self.encoder_identity.precision.value,
            "projector_digest": self.projector_digest,
            "modality_mask": self.modality_mask.canonical_dict(),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())

    @property
    def logical_scope(self) -> str:
        """Identity of the encoder *name*, excluding everything that may drift.

        Two entries sharing a logical scope but not a full identity are the
        stale case: same nominal encoder, different actual features.
        """
        return digest_of(
            {
                "provider": self.encoder_identity.provider,
                "model_name": self.encoder_identity.model_name,
            }
        )


def latent_cache_key(observation_digest: str, identity: CacheIdentity) -> str:
    """The full content-addressed key."""
    require_digest(observation_digest, "observation_digest")
    return digest_of({"observation": observation_digest, "identity": identity.canonical_dict()})


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    stale_rejections: int = 0

    @property
    def hit_ratio(self) -> float:
        looked_up = self.hits + self.misses
        return self.hits / looked_up if looked_up else 0.0

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "stale_rejections": self.stale_rejections,
            "hit_ratio": self.hit_ratio,
        }


@dataclass
class LatentCache:
    """On-disk content-addressed feature store.

    The index is a single JSON document rewritten atomically. That is slower
    than a database at a million entries and entirely adequate here, and it has
    the property that matters for an audit: the index is readable without the
    code that wrote it.
    """

    root: Path
    stats: CacheStats = field(default_factory=CacheStats)
    _index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _scope_map: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "payload").mkdir(parents=True, exist_ok=True)
        self._load_index()

    # ---- index ---------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def _load_index(self) -> None:
        if self.index_path.exists():
            document = json.loads(self.index_path.read_text())
            self._index = document.get("entries", {})
            self._scope_map = document.get("scopes", {})
        else:
            self._index = {}
            self._scope_map = {}

    def flush(self) -> None:
        document = {"version": 1, "entries": self._index, "scopes": self._scope_map}
        text = json.dumps(document, sort_keys=True, separators=(",", ":"))
        handle = tempfile.NamedTemporaryFile(
            "w", dir=self.root, delete=False, prefix=".index-", suffix=".tmp"
        )
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, self.index_path)

    # ---- payload paths -------------------------------------------------

    def _payload_path(self, key: str) -> Path:
        body = key.split(":", 1)[1]
        return self.root / "payload" / body[:2] / f"{body}.npy"

    # ---- read / write --------------------------------------------------

    def _scope_key(self, observation_digest: str, identity: CacheIdentity) -> str:
        return digest_of({"observation": observation_digest, "scope": identity.logical_scope})

    def get(self, observation_digest: str, identity: CacheIdentity) -> np.ndarray | None:
        """Return cached features, or None on a genuine miss.

        Raises `StaleCacheError` when this observation was cached under the same
        encoder name but a different identity, and `CacheCorruptionError` when
        the payload no longer hashes to what the index recorded.
        """
        key = latent_cache_key(observation_digest, identity)
        scope_key = self._scope_key(observation_digest, identity)
        stored_key = self._scope_map.get(scope_key)
        if stored_key is not None and stored_key != key:
            self.stats.stale_rejections += 1
            stored = self._index.get(stored_key, {})
            raise StaleCacheError(
                f"observation {observation_digest[:16]}... was cached under "
                f"{identity.encoder_identity.provider}/{identity.encoder_identity.model_name} "
                f"with identity {stored.get('identity_digest', '?')[:16]}..., "
                f"but is now requested with identity {identity.digest[:16]}...; "
                "the cache will not serve features across an identity change"
            )
        entry = self._index.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        path = self._payload_path(key)
        if not path.exists():
            self.stats.misses += 1
            return None
        payload = path.read_bytes()
        if digest_of_bytes(payload) != entry["payload_digest"]:
            raise CacheCorruptionError(
                f"payload for {key[:24]}... does not match its recorded digest"
            )
        self.stats.hits += 1
        with open(path, "rb") as handle:
            return np.load(handle, allow_pickle=False)

    def put(
        self,
        observation_digest: str,
        identity: CacheIdentity,
        features: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        key = latent_cache_key(observation_digest, identity)
        scope_key = self._scope_key(observation_digest, identity)
        existing = self._scope_map.get(scope_key)
        if existing is not None and existing != key:
            self.stats.stale_rejections += 1
            raise StaleCacheError(
                f"refusing to overwrite features for {observation_digest[:16]}... "
                "with a different encoder identity; build a new cache root instead"
            )
        path = self._payload_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, delete=False, prefix=".payload-", suffix=".tmp"
        )
        try:
            np.save(buffer, np.ascontiguousarray(features), allow_pickle=False)
            buffer.flush()
            os.fsync(buffer.fileno())
        finally:
            buffer.close()
        os.replace(buffer.name, path)
        payload_digest = digest_of_bytes(path.read_bytes())
        self._index[key] = {
            "identity_digest": identity.digest,
            "observation_digest": observation_digest,
            "payload_digest": payload_digest,
            "dtype": str(features.dtype),
            "shape": list(features.shape),
            "bytes": int(path.stat().st_size),
            "metadata": dict(metadata or {}),
        }
        self._scope_map[scope_key] = key
        self.stats.writes += 1
        return key

    # ---- reporting -----------------------------------------------------

    def size_report(self) -> dict[str, Any]:
        """Payload bytes and index/metadata bytes, measured, never estimated."""
        payload_bytes = sum(int(e["bytes"]) for e in self._index.values())
        resident = 0
        payload_root = self.root / "payload"
        if payload_root.exists():
            for path in payload_root.rglob("*.npy"):
                resident += path.stat().st_size
        index_bytes = self.index_path.stat().st_size if self.index_path.exists() else 0
        metadata_bytes = len(
            json.dumps(
                {k: v.get("metadata", {}) for k, v in self._index.items()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return {
            "entries": len(self._index),
            "payload_bytes_indexed": payload_bytes,
            "payload_bytes_resident": resident,
            "index_bytes": index_bytes,
            "metadata_bytes": metadata_bytes,
            "total_bytes": resident + index_bytes,
        }
