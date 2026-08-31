"""Canonical serialisation, content hashing, and component identity digests.

Every claim Phase 2 will eventually make -- "this model, trained on this data,
using these frozen features, scored this" -- is only as good as the identity of
its parts. So identity is computed here, once, from an explicit field list, and
every consumer hashes through these functions rather than inventing its own.

Two rules make the digests trustworthy.

**Canonical form before hashing.** Dicts are key-sorted, floats are emitted in
Python's shortest round-trip form, NaN and Infinity are rejected outright, and
the separators are fixed. Two processes that agree on the value agree on the
bytes, so a digest computed today matches one computed after a restart.

**No payload without a digest.** An array reaching a typed record arrives as an
`ArrayDigest`, never as a bare buffer. That makes it impossible to cache, split,
or train on data whose provenance was never recorded -- the failure mode that
turns a leaked test frame into an unexplained capability gain.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DIGEST_PREFIX = "sha256:"
"""Every digest string in Phase 2 carries its algorithm. Bare hex is rejected."""


class CanonicalisationError(ValueError):
    """A value cannot be canonically serialised, so it cannot be hashed.

    Raised rather than coerced. A silent coercion here would produce two
    different objects with the same digest, which is the one failure this
    module exists to prevent.
    """


def _canonical(value: Any) -> Any:
    """Reduce a value to JSON-canonical form, or refuse."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise CanonicalisationError(f"non-finite float {value!r} cannot be hashed")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalisationError(f"map key {key!r} is {type(key).__name__}, not str")
            out[key] = _canonical(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        try:
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
        except TypeError as exc:  # pragma: no cover - defensive
            raise CanonicalisationError(f"unsortable set contents: {exc}") from exc
    if hasattr(value, "canonical_dict"):
        return _canonical(value.canonical_dict())
    raise CanonicalisationError(f"{type(value).__name__} has no canonical form")


def canonical_json(value: Any) -> str:
    """Deterministic JSON text for any canonicalisable value."""
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def digest_of(value: Any) -> str:
    """Content hash of any canonicalisable value."""
    return DIGEST_PREFIX + hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_of_bytes(payload: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


def is_digest(text: object) -> bool:
    """True for a well-formed `sha256:<64 hex>` string."""
    if not isinstance(text, str) or not text.startswith(DIGEST_PREFIX):
        return False
    body = text[len(DIGEST_PREFIX) :]
    return len(body) == 64 and all(c in "0123456789abcdef" for c in body)


def require_digest(text: object, field: str) -> str:
    if not is_digest(text):
        raise CanonicalisationError(f"{field} must be a {DIGEST_PREFIX}<hex64> digest, got {text!r}")
    return str(text)


@dataclass(frozen=True, slots=True)
class ArrayDigest:
    """A reference to array data that carries its own identity.

    Numeric payloads never travel through Phase-2 records as bare buffers. The
    dtype and shape are part of the digest because a reinterpretation of the
    same bytes is a different observation, not the same one.
    """

    dtype: str
    shape: tuple[int, ...]
    digest: str

    def __post_init__(self) -> None:
        if not self.dtype:
            raise CanonicalisationError("ArrayDigest.dtype is empty")
        if any(int(d) < 0 for d in self.shape):
            raise CanonicalisationError(f"ArrayDigest.shape has a negative axis: {self.shape}")
        require_digest(self.digest, "ArrayDigest.digest")

    @property
    def size(self) -> int:
        total = 1
        for axis in self.shape:
            total *= int(axis)
        return total

    def canonical_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape), "digest": self.digest}


def digest_array(array: Any) -> ArrayDigest:
    """Hash a numpy-like array over dtype, shape, and C-contiguous bytes."""
    import numpy as np

    materialised = np.ascontiguousarray(np.asarray(array))
    hasher = hashlib.sha256()
    hasher.update(str(materialised.dtype).encode())
    hasher.update(canonical_bytes(list(materialised.shape)))
    hasher.update(materialised.tobytes())
    return ArrayDigest(
        dtype=str(materialised.dtype),
        shape=tuple(int(d) for d in materialised.shape),
        digest=DIGEST_PREFIX + hasher.hexdigest(),
    )


def digest_file(path: Any, chunk: int = 1 << 20) -> str:
    """Streaming digest of a file, for weights and lockfiles too large to load."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return DIGEST_PREFIX + hasher.hexdigest()


def digest_files(paths: Sequence[Any]) -> str:
    """Order-independent digest over a set of files, name and content included."""
    entries = sorted((str(p), digest_file(p)) for p in paths)
    return digest_of({"files": [{"name": n, "digest": d} for n, d in entries]})
