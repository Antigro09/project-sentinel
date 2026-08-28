"""Exact paired resampling for X65A-L1.

The resampling unit is supplied by the caller and is always a complete stream
or latent identity.  Task rows are never pooled as if they were independent.
Bootstrap endpoints are observed rational values, so no float enters the
authoritative artifact.
"""

from __future__ import annotations

import hashlib
import random
from fractions import Fraction


def stable_seed(*parts) -> int:
    b = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big")


def as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    raise TypeError(f"metric must be exact, got {type(value).__name__}")


def mean(values) -> Fraction:
    values = [as_fraction(v) for v in values]
    return sum(values, Fraction(0)) / len(values) if values else Fraction(0)


def paired_interval(a, b, *, reps: int = 5000, seed=20260827) -> dict:
    """Cluster bootstrap of mean(a-b), with exact empirical quantiles."""
    if len(a) != len(b) or not a:
        raise ValueError("paired non-empty vectors of equal length required")
    diffs = [as_fraction(x) - as_fraction(y) for x, y in zip(a, b)]
    rng = random.Random(stable_seed("x65a-l1-bootstrap", seed, len(diffs)))
    draws = []
    for _ in range(reps):
        draws.append(mean(diffs[rng.randrange(len(diffs))]
                          for _j in range(len(diffs))))
    draws.sort()
    lo_i = max(0, (reps * 25) // 1000)
    hi_i = min(reps - 1, (reps * 975 + 999) // 1000 - 1)
    return {
        "lo": draws[lo_i],
        "delta": mean(diffs),
        "hi": draws[hi_i],
        "unit": "complete_stream_or_latent_identity",
        "clusters": len(diffs),
        "resamples": reps,
        "seed": seed,
    }


def interval_includes_zero(interval) -> bool:
    return interval["lo"] <= 0 <= interval["hi"]
