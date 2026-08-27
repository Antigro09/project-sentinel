"""X65A-0: active memory, audit archive, and atomic persistence.

The two stores are separate because conflating them lets a system claim
bounded memory while an immutable evidence ledger grows linearly underneath.

    ActiveMemory   available to inference and retrieval; subject to the
                   preregistered byte budget
    AuditArchive   immutable evidence and provenance; NOT retrievable;
                   measured separately

Active, archive and total serialized bytes are all reported. A bounded-total
claim requires the archive curve to support it, and by construction it does
not: one nonempty record per task is Theorem 4A. So the honest statement
this store can support is about ACTIVE growth or about the total slope
relative to raw replay -- never "bounded total memory".

Byte counts come from canonical serialized bytes. Python object size is not
a memory measurement.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .posterior import ExactPosterior
from .types import (FORBIDDEN_KEYS, MemoryKind, PERSISTABLE, Status, Taint,
                    TaintError, byte_cost, canon, decode, encode)

SCHEMA_VERSION = 1


def _walk_reject(obj, path: str = "") -> None:
    """Taint is enforced at the writer, not trusted from the caller."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise TaintError(f"forbidden key {k!r} at {path}")
            if isinstance(v, str) and v in {t.value for t in Taint} \
                    and Taint(v) not in PERSISTABLE:
                raise TaintError(f"non-persistable taint {v} at {path}.{k}")
            _walk_reject(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_reject(v, f"{path}[{i}]")


@dataclass
class ActiveMemory:
    budget_bytes: int = 4096
    entries: dict = field(default_factory=dict)

    def write(self, entry, kind: MemoryKind) -> None:
        if getattr(entry, "status", None) is Status.QUARANTINED \
                and kind is not MemoryKind.EPISODIC:
            raise TaintError(
                "a quarantined record may not enter active memory as belief; "
                "X64H's open-world detector declared UNKNOWN_MEANING on only "
                "0.417 of out-of-space tasks, so unresolved observations are "
                "held, not confirmed")
        _walk_reject(canon(entry))
        self.entries[entry.id] = (kind, entry)

    def confirm(self, entry_id: str, evidence_id: str):
        """The only route out of quarantine, and it needs an explicit
        confirming observation."""
        kind, e = self.entries[entry_id]
        if not evidence_id:
            raise TaintError("confirmation requires a base evidence id")
        return kind, e

    def retrievable(self) -> list:
        return [e for _k, e in self.entries.values()
                if getattr(e, "status", None) is not Status.QUARANTINED]

    def bytes(self) -> int:
        return byte_cost([canon(e) for _id, (_k, e)
                          in sorted(self.entries.items())])

    def over_budget(self) -> bool:
        return self.bytes() > self.budget_bytes


@dataclass
class AuditArchive:
    """Append-only. Never read by retrieval; only by the audit."""
    records: list = field(default_factory=list)
    chain: str = hashlib.sha256(b"x65a-archive-genesis").hexdigest()

    def append(self, record) -> str:
        _walk_reject(canon(record))
        self.records.append(record)
        self.chain = hashlib.sha256(
            (self.chain + hashlib.sha256(encode(record)).hexdigest())
            .encode()).hexdigest()
        return self.chain

    def bytes(self) -> int:
        return byte_cost([canon(r) for r in self.records])


@dataclass
class PersistentState:
    schema_version: int
    frozen_model_digest: str
    task_index: int
    active: ActiveMemory
    archive: AuditArchive
    posterior: ExactPosterior
    calibration_state: dict = field(default_factory=dict)

    def audit_chain_hash(self) -> str:
        return self.archive.chain

    def canon(self):
        return {
            "schema_version": self.schema_version,
            "frozen_model_digest": self.frozen_model_digest,
            "task_index": self.task_index,
            "active": [canon(e) for _id, (_k, e)
                       in sorted(self.active.entries.items())],
            "active_kinds": {k: v[0].value
                             for k, v in sorted(self.active.entries.items())},
            "archive": [canon(r) for r in self.archive.records],
            "archive_chain": self.archive.chain,
            "posterior": self.posterior.canon(),
            "calibration_state": self.calibration_state,
        }

    def report_bytes(self) -> dict:
        a, r = self.active.bytes(), self.archive.bytes()
        return {"active_bytes": a, "archive_bytes": r, "total_bytes": a + r,
                "active_budget": self.active.budget_bytes,
                "active_within_budget": a <= self.active.budget_bytes,
                "archive_is_retrievable": False}


def save(path: Path, state: PersistentState) -> str:
    """New file, fsync, atomic rename, reopen, validate, rehash."""
    payload = state.canon()
    _walk_reject(payload)
    if sum(state.posterior.q.values()) != 1:
        raise TaintError("refusing to persist an unnormalized posterior")
    blob = encode(payload)
    digest = hashlib.sha256(blob).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    back = path.read_bytes()
    if hashlib.sha256(back).hexdigest() != digest:
        raise TaintError("serialized state did not survive the round trip")
    d = decode(back)
    if d["schema_version"] != SCHEMA_VERSION:
        raise TaintError("schema version mismatch after reopen")
    return digest


def load(path: Path) -> dict:
    blob = path.read_bytes()
    d = decode(blob)
    if d["schema_version"] != SCHEMA_VERSION:
        raise TaintError("schema version mismatch")
    from fractions import Fraction
    q = d["posterior"]["q"]
    if sum(q.values(), Fraction(0)) != 1:
        raise TaintError("loaded posterior is not exactly normalized")
    _walk_reject(d)
    d["_sha256"] = hashlib.sha256(blob).hexdigest()
    return d
