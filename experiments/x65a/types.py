"""X65A-0: immutable schemas, taint, and canonical serialization.

Every record is a frozen dataclass with a content-addressed id. Serialization
is canonical -- sorted keys, no insignificant whitespace, rationals as
strings -- so the same logical state always produces the same bytes and the
same hash. Byte counts are measured from those bytes, never from Python
object size.

Taint is enforced at the WRITER boundary rather than trusted from the
caller. ORACLE_ONLY, TARGET_ONLY and FUTURE may never be serialized by the
agent-facing writer; the evaluator has a separate interface.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Any


class Taint(enum.Enum):
    PUBLIC = "PUBLIC"
    OBSERVED = "OBSERVED"
    ORACLE_ONLY = "ORACLE_ONLY"
    TARGET_ONLY = "TARGET_ONLY"
    FUTURE = "FUTURE"


PERSISTABLE = frozenset({Taint.PUBLIC, Taint.OBSERVED})


class TaintError(Exception):
    """Raised at a writer boundary. Never caught to continue a run."""


class MemoryKind(enum.Enum):
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"
    NEGATIVE = "NEGATIVE"


class EdgeKind(enum.Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DERIVES = "DERIVES"
    COMPOSES = "COMPOSES"
    CONTEXT_GATES = "CONTEXT_GATES"
    INVALIDATES = "INVALIDATES"


class OpenWorld(enum.Enum):
    RELEVANT = "RELEVANT"
    NONE = "NONE"
    UNCERTAIN = "UNCERTAIN"
    CONTRADICTED = "CONTRADICTED"
    MISSING_REPRESENTATION = "MISSING_REPRESENTATION"
    UNKNOWN_TASK = "UNKNOWN_TASK"


class Decision(enum.Enum):
    EXECUTE = "EXECUTE"
    ASK = "ASK"
    ABSTAIN = "ABSTAIN"
    EXPAND = "EXPAND"


class Status(enum.Enum):
    """X64H's open-world detector declared UNKNOWN_MEANING on only 0.417 of
    out-of-space tasks, so an unknown observation may not be written as
    settled belief. QUARANTINED is the default for anything the open-world
    check did not positively resolve, and only an explicit confirmation
    event may promote it."""
    QUARANTINED = "QUARANTINED"
    CONFIRMED = "CONFIRMED"
    SUPERSEDED = "SUPERSEDED"
    DEFEATED = "DEFEATED"


# ------------------------------------------------ canonical serialization

def canon(obj: Any) -> Any:
    """Everything reduced to JSON-safe primitives, deterministically."""
    if isinstance(obj, Fraction):
        return {"__frac__": [int(obj.numerator), int(obj.denominator)]}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): canon(v) for k, v in sorted(obj.items(),
                                                    key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [canon(v) for v in obj]
    if isinstance(obj, (frozenset, set)):
        return sorted(canon(v) for v in obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        raise TaintError("floats are not canonically serializable; the exact "
                         "posterior uses Fraction")
    if hasattr(obj, "canon"):
        return canon(obj.canon())
    raise TaintError(f"no canonical form for {type(obj).__name__}")


def uncanon(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {"__frac__"}:
            n, d = obj["__frac__"]
            return Fraction(int(n), int(d))
        return {k: uncanon(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [uncanon(v) for v in obj]
    return obj


def encode(obj: Any) -> bytes:
    return json.dumps(canon(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def decode(b: bytes) -> Any:
    return uncanon(json.loads(b.decode("utf-8")))


def content_id(prefix: str, payload: Any) -> str:
    return f"{prefix}:{hashlib.sha256(encode(payload)).hexdigest()[:32]}"


def byte_cost(obj: Any) -> int:
    """The only admissible byte measurement: canonical serialized length."""
    return len(encode(obj))


# ----------------------------------------------------------- the records

@dataclass(frozen=True)
class ProvenanceRef:
    source_id: str
    observation_hash: str
    task_index: int
    context_id: str
    taint: Taint

    def canon(self):
        return {"source_id": self.source_id,
                "observation_hash": self.observation_hash,
                "task_index": self.task_index, "context_id": self.context_id,
                "taint": self.taint.value}


@dataclass(frozen=True)
class EpisodicEntry:
    """Immutable evidence. An episode is evidence, never truth by
    declaration, and it is the ONLY record type that may introduce a new
    likelihood factor."""
    id: str
    instruction: str
    demonstrations: tuple
    questions: tuple
    answers: tuple
    selected_interpretation: Any
    executed_program: Any
    complete_trace: tuple
    outcome: Any
    counterexamples: tuple
    context_id: str
    provenance: tuple
    acquired_at: int

    def canon(self):
        d = {k: v for k, v in self.__dict__.items()}
        d["provenance"] = [p.canon() for p in self.provenance]
        return d


@dataclass(frozen=True)
class SemanticEntry:
    id: str
    typed_claim: Any
    posterior_table: dict            # Fraction-valued, exact
    calibration_state: Any
    validity_context: str
    provenance: tuple
    created_at: int
    version: int
    supersedes: str | None
    status: Status
    base_evidence: frozenset         # the episodes it derives from
    derived: bool = True             # derived nodes never re-count evidence

    def canon(self):
        return {"id": self.id, "typed_claim": self.typed_claim,
                "posterior_table": self.posterior_table,
                "calibration_state": self.calibration_state,
                "validity_context": self.validity_context,
                "provenance": [p.canon() for p in self.provenance],
                "created_at": self.created_at, "version": self.version,
                "supersedes": self.supersedes, "status": self.status.value,
                "base_evidence": sorted(self.base_evidence),
                "derived": self.derived}


@dataclass(frozen=True)
class ProcedureEntry:
    """A procedure is verified over a DOMAIN, never universally. The
    addendum requires the domain, the interpreter that verified it, the
    probe set, and the continuation-effect summary to be recorded, so that
    `verified` can never be read as `proved`."""
    id: str
    program_ast: Any
    precondition: Any
    postcondition: Any
    continuation_effects: Any
    known_failures: tuple
    resource_contract: Any
    behavioral_signature: str
    confidence: Fraction
    validity_context: str
    provenance: tuple
    created_at: int
    version: int
    supersedes: str | None
    status: Status
    base_evidence: frozenset
    verification_domain: Any
    trusted_interpreter_digest: str
    probe_set_digest: str
    continuation_effect_summary: Any
    effect_summary_version: int
    proof_status: str                # "finite-domain-checked" | "proved"
    derived: bool = True

    def canon(self):
        d = {k: v for k, v in self.__dict__.items()}
        d["provenance"] = [p.canon() for p in self.provenance]
        d["status"] = self.status.value
        d["base_evidence"] = sorted(self.base_evidence)
        return d


@dataclass(frozen=True)
class NegativeEntry:
    id: str
    defeated_id: str
    scoped_context: str
    counterexample: Any
    rejection_reason: str
    source_reliability_state: Any
    provenance: tuple
    created_at: int
    version: int
    supersedes: str | None
    status: Status
    base_evidence: frozenset
    unique: bool = True              # unique counterexamples are never evicted
    derived: bool = True

    def canon(self):
        d = {k: v for k, v in self.__dict__.items()}
        d["provenance"] = [p.canon() for p in self.provenance]
        d["status"] = self.status.value
        d["base_evidence"] = sorted(self.base_evidence)
        return d


@dataclass(frozen=True)
class DependencyEdge:
    source_id: str
    target_id: str
    edge_kind: EdgeKind
    context_predicate: str
    provenance: tuple

    def canon(self):
        return {"source_id": self.source_id, "target_id": self.target_id,
                "edge_kind": self.edge_kind.value,
                "context_predicate": self.context_predicate,
                "provenance": [p.canon() for p in self.provenance]}


FORBIDDEN_KEYS = frozenset({
    "target", "targets", "future", "answer", "answer_key", "gold",
    "z_true", "b_true", "expected_output", "expected_outputs",
    "convention", "phi", "final_seed", "seed_secret", "evaluator_dag",
    "oracle", "next_task", "dependency_truth",
})
