"""X65A-S: per-identity convention-semantic memory.

WHAT IS STORED, and why it is exact.

X64H calibration grounds a meaning without language and then observes a
three-role utterance, so its likelihood is an INDICATOR:

    p(u | phi, z) = [ u3[phi, z] == u ]

The posterior after any number of grounded observations is therefore exactly
uniform over the surviving convention set, and the set of grounded pairs
(z, u) is an exact sufficient statistic for it. Better: any pair whose
removal leaves the surviving set unchanged is redundant, so `minimize`
prunes to a minimal sufficient subset with the posterior -- and the
posterior predictive -- provably unchanged. In X64H's families three
grounded pairs already separate the family, so a record SATURATES at three
or four pairs no matter how often an identity returns. That is the whole
reason active memory can scale with identity count rather than with episode
count.

Only grounded calibration evidence enters a persistent record. Transfer
observations have non-indicator likelihoods, do not admit this exact
compression, and are the thing being measured -- so they are used within the
current task and never written. This is a restriction, and it is the reason
the exactness claim here is a claim and not a hope.

NOT STORED: the sampled convention object, future meanings, future outputs,
complete future programs, evaluator dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import Status, TaintError, byte_cost, canon


@dataclass(frozen=True)
class GroundedObservation:
    z: int                  # form index -- a MEANING already observed
    u: int                  # the utterance code observed with it
    base_evidence: str      # ExternalEvidenceKey.base_id()

    def canon(self):
        return {"z": self.z, "u": self.u, "e": self.base_evidence}


@dataclass
class SemanticRecord:
    identity: str
    grounded: tuple = ()
    n_confirming: int = 0
    n_contradicting: int = 0
    context_id: str = "ctx0"
    status: Status = Status.QUARANTINED
    last_verified_use: int = -1
    created_at: int = 0
    version: int = 1
    supersedes: str | None = None
    entropy_bits: float | None = None
    surviving: int | None = None

    def canon(self):
        return {"identity": self.identity,
                "grounded": [g.canon() for g in self.grounded],
                "n_confirming": self.n_confirming,
                "n_contradicting": self.n_contradicting,
                "context_id": self.context_id, "status": self.status.value,
                "last_verified_use": self.last_verified_use,
                "created_at": self.created_at, "version": self.version,
                "supersedes": self.supersedes,
                "surviving": self.surviving}

    def bytes(self) -> int:
        return byte_cost(self.canon())


def surviving_mask(fam, grounded) -> np.ndarray:
    m = np.ones(fam.n, dtype=bool)
    for g in grounded:
        m &= (fam.u3[:, g.z] == g.u)
    return m


def prior_from(fam, grounded) -> np.ndarray:
    m = surviving_mask(fam, grounded)
    k = int(m.sum())
    if k == 0:
        raise TaintError("grounded observations are mutually inconsistent; "
                         "this is an open-world signal, not a prior")
    return m.astype(np.float64) / k


def entropy_of(fam, grounded) -> float:
    return math.log2(max(1, int(surviving_mask(fam, grounded).sum())))


def minimize(fam, grounded) -> tuple:
    """Drop every observation whose removal leaves the surviving set
    unchanged. Exact: the posterior is uniform over that set, so an
    observation that does not change it does not change the posterior."""
    keep = list(grounded)
    target = surviving_mask(fam, keep)
    i = 0
    while i < len(keep):
        trial = keep[:i] + keep[i + 1:]
        if np.array_equal(surviving_mask(fam, trial), target):
            keep = trial
        else:
            i += 1
    return tuple(keep)


def absorb(fam, record: SemanticRecord, obs: GroundedObservation,
           task_index: int, ledger, quarantine: bool = True
           ) -> tuple[SemanticRecord, str]:
    """Fold one grounded observation into a record.

    An observation that contradicts everything already stored does NOT
    overwrite the record: it is counted and the record is quarantined for a
    later phase to revise. Revision is out of scope for X65A-S and pretending
    otherwise here would be a silent capability claim."""
    if any(g.base_evidence == obs.base_evidence for g in record.grounded):
        return record, "duplicate_event"           # counted once, ever
    trial = record.grounded + (obs,)
    if int(surviving_mask(fam, trial).sum()) == 0:
        if quarantine:
            return (SemanticRecord(
                record.identity, record.grounded, record.n_confirming,
                record.n_contradicting + 1, record.context_id,
                Status.QUARANTINED, record.last_verified_use,
                record.created_at, record.version + 1, record.supersedes,
                record.entropy_bits, record.surviving), "contradiction_quarantined")
        trial = (obs,)                             # the no-quarantine arm
    kept = minimize(fam, trial)
    k = int(surviving_mask(fam, kept).sum())
    return (SemanticRecord(record.identity, kept, record.n_confirming + 1,
                           record.n_contradicting, record.context_id,
                           Status.CONFIRMED, task_index, record.created_at,
                           record.version + 1, record.supersedes,
                           math.log2(max(1, k)), k), "absorbed")


@dataclass
class SemanticStore:
    """Keyed by opaque identity. Lookup by key is the ONLY retrieval this
    phase grants; there is no scoring, no frontier and no selection."""
    records: dict = field(default_factory=dict)
    budget_bytes: int = 4096

    def get(self, identity: str) -> SemanticRecord | None:
        return self.records.get(identity)

    def put(self, rec: SemanticRecord) -> None:
        self.records[rec.identity] = rec

    def bytes(self) -> int:
        return byte_cost([canon(r) for _k, r in sorted(self.records.items())])

    def over_budget(self) -> bool:
        return self.bytes() > self.budget_bytes

    def canon(self):
        return [canon(r) for _k, r in sorted(self.records.items())]
