"""X65A-S: evidence identity, separated from observation content.

X65A-0 deduplicated by a caller-supplied `evidence_id`. That is too weak:
a caller can hand the same event two different ids, and two genuinely
independent observations can carry identical content. Identity must come
from the EVENT, not from the string a caller happens to pass or from the
bytes that were observed.

    ExternalEvidenceKey = (source_id, episode_id, external_event_sequence,
                           observation_hash, context_id)

The base evidence id is derived from that key. Consequences, each pinned by
a test and each paired with a planted defect:

  1. the same event under the same key counts once;
  2. the same event under a different caller-supplied memory id counts once;
  3. a deterministic summary of an event adds no factor;
  4. two independent events with EQUAL content and different authenticated
     keys count twice -- content is not identity;
  5. two memory entries pointing at one event never multiply confidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .types import TaintError, encode


@dataclass(frozen=True)
class ExternalEvidenceKey:
    source_id: str
    episode_id: str
    external_event_sequence: int
    observation_hash: str
    context_id: str

    def canon(self):
        return {"source_id": self.source_id, "episode_id": self.episode_id,
                "external_event_sequence": self.external_event_sequence,
                "observation_hash": self.observation_hash,
                "context_id": self.context_id}

    def base_id(self) -> str:
        return "ev:" + hashlib.sha256(encode(self.canon())).hexdigest()[:16]


def observation_hash(payload) -> str:
    return hashlib.sha256(encode(payload)).hexdigest()[:16]


@dataclass
class EvidenceLedger:
    """The authority on what has been absorbed. Memory entries reference
    base ids; they never mint them."""
    absorbed: dict = field(default_factory=dict)     # base_id -> key
    references: dict = field(default_factory=dict)   # base_id -> [entry ids]

    def absorb(self, key: ExternalEvidenceKey) -> tuple:
        bid = key.base_id()
        new = bid not in self.absorbed
        if new:
            self.absorbed[bid] = key
        return bid, new

    def reference(self, base_id: str, entry_id: str) -> None:
        """A memory entry pointing at an event. Any number of these may
        exist; none of them is evidence."""
        if base_id not in self.absorbed:
            raise TaintError(f"entry {entry_id!r} references an event that "
                             f"was never absorbed")
        self.references.setdefault(base_id, []).append(entry_id)

    def contribution_count(self, base_id: str) -> int:
        """However many entries point at it, an event contributes once."""
        return 1 if base_id in self.absorbed else 0

    def canon(self):
        return {"absorbed": {k: v.canon() for k, v in
                             sorted(self.absorbed.items())},
                "references": {k: sorted(v) for k, v in
                               sorted(self.references.items())}}
