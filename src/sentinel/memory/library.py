"""The skill library — claim 4, that intelligence accumulates.

Every world whose rules were verified leaves an entry behind: what it
looked like, what it turned out to be, and how strongly the evidence
supported that. A new world retrieves the entries that look most like it
and uses them to decide **what to try first**.

That last phrase is the whole design. The library never asserts an answer;
it reorders the search. A wrong prior costs a few extra verifier replays
and is then overruled by evidence, while a right one finds the rules almost
immediately. This keeps accumulation from turning into accumulated error --
the failure mode that makes memory dangerous rather than useful.

**The metric that matters is cost, not accuracy.** Phase 5 asks whether
environment N+1 gets cheaper as N grows. Search already reaches the right
answer without any library at all, so a library that improved accuracy
would be suspicious. What it should improve is how many hypotheses have to
be tested before the right one is found.

**Measured, it does not.** Over 56 held-out worlds at identical accuracy,
retrieval ordering costs 19.4 verifier replays per world against 10.1 for
simply trying the simplest rule sets first. The library is roughly twice as
expensive as having no memory at all, and it is not the default anywhere.

The reason is structural rather than a tuning failure. This generator draws
a world's mechanics *independently of its layout*, so how a world looks
carries no information about how it behaves and a layout signature cannot
predict rules. Claim 4 is therefore untestable on this generator: there is
nothing to accumulate. Testing it needs environments where appearance and
behaviour genuinely correlate, which is true of real domains and not of
this one -- that is a generator change, and it is the honest prerequisite
before this layer can be said to work or fail.

An earlier version of this module reported a 13x cost reduction. It was
reaching a different member of a scoring tie, and accuracy fell from 58% to
28.6%. See `adapt.search.exhaustive_search` for the ordering-independence
that closed that hole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sentinel.gen.spec import Mechanics

from .signature import Signature


@dataclass(frozen=True, slots=True)
class Entry:
    """One world's verified outcome."""

    world_id: str
    signature: Signature
    classes: tuple[int, ...]
    fitness: float

    def to_json(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "signature": self.signature.to_json(),
            "classes": list(self.classes),
            "fitness": self.fitness,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Entry:
        return cls(
            world_id=str(d["world_id"]),
            signature=Signature.from_json(d["signature"]),
            classes=tuple(int(c) for c in d["classes"]),
            fitness=float(d["fitness"]),
        )


NCLASS = (2, 3, 2, 2, 2, 2)
"""Classes per head, mirroring core.encoding.HEADS."""


@dataclass
class SkillLibrary:
    """Verified rule sets, retrievable by what a world looks like."""

    entries: list[Entry] = field(default_factory=list)
    min_fitness: float = 0.999
    """Only near-perfect explanations are remembered. A half-right rule set
    is worse than no prior at all, because it pulls search toward a
    confident wrong answer."""

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: Entry) -> bool:
        """Record a verified world. Returns whether it was kept."""
        if entry.fitness < self.min_fitness:
            return False
        self.entries = [e for e in self.entries if e.world_id != entry.world_id]
        self.entries.append(entry)
        return True

    def neighbours(self, signature: Signature, k: int = 8) -> list[tuple[float, Entry]]:
        scored = [(signature.distance(e.signature), e) for e in self.entries]
        scored.sort(key=lambda p: p[0])
        return scored[:k]

    def prior(self, signature: Signature, k: int = 8, strength: float = 1.0) -> list[list[float]]:
        """Per-head class probabilities implied by similar past worlds.

        Distance-weighted, and smoothed toward uniform so that a small or
        unrepresentative library cannot make search confident. With no
        entries this returns exactly the uniform prior, which is the
        no-memory condition -- the ablation is therefore free.
        """
        out = [[1.0] * n for n in NCLASS]
        for distance, entry in self.neighbours(signature, k):
            weight = strength / (1.0 + distance)
            for head, cls in enumerate(entry.classes):
                if 0 <= cls < NCLASS[head]:
                    out[head][cls] += weight
        return [[v / sum(head) for v in head] for head in out]

    def rank(self, signature: Signature, candidates: Iterable[tuple[int, ...]],
             k: int = 8) -> list[tuple[int, ...]]:
        """Order hypotheses by how plausible the library finds them.

        Ties broken toward simpler rule sets, matching the tie-break used
        when the evidence itself cannot separate two hypotheses.
        """
        p = self.prior(signature, k)
        def score(c: tuple[int, ...]) -> tuple[float, int]:
            logp = 0.0
            for head, cls in enumerate(c):
                logp += p[head][cls]
            return (-logp, sum(c))
        return sorted(candidates, key=score)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"entries": [e.to_json() for e in self.entries]}, indent=2),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> SkillLibrary:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(entries=[Entry.from_json(e) for e in data.get("entries", [])])

    def summary(self) -> str:
        if not self.entries:
            return "skill library: empty"
        return f"skill library: {len(self.entries)} verified worlds"
