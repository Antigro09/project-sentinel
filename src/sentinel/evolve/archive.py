"""Every version ever tried, and the ability to go back to any of them.

A system that rewrites itself against a score will find holes in that
score. The archive is what makes that survivable: nothing is ever replaced
in place, every generation is retained with what it scored on both the
tuning set and the guard set, and the best-known-good configuration can be
restored at any point.

Keep this even when it seems unnecessary. The whole risk of `evolve/` is
that a change looks like an improvement and is not, and an archive is the
difference between noticing that and having already overwritten the
evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .genome import ScaffoldGenome


@dataclass(frozen=True, slots=True)
class Version:
    """One evaluated configuration."""

    generation: int
    genome: ScaffoldGenome
    train_score: float
    guard_score: float | None
    actions: float
    promoted: bool
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "genome": self.genome.to_json(),
            "train_score": self.train_score,
            "guard_score": self.guard_score,
            "actions": self.actions,
            "promoted": self.promoted,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Version:
        return cls(
            generation=int(d["generation"]),
            genome=ScaffoldGenome.from_json(d["genome"]),
            train_score=float(d["train_score"]),
            guard_score=None if d.get("guard_score") is None else float(d["guard_score"]),
            actions=float(d.get("actions", 0.0)),
            promoted=bool(d.get("promoted", False)),
            note=str(d.get("note", "")),
        )

    def summary(self) -> str:
        guard = "-" if self.guard_score is None else f"{self.guard_score:.3f}"
        mark = "PROMOTED" if self.promoted else "        "
        return (
            f"gen {self.generation:2d} {mark} train={self.train_score:.3f} "
            f"guard={guard} actions={self.actions:5.1f}  {self.genome.summary()}"
            + (f"  [{self.note}]" if self.note else "")
        )


@dataclass
class Archive:
    versions: list[Version] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.versions)

    def record(self, version: Version) -> None:
        self.versions.append(version)

    @property
    def promoted(self) -> list[Version]:
        return [v for v in self.versions if v.promoted]

    def best(self) -> Version | None:
        """Best PROMOTED version, ranked on the guard set.

        Deliberately never ranks on the tuning score: the whole purpose of
        the guard is that improvements there are the ones that were real.
        """
        candidates = self.promoted
        if not candidates:
            return None
        return max(candidates, key=lambda v: (v.guard_score or 0.0, -v.actions))

    def rollback(self) -> ScaffoldGenome:
        """The configuration to actually run. Baseline if nothing beat it."""
        best = self.best()
        return best.genome if best else ScaffoldGenome()

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"versions": [v.to_json() for v in self.versions]}, indent=2),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> Archive:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(versions=[Version.from_json(v) for v in data.get("versions", [])])

    def report(self) -> str:
        return "\n".join(v.summary() for v in self.versions)
