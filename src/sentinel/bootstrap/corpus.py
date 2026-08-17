"""Corpus storage.

One JSONL record per world: the spec that generated it, the best program
the teacher produced, its verification scores, and every attempt along the
way. JSONL because a corpus build is a long unattended job that can die at
hour four, and a line-oriented format means what completed is still valid.

Failed inductions are stored too, marked. The record of *how* the teacher
failed is training signal in its own right, and discarding it would leave
the corpus quietly biased toward whatever happens to be easy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from sentinel.gen.spec import WorldSpec

from .teacher import InductionResult


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """One (environment, program, score) triple."""

    world_id: str
    split: str
    """train | holdout_seed | holdout_mechanics"""
    spec: WorldSpec
    source: str | None
    fitness: float
    solved: bool
    transition_match: float
    coverage: float
    outcome_accuracy: float
    rounds: int
    prompt_tokens: int
    output_tokens: int
    seconds: float
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.source is not None and self.fitness > 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "split": self.split,
            "spec": self.spec.to_json(),
            "source": self.source,
            "fitness": self.fitness,
            "solved": self.solved,
            "transition_match": self.transition_match,
            "coverage": self.coverage,
            "outcome_accuracy": self.outcome_accuracy,
            "rounds": self.rounds,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "seconds": self.seconds,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CorpusRecord:
        return cls(
            world_id=str(d["world_id"]),
            split=str(d["split"]),
            spec=WorldSpec.from_json(d["spec"]),
            source=d.get("source"),
            fitness=float(d.get("fitness", 0.0)),
            solved=bool(d.get("solved", False)),
            transition_match=float(d.get("transition_match", 0.0)),
            coverage=float(d.get("coverage", 0.0)),
            outcome_accuracy=float(d.get("outcome_accuracy", 0.0)),
            rounds=int(d.get("rounds", 0)),
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            seconds=float(d.get("seconds", 0.0)),
            error=d.get("error"),
        )

    @classmethod
    def from_result(
        cls, spec: WorldSpec, split: str, result: InductionResult
    ) -> CorpusRecord:
        report = result.best_report
        return cls(
            world_id=spec.world_id,
            split=split,
            spec=spec,
            source=result.best_source,
            fitness=result.best_fitness,
            solved=result.solved,
            transition_match=report.transition_match if report else 0.0,
            coverage=report.coverage if report else 0.0,
            outcome_accuracy=report.outcome_accuracy if report else 0.0,
            rounds=len(result.attempts),
            prompt_tokens=sum(a.prompt_tokens for a in result.attempts),
            output_tokens=sum(a.output_tokens for a in result.attempts),
            seconds=sum(a.seconds for a in result.attempts),
            error=result.error,
        )


class CorpusWriter:
    """Append-only JSONL writer that flushes every record.

    Flushing per record costs nothing at these rates and means a build
    killed at hour four keeps four hours of work.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self.written = 0

    def append(self, record: CorpusRecord) -> None:
        self._handle.write(json.dumps(record.to_json()) + "\n")
        self._handle.flush()
        self.written += 1

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> CorpusWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_corpus(path: str | Path) -> list[CorpusRecord]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[CorpusRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(CorpusRecord.from_json(json.loads(line)))
        except Exception:  # noqa: BLE001 - a torn final line must not lose the rest
            continue
    return out


def completed_ids(path: str | Path) -> set[str]:
    """World ids already processed, so a build can resume where it stopped."""
    return {r.world_id for r in read_corpus(path)}


def iter_usable(path: str | Path) -> Iterator[CorpusRecord]:
    for record in read_corpus(path):
        if record.usable:
            yield record


def corpus_stats(records: list[CorpusRecord]) -> str:
    if not records:
        return "empty corpus"

    total = len(records)
    usable = sum(1 for r in records if r.usable)
    solved = sum(1 for r in records if r.solved)
    tokens = sum(r.prompt_tokens + r.output_tokens for r in records)
    seconds = sum(r.seconds for r in records)
    mean_fitness = sum(r.fitness for r in records) / total

    by_split: dict[str, int] = {}
    for r in records:
        by_split[r.split] = by_split.get(r.split, 0) + 1

    splits = " ".join(f"{k}={v}" for k, v in sorted(by_split.items()))
    return (
        f"{total} worlds | {usable} usable ({usable / total:.0%}) | "
        f"{solved} solved exactly ({solved / total:.0%}) | "
        f"mean fitness {mean_fitness:.3f} | {splits} | "
        f"{tokens:,} tokens | {seconds / 60:.1f} min"
    )
