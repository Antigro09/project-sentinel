"""Verification results.

Three numbers, never collapsed into one:

- **accuracy**  — of the cells the model chose to predict, how many were right
- **coverage**  — how much of the grid it was willing to predict at all
- **outcome**   — did it correctly call level_complete / game_over

Collapsing these destroys the reward signal in both directions. A model
that abstains everywhere scores accuracy 1.0 at coverage 0.0 and is
worthless; a model that predicts the whole grid at 95% accuracy may be
far more useful than one predicting 10% of it perfectly. Which is better
depends on what the planner needs, so the verifier reports and the caller
decides.

`first_divergence` is the most actionable field here. It is where the
model's story about the world first stopped matching reality, which is
exactly the evidence a repair step needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentinel.env.types import Action
from sentinel.wm.contract import Outcome


@dataclass(frozen=True, slots=True)
class CellStats:
    """Per-frame cell-level tally."""

    total: int
    predicted: int
    correct: int

    @property
    def accuracy(self) -> float:
        """Of predicted cells, fraction correct. Abstentions never count against."""
        return 1.0 if self.predicted == 0 else self.correct / self.predicted

    @property
    def coverage(self) -> float:
        return 0.0 if self.total == 0 else self.predicted / self.total

    def __add__(self, other: CellStats) -> CellStats:
        return CellStats(
            total=self.total + other.total,
            predicted=self.predicted + other.predicted,
            correct=self.correct + other.correct,
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    """How the model did on one recorded step."""

    index: int
    action: Action | None
    cells: CellStats
    frame_match: bool
    outcome_predicted: Outcome | None
    outcome_actual: Outcome
    scored: bool
    """False at discontinuities (resets, level boundaries) where render is not scored."""
    boundary: bool = False
    error: str | None = None

    @property
    def outcome_correct(self) -> bool:
        return self.outcome_predicted == self.outcome_actual

    def mismatch_summary(self) -> str:
        if self.error:
            return f"step {self.index}: model raised — {self.error}"
        wrong = self.cells.predicted - self.cells.correct
        return (
            f"step {self.index} ({self.action}): {wrong} of "
            f"{self.cells.predicted} predicted cells wrong "
            f"(coverage {self.cells.coverage:.0%})"
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Full result of replaying one history through one model."""

    model_name: str
    game_id: str
    seed: int
    steps: tuple[StepResult, ...]
    crashed: bool = False
    crash_detail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scored_steps(self) -> tuple[StepResult, ...]:
        return tuple(s for s in self.steps if s.scored)

    @property
    def transition_match(self) -> float:
        """Fraction of scored steps where every predicted cell was right.

        The strict number. A model can have high cell accuracy and low
        transition match by being wrong about one cell every frame — which
        is precisely the kind of error that compounds when a planner
        simulates forward, so it is worth measuring separately.
        """
        scored = self.scored_steps
        return 0.0 if not scored else sum(s.frame_match for s in scored) / len(scored)

    @property
    def cells(self) -> CellStats:
        total = CellStats(0, 0, 0)
        for s in self.scored_steps:
            total = total + s.cells
        return total

    @property
    def accuracy(self) -> float:
        return self.cells.accuracy

    @property
    def coverage(self) -> float:
        return self.cells.coverage

    @property
    def outcome_accuracy(self) -> float:
        judged = [s for s in self.steps if s.outcome_predicted is not None]
        return 0.0 if not judged else sum(s.outcome_correct for s in judged) / len(judged)

    @property
    def first_divergence(self) -> int | None:
        """Index of the first step the model got wrong, in any channel.

        The repair signal: everything before this point is explained, so
        this is where the model's story about the world breaks.

        All three channels count. A model can render every frame perfectly
        and still be wrong by never noticing a level ended — and a repair
        step that was only told about render errors would have nothing to
        work with. Outcome errors are checked even at boundaries, where
        render deliberately is not: calling the boundary is precisely the
        model's job there.
        """
        for s in self.steps:
            if s.error is not None:
                return s.index
            if s.scored and not s.frame_match:
                return s.index
            if s.outcome_predicted is not None and not s.outcome_correct:
                return s.index
        return None

    @property
    def explained_prefix(self) -> int:
        """How many steps the model explained before diverging."""
        d = self.first_divergence
        return len(self.steps) if d is None else d

    @property
    def fitness(self) -> float:
        """Single number for *ranking* candidates. Lossy by construction.

        Exists because the proposer's escalation ladder and the planner's
        model selection need a total order, and because a subtlety bites
        otherwise: `transition_match` is vacuously 1.0 when a model
        predicts nothing, since zero predicted cells are all trivially
        correct. Multiplying accuracy by coverage removes that free ride —
        an abstaining model scores 0 here, as it should.

        Never report this in place of the three components. It answers
        "which of these is better", not "how good is this".
        """
        if self.crashed:
            return 0.0
        return self.accuracy * self.coverage * (0.5 + 0.5 * self.outcome_accuracy)

    @property
    def is_perfect(self) -> bool:
        """Complete and exactly right — no divergence, no abstentions, no crash."""
        return (
            not self.crashed
            and self.first_divergence is None
            and self.coverage == 1.0
            and self.outcome_accuracy == 1.0
        )

    def divergences(self, limit: int = 5) -> list[StepResult]:
        out = [s for s in self.steps if s.error or (s.scored and not s.frame_match)]
        return out[:limit]

    def summary(self) -> str:
        if self.crashed:
            return (
                f"{self.model_name} on {self.game_id}: CRASHED at step "
                f"{self.first_divergence} — {self.crash_detail}"
            )
        d = self.first_divergence
        return (
            f"{self.model_name} on {self.game_id}: "
            f"match={self.transition_match:.1%} "
            f"acc={self.accuracy:.1%} "
            f"cov={self.coverage:.1%} "
            f"outcome={self.outcome_accuracy:.1%} "
            f"scored={len(self.scored_steps)}/{len(self.steps)} "
            + ("no divergence" if d is None else f"first divergence @{d}")
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "game_id": self.game_id,
            "seed": self.seed,
            "transition_match": self.transition_match,
            "fitness": self.fitness,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "outcome_accuracy": self.outcome_accuracy,
            "first_divergence": self.first_divergence,
            "explained_prefix": self.explained_prefix,
            "scored_steps": len(self.scored_steps),
            "total_steps": len(self.steps),
            "crashed": self.crashed,
            "crash_detail": self.crash_detail,
            "extra": dict(self.extra),
        }
