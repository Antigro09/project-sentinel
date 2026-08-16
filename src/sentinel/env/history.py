"""Typed interaction history.

This is the evidence store. Every claim a world model makes is checked
against it, so it has to preserve *structure*, not just a flat list of
grids: which frames were settled decisions, which were animation, where
levels began and ended, and where resets cut the causal chain.

Tycho found this separation load-bearing. A history that flattens frames
lets a correct model look wrong and a wrong model look correct.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator

from .types import Action, FrameKind, Observation, Step


@dataclass
class History:
    """Ordered record of one episode's interaction.

    Levels are tracked explicitly because a world model that generalizes
    must be validated against levels it was not built from. `reset_points`
    marks indices where causal continuity breaks — a transition spanning a
    reset is not a real transition and must never be scored as one.
    """

    game_id: str
    seed: int
    initial: Observation
    steps: list[Step] = field(default_factory=list)
    reset_points: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    @property
    def current_level(self) -> int:
        return self.steps[-1].level_index if self.steps else 0

    @property
    def last(self) -> Observation:
        return self.steps[-1].settled if self.steps else self.initial

    def append(self, step: Step) -> None:
        if step.settled.full_reset:
            self.reset_points.append(step.index)
        self.steps.append(step)

    def observation_before(self, index: int) -> Observation:
        """The settled frame the agent saw before taking step `index`."""
        return self.initial if index == 0 else self.steps[index - 1].settled

    def transition_pairs(
        self, level: int | None = None
    ) -> list[tuple[Observation, Action, Observation]]:
        """(before, action, after) triples that a world model must reproduce.

        Pairs spanning a reset are excluded: the engine discontinuously
        rewrote state there, so no transition function should be expected
        to explain them. Including them would penalise correct models.
        """
        out: list[tuple[Observation, Action, Observation]] = []
        resets = set(self.reset_points)
        for step in self.steps:
            if step.index in resets:
                continue
            if level is not None and step.level_index != level:
                continue
            out.append((self.observation_before(step.index), step.action, step.settled))
        return out

    def decision_frames(self) -> list[Observation]:
        return [self.initial, *(s.settled for s in self.steps)]

    def levels_seen(self) -> list[int]:
        return sorted({s.level_index for s in self.steps})

    def steps_for_level(self, level: int) -> list[Step]:
        return [s for s in self.steps if s.level_index == level]

    def action_sequence(self) -> list[Action]:
        return [s.action for s in self.steps]

    def digest(self) -> str:
        """Content hash of the entire episode.

        Two runs that produce the same digest are the same episode. This is
        the Phase 0 determinism check and, later, the cheapest possible
        regression test on the engine itself.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(f"{self.game_id}|{self.seed}|".encode())
        h.update(self.initial.digest().encode())
        for step in self.steps:
            h.update(f"|{step.index}|{step.action}|{step.level_index}|".encode())
            for f in step.frames:
                h.update(f.digest().encode())
        return h.hexdigest()

    def summary(self) -> str:
        settled = self.last
        transient_total = sum(len(s.transients) for s in self.steps)
        return (
            f"{self.game_id} seed={self.seed} steps={len(self.steps)} "
            f"levels={settled.levels_completed}/{settled.win_levels} "
            f"state={settled.state.value} transients={transient_total} "
            f"resets={len(self.reset_points)} digest={self.digest()[:12]}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "initial": self.initial.to_json(),
            "steps": [s.to_json() for s in self.steps],
            "reset_points": list(self.reset_points),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> History:
        return cls(
            game_id=str(d["game_id"]),
            seed=int(d["seed"]),
            initial=Observation.from_json(d["initial"]),
            steps=[Step.from_json(s) for s in d["steps"]],
            reset_points=[int(i) for i in d["reset_points"]],
        )


def classify(
    grid_index: int,
    total_grids: int,
    state_is_terminal: bool,
    is_reset: bool,
) -> FrameKind:
    """Assign a FrameKind to one grid within a single engine emission.

    The last grid in an emission is the settled state; everything before it
    is animation. Terminal beats decision, and reset beats both, because
    those carry stronger constraints for the verifier.
    """
    last = grid_index == total_grids - 1
    if not last:
        return FrameKind.TRANSIENT
    if is_reset:
        return FrameKind.RESET
    if state_is_terminal:
        return FrameKind.TERMINAL
    return FrameKind.DECISION
