"""What a history can and cannot falsify.

A verifier is only as sharp as the evidence it is given. If a history
contains no level completion, then a model claiming "levels never end" is
factually correct on that history and cannot be refuted by it. If every
recorded action was ACTION1, nothing can be learned about ACTION2.

This is not a defect to paper over. It is the exploration problem arriving
early, and the honest response is to *report* it: a channel that could not
be tested must be visibly untested, never silently passed. Later, `explore/`
turns this into an objective — choose the actions that would raise evidence
coverage most — which is the same idea pointed forward instead of backward.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.env.history import History
from sentinel.env.types import GameStateName


@dataclass(frozen=True, slots=True)
class EvidenceCoverage:
    """Which model channels a given history is capable of falsifying."""

    steps: int
    distinct_actions: int
    has_level_boundary: bool
    has_game_over: bool
    has_win: bool
    has_state_change: bool
    action_sensitive: bool
    """True if the same visible grid was seen with different successors, or
    if distinct actions produced distinct outcomes — i.e. the history can
    tell an action-blind model apart from a correct one."""

    @property
    def can_test_render(self) -> bool:
        return self.steps > 0 and self.has_state_change

    @property
    def can_test_transition(self) -> bool:
        return self.distinct_actions >= 2 and self.action_sensitive

    @property
    def can_test_outcome(self) -> bool:
        return self.has_level_boundary or self.has_game_over or self.has_win

    def unexercised(self) -> list[str]:
        """Channels this history cannot falsify anything about."""
        missing = []
        if not self.can_test_render:
            missing.append("render")
        if not self.can_test_transition:
            missing.append("transition")
        if not self.can_test_outcome:
            missing.append("outcome")
        return missing

    def summary(self) -> str:
        gaps = self.unexercised()
        return (
            f"{self.steps} steps, {self.distinct_actions} distinct actions, "
            f"boundary={self.has_level_boundary} over={self.has_game_over} "
            f"win={self.has_win} action_sensitive={self.action_sensitive} "
            + ("all channels testable" if not gaps else f"UNTESTABLE: {', '.join(gaps)}")
        )


def evidence_coverage(history: History) -> EvidenceCoverage:
    """Assess what `history` is able to prove a model wrong about."""
    steps = history.steps
    distinct = {s.action.action_id for s in steps}

    has_boundary = False
    has_state_change = False
    previous = history.initial
    for step in steps:
        settled = step.settled
        if settled.levels_completed != previous.levels_completed:
            has_boundary = True
        if settled.grid != previous.grid:
            has_state_change = True
        previous = settled

    states = {s.settled.state for s in steps}
    has_over = GameStateName.GAME_OVER in states
    has_win = GameStateName.WIN in states

    # Action sensitivity: find a repeated grid where different actions led
    # to different successors. That is the direct proof a history can
    # distinguish an action-blind model from a correct one.
    by_grid: dict[tuple, set[tuple[int, tuple]]] = {}
    for step in steps:
        before = history.observation_before(step.index).grid
        by_grid.setdefault(before, set()).add((step.action.action_id, step.settled.grid))
    action_sensitive = any(
        len({aid for aid, _ in outcomes}) >= 2 and len({g for _, g in outcomes}) >= 2
        for outcomes in by_grid.values()
    )
    # Fall back to a weaker signal: distinct actions with distinct effects
    # anywhere in the history.
    if not action_sensitive and len(distinct) >= 2:
        effects: dict[int, set[bool]] = {}
        for step in steps:
            before = history.observation_before(step.index).grid
            effects.setdefault(step.action.action_id, set()).add(step.settled.grid != before)
        action_sensitive = len({frozenset(v) for v in effects.values()}) >= 2

    return EvidenceCoverage(
        steps=len(steps),
        distinct_actions=len(distinct),
        has_level_boundary=has_boundary,
        has_game_over=has_over,
        has_win=has_win,
        has_state_change=has_state_change,
        action_sensitive=action_sensitive,
    )
