"""The replay verifier.

Threads a recorded history through a candidate world model and reports
where the model's story stopped matching reality.

This is the reward signal for the entire program. It is deterministic,
uses no LLM, and runs thousands of times per second, which is what makes
it possible to train the core on automatically-labelled data instead of
human annotation. Everything downstream inherits its correctness, so it
is deliberately strict and deliberately boring.

**Discontinuities are not scored.** Two things break causal continuity:
an engine reset, and a level boundary. At a level boundary the engine
swaps in a layout the model has never observed, so asking it to predict
that frame would penalise correct models for failing to be clairvoyant.
The model is still judged on whether it *called* the boundary — that is
the `outcome` check — but its render is not scored across the gap, and
its state is re-initialised for the new level.
"""

from __future__ import annotations

from typing import Any, Iterable

from sentinel.env.history import History
from sentinel.env.types import Action, GameStateName, Grid, Observation
from sentinel.wm.contract import (
    ABSTAIN,
    ModelError,
    Outcome,
    WorldModel,
    validate_rendered,
)

from .report import CellStats, StepResult, VerificationReport


def observed_outcome(obs: Observation, previous: Observation | None) -> Outcome:
    """What the environment actually did, in the model's vocabulary."""
    if obs.state is GameStateName.GAME_OVER:
        return Outcome.GAME_OVER
    if previous is not None and obs.levels_completed > previous.levels_completed:
        return Outcome.LEVEL_COMPLETE
    if obs.state is GameStateName.WIN:
        return Outcome.LEVEL_COMPLETE
    return Outcome.ONGOING


def compare(rendered: Any, actual: Grid) -> tuple[CellStats, bool]:
    """Score one predicted grid against the observed one.

    Abstained cells are excluded from accuracy and counted against
    coverage. `frame_match` is True only if every predicted cell is right,
    which is the property a planner actually needs when simulating forward.
    """
    total = 0
    predicted = 0
    correct = 0
    for row_pred, row_actual in zip(rendered, actual):
        for cell_pred, cell_actual in zip(row_pred, row_actual):
            total += 1
            if cell_pred == ABSTAIN:
                continue
            predicted += 1
            if cell_pred == cell_actual:
                correct += 1
    return CellStats(total=total, predicted=predicted, correct=correct), correct == predicted


class Verifier:
    """Replays histories through world models."""

    def __init__(self, strict_render: bool = True, stop_on_crash: bool = True) -> None:
        self.strict_render = strict_render
        self.stop_on_crash = stop_on_crash

    def verify(self, model: WorldModel, history: History) -> VerificationReport:
        """Score `model` against `history`."""
        steps: list[StepResult] = []
        crashed = False
        crash_detail: str | None = None

        try:
            state = model.init_state()
        except Exception as exc:  # noqa: BLE001 - a crash is a verification result
            return VerificationReport(
                model_name=model.name,
                game_id=history.game_id,
                seed=history.seed,
                steps=(
                    StepResult(
                        index=0,
                        action=None,
                        cells=CellStats(0, 0, 0),
                        frame_match=False,
                        outcome_predicted=None,
                        outcome_actual=Outcome.ONGOING,
                        scored=True,
                        error=f"init_state: {type(exc).__name__}: {exc}",
                    ),
                ),
                crashed=True,
                crash_detail=f"init_state raised {type(exc).__name__}: {exc}",
            )

        # Score the initial frame before any action is taken. A model that
        # cannot even draw the starting position has not understood the
        # environment's layout, and that is worth catching immediately.
        initial_result, state, err = self._score_frame(
            model, state, index=0, action=None,
            actual=history.initial, previous=None, scored=True, boundary=False,
        )
        steps.append(initial_result)
        if err and self.stop_on_crash:
            return VerificationReport(
                model_name=model.name, game_id=history.game_id, seed=history.seed,
                steps=tuple(steps), crashed=True, crash_detail=err,
            )

        reset_indices = set(history.reset_points)
        previous_obs = history.initial

        for step in history.steps:
            actual = step.settled
            is_reset = step.index in reset_indices
            is_level_change = actual.levels_completed != previous_obs.levels_completed
            boundary = is_reset or is_level_change

            try:
                state = model.transition(state, step.action)
            except Exception as exc:  # noqa: BLE001
                detail = f"transition({step.action}): {type(exc).__name__}: {exc}"
                steps.append(
                    StepResult(
                        index=step.index + 1, action=step.action,
                        cells=CellStats(0, 0, 0), frame_match=False,
                        outcome_predicted=None,
                        outcome_actual=observed_outcome(actual, previous_obs),
                        scored=True, boundary=boundary, error=detail,
                    )
                )
                crashed, crash_detail = True, detail
                if self.stop_on_crash:
                    break
                previous_obs = actual
                continue

            result, state, err = self._score_frame(
                model, state, index=step.index + 1, action=step.action,
                actual=actual, previous=previous_obs,
                scored=not boundary, boundary=boundary,
            )
            steps.append(result)

            if err:
                crashed, crash_detail = True, err
                if self.stop_on_crash:
                    break

            # Re-seed the model at a discontinuity. Without this, a model
            # that is entirely correct within levels would report near-zero
            # accuracy the moment the first level ended.
            if boundary:
                try:
                    state = model.reset_to(state)
                except Exception as exc:  # noqa: BLE001
                    crashed = True
                    crash_detail = f"reset_to: {type(exc).__name__}: {exc}"
                    if self.stop_on_crash:
                        break

            previous_obs = actual

        return VerificationReport(
            model_name=model.name,
            game_id=history.game_id,
            seed=history.seed,
            steps=tuple(steps),
            crashed=crashed,
            crash_detail=crash_detail,
        )

    def _score_frame(
        self,
        model: WorldModel,
        state: Any,
        *,
        index: int,
        action: Action | None,
        actual: Observation,
        previous: Observation | None,
        scored: bool,
        boundary: bool,
    ) -> tuple[StepResult, Any, str | None]:
        """Render and classify one state; never raises."""
        actual_outcome = observed_outcome(actual, previous)

        try:
            rendered = model.render(state)
            if self.strict_render:
                rendered = validate_rendered(rendered)
        except Exception as exc:  # noqa: BLE001
            detail = f"render: {type(exc).__name__}: {exc}"
            return (
                StepResult(
                    index=index, action=action, cells=CellStats(0, 0, 0),
                    frame_match=False, outcome_predicted=None,
                    outcome_actual=actual_outcome, scored=scored,
                    boundary=boundary, error=detail,
                ),
                state,
                detail,
            )

        try:
            predicted_outcome: Outcome | None = model.outcome(state)
        except Exception as exc:  # noqa: BLE001
            detail = f"outcome: {type(exc).__name__}: {exc}"
            return (
                StepResult(
                    index=index, action=action, cells=CellStats(0, 0, 0),
                    frame_match=False, outcome_predicted=None,
                    outcome_actual=actual_outcome, scored=scored,
                    boundary=boundary, error=detail,
                ),
                state,
                detail,
            )

        cells, frame_match = compare(rendered, actual.grid)
        return (
            StepResult(
                index=index, action=action, cells=cells,
                frame_match=frame_match if scored else True,
                outcome_predicted=predicted_outcome,
                outcome_actual=actual_outcome,
                scored=scored, boundary=boundary, error=None,
            ),
            state,
            None,
        )

    def verify_many(
        self, model: WorldModel, histories: Iterable[History]
    ) -> list[VerificationReport]:
        return [self.verify(model, h) for h in histories]


def verify(model: WorldModel, history: History) -> VerificationReport:
    """Convenience wrapper around the default Verifier."""
    return Verifier().verify(model, history)
