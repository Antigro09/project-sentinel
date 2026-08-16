"""Phase 1 gate: the verifier must not be foolable.

Two things are checked here.

1. **Metric semantics.** Accuracy, coverage and outcome must stay
   independent. The abstaining model is the canary: it is never wrong, so
   any scheme that collapses the numbers will rate it as good.

2. **Detection power.** A known-correct model is broken in eight specific
   ways and the verifier must catch every one. This is the gate. If it
   fails, the reward signal has a blind spot, and everything downstream
   would eventually be optimised into it.
"""

from __future__ import annotations

import random

import pytest

from sentinel.env import Action, History, Runner, available_games
from sentinel.env.types import GRID_SIZE
from sentinel.gen import (
    GridOnlyToyModel,
    ToyModel,
    ToyWorld,
    default_levels,
    solve_level,
)
from sentinel.verify import Verifier, evidence_coverage, verify
from sentinel.verify.mutate import standard_mutations
from sentinel.wm.reference import AbstainModel, OracleModel, StaticModel

GAMES = available_games()


def build_history(game_id: str, steps: int = 80, seed: int = 5) -> History:
    runner = Runner(game_id, seed=0)
    runner.reset()
    rng = random.Random(seed)
    for _ in range(steps):
        if runner.done:
            break
        legal = list(runner.last.available_actions)
        if not legal:
            break
        choice = rng.choice(legal)
        action = (
            Action(6, rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE))
            if choice == 6
            else Action(choice)
        )
        runner.step(action)
    return runner.history


@pytest.fixture(scope="module")
def history() -> History:
    return build_history(GAMES[0])


# --------------------------------------------------------------------------
# Metric semantics
# --------------------------------------------------------------------------


def test_oracle_scores_perfect(history: History) -> None:
    report = verify(OracleModel(history), history)
    assert report.is_perfect
    assert report.accuracy == 1.0
    assert report.coverage == 1.0
    assert report.first_divergence is None
    assert not report.crashed


def test_abstainer_is_accurate_but_useless(history: History) -> None:
    """The canary. Never wrong, never useful — the metrics must show both."""
    report = verify(AbstainModel(), history)
    assert report.accuracy == 1.0, "abstention must never count as an error"
    assert report.coverage == 0.0, "abstention must count against coverage"
    assert not report.is_perfect, "an empty model must never read as perfect"
    assert report.fitness == 0.0, "ranking must give abstention no credit"


def test_fitness_orders_models_sensibly(history: History) -> None:
    oracle = verify(OracleModel(history), history)
    static = verify(StaticModel(history.initial), history)
    abstain = verify(AbstainModel(), history)

    assert oracle.fitness > static.fitness > abstain.fitness


def test_static_model_has_full_coverage_but_diverges(history: History) -> None:
    """The floor: right about the wallpaper, wrong about the dynamics."""
    report = verify(StaticModel(history.initial), history)
    assert report.coverage == 1.0
    assert report.accuracy < 1.0
    assert report.first_divergence is not None


# --------------------------------------------------------------------------
# The gate: detection power
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy() -> tuple[History, tuple]:
    """A history that deliberately exercises every channel.

    Random play on a real game routinely never completes a level and never
    dies, so it cannot falsify a model that claims nothing ever ends. The
    toy world is solved level by level so boundaries, wins and
    action-sensitivity are all present in the evidence.
    """
    levels = default_levels()
    world = ToyWorld(levels)
    world.reset()
    for _ in range(len(levels)):
        if world.done:
            break
        path = solve_level(world.state, levels)
        assert path is not None, "toy level must be solvable"
        for action in path:
            world.step(action)
    return world.history, levels


def test_toy_evidence_exercises_all_channels(toy) -> None:
    """Precondition for the gate: the evidence must be able to refute."""
    history, _ = toy
    coverage = evidence_coverage(history)
    assert not coverage.unexercised(), coverage.summary()
    assert coverage.has_level_boundary
    assert coverage.action_sensitive


@pytest.mark.parametrize("spec", standard_mutations(cell=(1, 1)), ids=lambda s: s.label)
def test_verifier_catches_every_injected_bug(spec, toy) -> None:
    """THE PHASE 1 GATE.

    A model correct by construction, broken one way at a time. Every break
    must be detected. A miss means the reward signal has a blind spot, and
    everything downstream would eventually optimise into it.
    """
    history, levels = toy
    clean = verify(ToyModel(levels), history)
    assert clean.is_perfect, f"precondition failed: {clean.summary()}"

    broken = verify(spec.build(ToyModel(levels)), history)

    assert not broken.is_perfect, f"{spec.label} went undetected"
    assert broken.fitness < clean.fitness, f"{spec.label} did not lower fitness"
    assert (
        broken.first_divergence is not None or broken.crashed
    ), f"{spec.label} produced no divergence and no crash"


def test_hidden_state_denial_is_caught(toy) -> None:
    """The most important negative control in the project.

    `GridOnlyToyModel` refuses to posit unobserved structure. It is right
    about 99.9% of *cells* — the grid is mostly background — while being
    wrong on nearly every *frame*. If accuracy alone were the score, it
    would read as essentially correct, and the claim that understanding
    requires hidden state would be untestable.
    """
    history, levels = toy
    exact = verify(ToyModel(levels), history)
    blind = verify(GridOnlyToyModel(levels), history)

    assert exact.is_perfect
    assert not blind.is_perfect
    assert blind.accuracy > 0.99, "the trap: cell accuracy stays near-perfect"
    assert blind.transition_match < 0.5, "but frame-level match must collapse"
    assert blind.first_divergence is not None


def test_unexercised_channels_are_reported_not_hidden() -> None:
    """A history that cannot refute a channel must say so.

    Silently passing is the dangerous outcome — it reads as evidence of
    correctness when it is only absence of evidence.
    """
    levels = default_levels()
    world = ToyWorld(levels)
    world.reset()
    for _ in range(5):
        world.step(Action(5))  # wait only: no movement, no level ends

    coverage = evidence_coverage(world.history)
    assert "outcome" in coverage.unexercised()
    assert "UNTESTABLE" in coverage.summary()


@pytest.mark.parametrize("game_id", GAMES[:4])
def test_render_channel_gate_holds_on_real_games(game_id: str) -> None:
    """On real games, random play can only test the render channel.

    Asserted narrowly and honestly: the transition and outcome channels are
    checked against the toy world above, where the evidence supports it.
    """
    hist = build_history(game_id, steps=60, seed=9)
    clean = verify(OracleModel(hist), hist)
    if not clean.is_perfect:
        pytest.skip(f"{game_id}: oracle precondition not met")

    render_mutations = [s for s in standard_mutations() if s.targets == "render"]
    undetected = [
        spec.label
        for spec in render_mutations
        if verify(spec.build(OracleModel(hist)), hist).is_perfect
    ]
    assert not undetected, f"{game_id}: undetected render mutations {undetected}"


def test_single_wrong_cell_is_detected(history: History) -> None:
    """Sharpest resolution test: one cell, one frame, must still be caught."""
    from sentinel.verify.mutate import CorruptOneCell

    broken = verify(CorruptOneCell(OracleModel(history), x=10, y=10), history)
    assert broken.first_divergence is not None
    assert broken.transition_match < 1.0


def test_crash_is_recorded_not_raised(history: History) -> None:
    """A model that raises must be scored, not allowed to escape."""
    from sentinel.verify.mutate import CrashOnStep

    report = verify(CrashOnStep(OracleModel(history), after=3), history)
    assert report.crashed
    assert report.crash_detail and "injected fault" in report.crash_detail
    assert report.fitness == 0.0


def test_malformed_render_is_rejected(history: History) -> None:
    """Structurally invalid output is a different failure than a wrong guess."""
    from sentinel.wm.contract import RenderedGrid, WorldModel

    class Malformed(OracleModel):
        name = "malformed"

        def render(self, state) -> RenderedGrid:  # type: ignore[override]
            return ((1, 2, 3),)  # wrong shape

    report = verify(Malformed(history), history)
    assert report.crashed
    assert report.crash_detail and "render" in report.crash_detail


# --------------------------------------------------------------------------
# Discontinuity handling
# --------------------------------------------------------------------------


def test_boundaries_are_not_scored_but_are_counted(history: History) -> None:
    """Level/reset boundaries must be excluded from render scoring.

    Otherwise a model that is perfectly correct within a level is punished
    for failing to predict a layout it has never observed.
    """
    report = verify(OracleModel(history), history)
    boundaries = [s for s in report.steps if s.boundary]
    for step in boundaries:
        assert not step.scored
    assert len(report.scored_steps) == len(report.steps) - len(boundaries)


def test_verifier_can_continue_past_crash(history: History) -> None:
    from sentinel.verify.mutate import CrashOnStep

    lenient = Verifier(stop_on_crash=False)
    report = lenient.verify(CrashOnStep(OracleModel(history), after=3), history)
    assert report.crashed
    assert len(report.steps) > 4, "should have kept scoring after the fault"
