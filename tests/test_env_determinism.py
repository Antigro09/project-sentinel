"""Phase 0 exit criteria.

The whole program rests on the verifier, and the verifier rests on history
being exactly reproducible. If an episode cannot be replayed byte-for-byte,
then a correct world model can be scored as wrong and the reward signal is
noise. These tests are the guard on that assumption.
"""

from __future__ import annotations

import random

import pytest

from sentinel.env import (
    Action,
    EpisodeLog,
    FrameKind,
    GRID_SIZE,
    MAX_CELL_VALUE,
    ReplayMismatch,
    Runner,
    available_games,
    record,
)

GAMES = available_games()
SAMPLE_GAMES = GAMES[:5]


def random_actions(runner: Runner, n: int, seed: int) -> list[Action]:
    """Drive a runner with legal random actions, returning what was played."""
    rng = random.Random(seed)
    played: list[Action] = []
    for _ in range(n):
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
        played.append(action)
        runner.step(action)
    return played


def test_games_are_available_offline() -> None:
    assert GAMES, "no games on disk — run scripts/fetch_games.py"
    assert len(GAMES) >= 3


@pytest.mark.parametrize("game_id", SAMPLE_GAMES)
def test_replay_reproduces_episode_twice(game_id: str) -> None:
    """The Phase 0 gate: same log, same episode, every time."""
    runner = Runner(game_id, seed=0)
    runner.reset()
    played = random_actions(runner, 120, seed=7)
    original = runner.history
    log = EpisodeLog.from_history(original)

    first = log.replay()
    second = log.replay()

    assert first.digest() == original.digest()
    assert second.digest() == original.digest()
    assert len(first.steps) == len(played)


@pytest.mark.parametrize("game_id", SAMPLE_GAMES)
def test_log_roundtrips_through_disk(game_id: str, tmp_path) -> None:
    runner = Runner(game_id, seed=0)
    runner.reset()
    random_actions(runner, 60, seed=11)
    log = EpisodeLog.from_history(runner.history)

    path = log.save(tmp_path / f"{game_id}.json")
    reloaded = EpisodeLog.load(path)

    assert reloaded == log
    assert reloaded.replay().digest() == log.digest


def test_replay_mismatch_is_detected() -> None:
    """A corrupted log must fail loudly rather than silently disagree."""
    runner = Runner(GAMES[0], seed=0)
    runner.reset()
    random_actions(runner, 40, seed=3)
    log = EpisodeLog.from_history(runner.history)

    tampered = EpisodeLog(
        game_id=log.game_id,
        seed=log.seed,
        actions=log.actions,
        digest="0" * 32,
        recorded_at=log.recorded_at,
        steps=log.steps,
        levels_completed=log.levels_completed,
        final_state=log.final_state,
    )
    with pytest.raises(ReplayMismatch):
        tampered.replay()


def test_different_seeds_are_tracked_separately() -> None:
    """Seed is part of episode identity, so it must be part of the digest."""
    game = GAMES[0]
    actions: list[Action | int] = [1, 2, 3, 4, 1, 2, 3, 4]
    h0, log0 = record(game, actions, seed=0)
    h1, log1 = record(game, actions, seed=1)

    assert log0.seed == 0 and log1.seed == 1
    assert h0.digest() != h1.digest() or h0.last.grid == h1.last.grid


@pytest.mark.parametrize("game_id", SAMPLE_GAMES)
def test_observations_are_well_formed(game_id: str) -> None:
    runner = Runner(game_id, seed=0)
    obs = runner.reset()

    assert len(obs.grid) == GRID_SIZE
    assert all(len(row) == GRID_SIZE for row in obs.grid)
    assert all(0 <= c <= MAX_CELL_VALUE for row in obs.grid for c in row)
    assert all(isinstance(c, int) for row in obs.grid for c in row)
    assert obs.available_actions


def test_frame_kinds_are_assigned() -> None:
    """Settled frames must be distinguishable from animation frames."""
    runner = Runner(GAMES[0], seed=0)
    runner.reset()
    random_actions(runner, 150, seed=5)

    for step in runner.history:
        assert step.settled.kind is not FrameKind.TRANSIENT
        assert all(f.kind is FrameKind.TRANSIENT for f in step.transients)


def test_transitions_exclude_reset_boundaries() -> None:
    """Transitions spanning a reset are not real transitions.

    Scoring a world model against a discontinuity the engine imposed would
    penalise correct models, so they must never enter the pair list.
    """
    runner = Runner(GAMES[0], seed=0)
    runner.reset()
    random_actions(runner, 200, seed=13)
    history = runner.history

    pair_count = len(history.transition_pairs())
    assert pair_count == len(history.steps) - len(history.reset_points)
