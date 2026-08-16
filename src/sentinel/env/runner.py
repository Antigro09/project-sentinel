"""Environment runner — the only place that touches the ARC engine.

Everything above this module works with `Observation`/`Action`/`History`
and never imports arcengine. That boundary is deliberate: Phase 6 swaps a
different environment in behind this interface, and nothing upstream
should need to change when it does.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

from .history import History, classify
from .types import Action, GameStateName, Grid, Observation

_TERMINAL_STATES = {GameStateName.WIN, GameStateName.GAME_OVER}

_ENGINE_LOGGERS = (
    "arc_agi",
    "arc_agi.base",
    "arc_agi.local_wrapper",
    "arc_agi.scorecard",
    "arc_agi.wrapper",
)


def silence_engine_logging(level: int = logging.ERROR) -> None:
    """Quiet the engine's INFO chatter.

    Call after constructing an Arcade — it resets its own logger in
    __init__, so anything set beforehand is discarded.
    """
    for name in _ENGINE_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False


def _to_grid(raw: Any) -> Grid:
    """Convert an engine grid to an immutable tuple-of-tuples of plain ints.

    The engine hands back numpy int8. Those do not hash stably across
    processes and they serialise awkwardly, so they are normalised here at
    the boundary rather than defensively everywhere upstream.
    """
    return tuple(tuple(int(cell) for cell in row) for row in raw)


def _observations_from_frame(frame_data: Any) -> tuple[Observation, ...]:
    """Split one engine emission into typed observations."""
    grids = frame_data.frame
    state = GameStateName(frame_data.state.name)
    is_terminal = state in _TERMINAL_STATES
    is_reset = bool(frame_data.full_reset)
    total = len(grids)

    return tuple(
        Observation(
            grid=_to_grid(g),
            state=state,
            levels_completed=int(frame_data.levels_completed),
            win_levels=int(frame_data.win_levels),
            available_actions=tuple(int(a) for a in frame_data.available_actions),
            full_reset=is_reset,
            kind=classify(i, total, is_terminal, is_reset),
        )
        for i, g in enumerate(grids)
    )


class Runner:
    """Drives one environment and records a typed History.

    Construct, `reset()`, then `step()` repeatedly. The History is built as
    you go and is the artifact everything downstream consumes.
    """

    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        environments_dir: str = "environment_files",
        quiet: bool = True,
    ) -> None:
        import arc_agi  # imported lazily so the package stays importable without it

        self.game_id = game_id
        self.seed = seed

        # OFFLINE is not just a default — it is a guarantee. In any other
        # mode the Arcade constructor will reach for an anonymous API key
        # and hit the network, which would silently break the premise that
        # this whole system runs without it.
        # Silenced on both sides of construction: the ScorecardManager logs
        # while Arcade is still being built, and Arcade then clears handlers
        # and forces its own logger back to INFO. At the episode volumes this
        # system runs at, the default chatter is louder than any signal in it.
        if quiet:
            silence_engine_logging()

        self._arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=environments_dir,
        )
        if quiet:
            silence_engine_logging()

        env = self._arcade.make(game_id, seed=seed)
        if env is None:
            available = sorted(
                {e.game_id.split("-")[0] for e in self._arcade.get_environments()}
            )
            raise RuntimeError(
                f"could not make environment {game_id!r}. "
                f"Available locally: {', '.join(available) or '(none)'}. "
                f"Run scripts/fetch_games.py to download them."
            )
        self._env = env
        self._history: History | None = None
        self._level_index = 0
        self._closed = False

    @property
    def history(self) -> History:
        if self._history is None:
            raise RuntimeError("call reset() before accessing history")
        return self._history

    @property
    def last(self) -> Observation:
        return self.history.last

    @property
    def done(self) -> bool:
        return self.last.is_terminal

    def reset(self) -> Observation:
        """Start a fresh episode and begin a new History."""
        frame_data = self._env.reset()
        if frame_data is None:
            raise RuntimeError(f"reset failed for {self.game_id}")
        obs = _observations_from_frame(frame_data)[-1]
        self._level_index = obs.levels_completed
        self._history = History(game_id=self.game_id, seed=self.seed, initial=obs)
        return obs

    def step(self, action: Action | int) -> Observation:
        """Submit one action; return the settled observation."""
        from arcengine import GameAction

        if isinstance(action, int):
            action = Action(action)
        history = self.history

        frame_data = self._env.step(
            GameAction[f"ACTION{action.action_id}"], data=action.data
        )
        if frame_data is None:
            raise RuntimeError(f"step failed: {action} on {self.game_id}")

        frames = _observations_from_frame(frame_data)
        settled = frames[-1]
        self._level_index = settled.levels_completed

        from .types import Step as StepRecord

        history.append(
            StepRecord(
                index=len(history.steps),
                action=action,
                frames=frames,
                level_index=self._level_index,
            )
        )
        return settled

    def run(self, actions: Iterable[Action | int], stop_on_done: bool = True) -> History:
        """Execute a sequence of actions. Used by replay and by planners."""
        if self._history is None:
            self.reset()
        for a in actions:
            if stop_on_done and self.done:
                break
            self.step(a)
        return self.history

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Runner:
        self.reset()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def available_games(
    environments_dir: str = "environment_files", quiet: bool = True
) -> list[str]:
    """Game ids present on disk, usable with no network."""
    import arc_agi

    if not os.path.isdir(environments_dir):
        return []
    if quiet:
        silence_engine_logging()
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=environments_dir,
    )
    if quiet:
        silence_engine_logging()
    return sorted({e.game_id.split("-")[0] for e in arcade.get_environments()})
