"""Episode logging and deterministic replay.

An episode is fully determined by (game_id, seed, action sequence). That
makes the log compact: we store the inputs plus a digest of what happened,
not megabytes of grids. Replaying regenerates the frames and checks the
digest still matches.

This matters beyond tidiness. The verifier scores world models against
recorded history, so if history is not exactly reproducible then a model
that is genuinely correct can be marked wrong by engine drift. Determinism
is the assumption the entire reward signal rests on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .history import History
from .runner import Runner
from .types import Action


class ReplayMismatch(RuntimeError):
    """Replaying a log did not reproduce the recorded episode."""


@dataclass(frozen=True, slots=True)
class EpisodeLog:
    """Compact, replayable record of one episode."""

    game_id: str
    seed: int
    actions: tuple[Action, ...]
    digest: str
    recorded_at: str
    steps: int
    levels_completed: int
    final_state: str

    @classmethod
    def from_history(cls, history: History) -> EpisodeLog:
        last = history.last
        return cls(
            game_id=history.game_id,
            seed=history.seed,
            actions=tuple(history.action_sequence()),
            digest=history.digest(),
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            steps=len(history.steps),
            levels_completed=last.levels_completed,
            final_state=last.state.value,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "actions": [a.to_json() for a in self.actions],
            "digest": self.digest,
            "recorded_at": self.recorded_at,
            "steps": self.steps,
            "levels_completed": self.levels_completed,
            "final_state": self.final_state,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> EpisodeLog:
        return cls(
            game_id=str(d["game_id"]),
            seed=int(d["seed"]),
            actions=tuple(Action.from_json(a) for a in d["actions"]),
            digest=str(d["digest"]),
            recorded_at=str(d["recorded_at"]),
            steps=int(d["steps"]),
            levels_completed=int(d["levels_completed"]),
            final_state=str(d["final_state"]),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> EpisodeLog:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def replay(self, environments_dir: str = "environment_files") -> History:
        """Re-execute the episode and return the regenerated History.

        Raises ReplayMismatch if the result differs from what was recorded.
        """
        runner = Runner(
            self.game_id, seed=self.seed, environments_dir=environments_dir
        )
        runner.reset()
        # stop_on_done is off: the recorded sequence is authoritative. If the
        # episode ends earlier on replay than it did on record, that is a
        # determinism failure we want surfaced, not quietly truncated.
        history = runner.run(self.actions, stop_on_done=False)

        if history.digest() != self.digest:
            raise ReplayMismatch(
                f"{self.game_id} seed={self.seed}: "
                f"recorded {self.digest[:16]} but replayed {history.digest()[:16]} "
                f"({self.steps} recorded steps vs {len(history.steps)} replayed)"
            )
        return history


def record(
    game_id: str,
    actions: list[Action | int],
    seed: int = 0,
    environments_dir: str = "environment_files",
) -> tuple[History, EpisodeLog]:
    """Run an action sequence and return both the history and its log."""
    runner = Runner(game_id, seed=seed, environments_dir=environments_dir)
    runner.reset()
    history = runner.run(actions, stop_on_done=False)
    return history, EpisodeLog.from_history(history)
