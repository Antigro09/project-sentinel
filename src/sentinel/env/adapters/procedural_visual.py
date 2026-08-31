"""Adapter B: a procedurally generated visual domain.

The strategy document asks for "Procgen or a small generated equivalent" at
Stratum A. This is the generated equivalent, and it is generated rather than
downloaded for a reason that matters beyond convenience: Scale 0 must be able to
hold appearance fixed while mechanics change, and mechanics fixed while
appearance changes. A third-party level generator does not let you do that, and
without it the Scale-4 surface/mechanism swap has no environment to run in.

So the two axes are separate seeds. `appearance_seed` picks the palette and the
decorative texture; `dynamics_seed` picks the movement rule and the wall layout.
Neither is visible to the model as a number -- only their consequences are.

The hidden variable is `charge`, a counter that is never rendered and that makes
every third move travel two cells. It is what stops the rendered frame from
being a sufficient state, which is the condition under which a recurrent belief
has anything to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sentinel.env.adapters.base import (
    EnvironmentIdentity,
    HiddenSnapshot,
    ProbeSet,
    StepResult,
)
from sentinel.wm.authority import AuthorityGate, AuthorizationToken
from sentinel.wm.latent_contract import (
    ContractViolation,
    Modality,
    ModalityMask,
    ObservationEnvelope,
    Taint,
)
from sentinel.wm.versioning import digest_array, digest_of

GRID = 12
"""Cells per side. The rendered frame is GRID*CELL square."""
CELL = 2
FRAME = GRID * CELL
ACTIONS: tuple[int, ...] = (0, 1, 2, 3)  # up, right, down, left
DELTAS: Mapping[int, tuple[int, int]] = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
CHARGE_PERIOD = 3
"""Every third move travels two cells. Never rendered; inferable from history."""
DEFAULT_HORIZON = 32
GENERATOR_VERSION = "procedural-visual-v1"

MASK = ModalityMask(
    declared=(Modality.IMAGE, Modality.STRUCTURED, Modality.GOAL),
    present=(Modality.IMAGE, Modality.STRUCTURED, Modality.GOAL),
)


def _stream(seed_material: Mapping[str, Any], count: int) -> np.ndarray:
    """Deterministic integer stream from a digest. No global RNG, ever."""
    import hashlib

    material = digest_of(dict(seed_material)).encode()
    blocks: list[bytes] = []
    counter = 0
    while sum(len(b) for b in blocks) < count * 4:
        blocks.append(hashlib.sha256(material + counter.to_bytes(4, "big")).digest())
        counter += 1
    return np.frombuffer(b"".join(blocks)[: count * 4], dtype=np.uint32)


@dataclass(frozen=True, slots=True)
class Level:
    """One generated level: layout and mechanics from one seed, looks from another."""

    walls: np.ndarray
    palette: np.ndarray
    texture: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    mirrored: bool

    @property
    def digest(self) -> str:
        return digest_of(
            {
                "walls": digest_array(self.walls).canonical_dict(),
                "palette": digest_array(self.palette).canonical_dict(),
                "texture": digest_array(self.texture).canonical_dict(),
                "start": list(self.start),
                "goal": list(self.goal),
                "mirrored": self.mirrored,
            }
        )


def build_level(dynamics_seed: int, appearance_seed: int) -> Level:
    """Layout and rule from one seed; palette and texture from the other.

    Keeping the draws in separate streams is what makes the two axes genuinely
    independent -- a shared stream would correlate appearance with layout and
    every later "appearance changed, mechanics did not" claim would be false.
    """
    layout_draw = _stream({"axis": "dynamics", "seed": int(dynamics_seed)}, GRID * GRID + 8)
    look_draw = _stream({"axis": "appearance", "seed": int(appearance_seed)}, GRID * GRID + 16)

    walls = (layout_draw[: GRID * GRID] % 100 < 18).reshape(GRID, GRID)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True

    def free_cell(offset: int) -> tuple[int, int]:
        for i in range(GRID * GRID):
            index = int(layout_draw[GRID * GRID + offset % 8] + i) % (GRID * GRID)
            row, column = divmod(index, GRID)
            if not walls[row, column]:
                return row, column
        raise ContractViolation("generated level has no free cell")

    start = free_cell(0)
    goal = free_cell(4)
    if goal == start:
        goal = free_cell(5)
    walls[start] = walls[goal] = False

    palette = (look_draw[:12] % 200 + 30).astype(np.uint8).reshape(4, 3)
    texture = (look_draw[16 : 16 + GRID * GRID] % 40).astype(np.uint8).reshape(GRID, GRID)
    mirrored = bool(layout_draw[GRID * GRID + 1] % 2)
    return Level(walls, palette, texture, start, goal, mirrored)


def render(level: Level, position: tuple[int, int]) -> np.ndarray:
    """RGB frame. The hidden charge counter is deliberately not drawn."""
    frame = np.zeros((GRID, GRID, 3), dtype=np.uint8)
    frame[:, :] = level.palette[0]
    frame[level.walls] = level.palette[1]
    frame += level.texture[:, :, None] // 4
    frame[level.goal] = level.palette[2]
    frame[position] = level.palette[3]
    return np.repeat(np.repeat(frame, CELL, axis=0), CELL, axis=1)


@dataclass
class ProceduralVisualAdapter:
    """Reset, step, legal actions, observable probes, branch and restore."""

    gate: AuthorityGate
    horizon: int = DEFAULT_HORIZON
    appearance_offset: int = 0
    _level: Level | None = field(default=None, init=False)
    _position: tuple[int, int] = field(default=(0, 0), init=False)
    _charge: int = field(default=0, init=False)
    _step: int = field(default=0, init=False)
    _seed: int = field(default=0, init=False)
    _dynamic: str = field(default="base", init=False)
    _last_blocked: bool = field(default=False, init=False)
    _reached: bool = field(default=False, init=False)
    _interactions: int = field(default=0, init=False)

    @property
    def identity(self) -> EnvironmentIdentity:
        return EnvironmentIdentity(
            name="procedural_visual",
            version=GENERATOR_VERSION,
            generator_digest=digest_of(
                {
                    "grid": GRID,
                    "cell": CELL,
                    "actions": list(ACTIONS),
                    "charge_period": CHARGE_PERIOD,
                    "version": GENERATOR_VERSION,
                }
            ),
            supports_branching=True,
        )

    @property
    def interactions(self) -> int:
        return self._interactions

    @property
    def episode_id(self) -> str:
        return f"procedural_visual:{self._dynamic}:{self._seed}"

    def _require_level(self) -> Level:
        if self._level is None:
            raise ContractViolation("adapter used before reset")
        return self._level

    def _frame(self) -> np.ndarray:
        return render(self._require_level(), self._position)

    def _observation(self) -> ObservationEnvelope:
        level = self._require_level()
        frame = self._frame()
        return ObservationEnvelope(
            episode_id=self.episode_id,
            step=self._step,
            timestamp_ns=self._step,
            modality_payloads={"image": digest_array(frame)},
            structured_observation={
                "steps_remaining": self.horizon - self._step,
                "frame_shape": list(frame.shape),
                "goal_visible": True,
            },
            modality_mask=MASK,
            available_action_digest=digest_of(list(ACTIONS)),
            environment_version=digest_of(
                {"identity": self.identity.digest, "level": level.digest}
            ),
            taint=frozenset({Taint.DEVELOPMENT}),
        )

    def frame(self) -> np.ndarray:
        """The rendered pixels, for an encoder. Never includes hidden state."""
        return self._frame()

    def probes(self) -> ProbeSet:
        level = self._require_level()
        distance = abs(self._position[0] - level.goal[0]) + abs(self._position[1] - level.goal[1])
        return ProbeSet(
            {
                "reward": 1.0 if self._reached else 0.0,
                "termination": self._reached or self._step >= self.horizon,
                "goal_progress": float(1.0 - distance / (2 * GRID)),
                "constraint_violation": bool(level.walls[self._position]),
                "action_succeeded": not self._last_blocked,
                "observable_signature": int(self._position[0] * GRID + self._position[1]),
            }
        )

    def legal_actions(self) -> tuple[int, ...]:
        return ACTIONS

    def reset(self, seed: int, dynamic: str = "base") -> StepResult:
        self._seed = int(seed)
        self._dynamic = dynamic
        appearance = int(seed) + self.appearance_offset
        if dynamic.startswith("appearance:"):
            appearance = int(dynamic.split(":", 1)[1])
            dynamics_seed = int(seed)
        elif dynamic.startswith("mechanic:"):
            dynamics_seed = int(dynamic.split(":", 1)[1])
        else:
            dynamics_seed = int(seed)
        self._level = build_level(dynamics_seed, appearance)
        self._position = self._level.start
        self._charge = 0
        self._step = 0
        self._reached = False
        self._last_blocked = False
        return self._result()

    def _result(self) -> StepResult:
        probes = self.probes()
        return StepResult(
            observation=self._observation(),
            reward=float(probes.values["reward"]),
            terminated=bool(probes.values["termination"]),
            probes=probes,
            legal_actions=self.legal_actions(),
            info={"dynamic": self._dynamic},
        )

    def step(self, action: int, token: AuthorizationToken) -> StepResult:
        self.gate.consume(token, action)
        if action not in ACTIONS:
            raise ContractViolation(f"action {action} is not legal here; legal actions {ACTIONS}")
        level = self._require_level()
        row_delta, column_delta = DELTAS[action]
        if level.mirrored:
            row_delta, column_delta = -row_delta, -column_delta
        distance = 2 if (self._charge + 1) % CHARGE_PERIOD == 0 else 1

        position = self._position
        blocked = False
        for _ in range(distance):
            candidate = (position[0] + row_delta, position[1] + column_delta)
            if not (0 <= candidate[0] < GRID and 0 <= candidate[1] < GRID):
                blocked = True
                break
            if level.walls[candidate]:
                blocked = True
                break
            position = candidate

        self._position = position
        self._last_blocked = blocked
        self._charge = (self._charge + 1) % CHARGE_PERIOD
        self._step += 1
        self._interactions += 1
        if position == level.goal:
            self._reached = True
        return self._result()

    def snapshot(self) -> HiddenSnapshot:
        level = self._require_level()
        return HiddenSnapshot(
            payload={
                "position": list(self._position),
                "charge": self._charge,
                "step": self._step,
                "seed": self._seed,
                "dynamic": self._dynamic,
                "reached": self._reached,
                "last_blocked": self._last_blocked,
                "level_digest": level.digest,
                "_non_observable": ("charge",),
            },
            environment_version=self.identity.digest,
        )

    def restore(self, snapshot: HiddenSnapshot) -> StepResult:
        payload = snapshot.reveal("restore")
        if self._level is None or payload["seed"] != self._seed or payload["dynamic"] != self._dynamic:
            self.reset(int(payload["seed"]), str(payload["dynamic"]))
        level = self._require_level()
        if level.digest != payload["level_digest"]:
            raise ContractViolation(
                "restore target was generated by a different level; the snapshot "
                "cannot be replayed into this environment"
            )
        self._position = tuple(int(v) for v in payload["position"])  # type: ignore[assignment]
        self._charge = int(payload["charge"])
        self._step = int(payload["step"])
        self._reached = bool(payload["reached"])
        self._last_blocked = bool(payload["last_blocked"])
        return self._result()
