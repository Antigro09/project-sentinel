"""Scale 1A-0R-O. Semantic roles, three appearance regimes, and honest renaming.

Phase N called an interface a "cell-aligned oracle". It was not an oracle: it read the
environment's own cell lattice out of the PIXELS and was therefore just as palette-bound
as everything else, which is exactly why it fell to 0.5000 under appearance shift along
with the rest. It is renamed `cell_aligned_color_grid` here, and a real oracle is added
that never sees a colour at all.

Three regimes, kept apart because they ask different questions:

  PHOTOMETRIC_JITTER      colours move, but the structural relation the renderer builds
                          in -- the agent is drawn as the inverse of the floor base --
                          survives, so roles stay recoverable from one frame without any
                          episode-level calibration.
  HIDDEN_PALETTE_CONVENTION  a per-episode bijection from roles to a fixed colour pool,
                          which destroys that relation. Nothing in a colour says what it
                          is; only behaviour does.
  PER_FRAME_PERMUTATION   a fresh bijection every frame. The impossibility control: no
                          persistent convention exists to infer.

    imported by o_identifiability.py, o_posterior.py, o_detect.py
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS, CELL, GRID, MARKERS, ProceduralVisualV2Adapter, build_level_v2)
from sentinel.wm.authority import AuthorityGate  # noqa: E402

ROLES = ("EMPTY", "WALL", "SWITCH", "GOAL_ALPHA", "GOAL_BETA", "AGENT")
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}
N_ROLES = len(ROLES)
FRAME = GRID * CELL

# A pool of well-separated colours. The bijection draws from this pool, so no colour
# carries any structural relation to any other -- which is the point.
COLOUR_POOL = np.array([
    [220, 40, 40], [40, 200, 60], [50, 80, 230], [235, 200, 40],
    [200, 60, 210], [40, 210, 215], [250, 140, 30], [150, 150, 150],
], dtype=np.uint8)

REGIMES = ("PHOTOMETRIC_JITTER", "HIDDEN_PALETTE_CONVENTION", "PER_FRAME_PERMUTATION")


def role_grid(level, position: tuple[int, int]) -> np.ndarray:
    """Evaluator-only semantic roles. Never an input to a learned visual arm.

    Occlusion is preserved exactly as the renderer applies it: the agent's own cell is
    AGENT, whatever lies beneath. A role grid that ignored that would let the semantic
    oracle see through the agent and would stop being a fair ceiling.
    """
    grid = np.full((GRID, GRID), ROLE_INDEX["EMPTY"], dtype=np.int64)
    grid[np.asarray(level.walls, dtype=bool)] = ROLE_INDEX["WALL"]
    for cell in level.switches:
        grid[tuple(int(v) for v in cell)] = ROLE_INDEX["SWITCH"]
    for marker, role in (("alpha", "GOAL_ALPHA"), ("beta", "GOAL_BETA")):
        grid[tuple(int(v) for v in level.markers[marker])] = ROLE_INDEX[role]
    grid[position] = ROLE_INDEX["AGENT"]
    return grid


def sample_bijection(seed: int) -> np.ndarray:
    """A role -> colour-pool-index bijection. Injective, so roles never share a colour."""
    rng = np.random.default_rng(seed)
    return rng.permutation(len(COLOUR_POOL))[:N_ROLES]


def render_roles(grid: np.ndarray, bijection: np.ndarray, stripe: int | None,
                 jitter: np.ndarray | None = None) -> np.ndarray:
    """Render a role grid through a bijection, at the environment's own cell scale."""
    colours = COLOUR_POOL[bijection]                       # (N_ROLES, 3)
    frame = colours[grid]                                  # (12, 12, 3)
    if jitter is not None:
        frame = np.clip(frame.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    rendered = np.repeat(np.repeat(frame, CELL, axis=0), CELL, axis=1)
    if stripe is not None:
        rendered[0, :] = 255 if stripe else 0
    return rendered.astype(np.uint8)


def photometric_jitter(seed: int, shape=(GRID, GRID, 3)) -> np.ndarray:
    """Per-cell additive texture plus a global brightness/contrast shift.

    Applied on TOP of a fixed bijection, so role identity stays recoverable from one
    frame: this regime perturbs appearance without renaming anything.
    """
    rng = np.random.default_rng(seed)
    texture = rng.integers(-18, 19, size=shape)
    brightness = rng.integers(-25, 26)
    return texture + brightness


IDENTITY_BIJECTION = np.arange(N_ROLES)


@dataclass
class Episode:
    layout: int
    regime: str
    palette_seed: int
    bijection: np.ndarray               # (N_ROLES,) role -> colour pool index
    per_frame_bijections: np.ndarray | None
    frames: np.ndarray                  # (T, 24, 24, 3) uint8
    roles: np.ndarray                   # (T, 12, 12) evaluator-only
    actions: np.ndarray                 # (T,)
    positions: np.ndarray               # (T, 2) evaluator-only
    polarity: np.ndarray                # (T,) evaluator-only
    event: np.ndarray                   # (T,) public target
    displacement: np.ndarray            # (T,)
    blocked: np.ndarray                 # (T,) was the previous action blocked
    goal_marker: str
    stripe: int

    @property
    def length(self) -> int:
        return len(self.frames)


def collect_appearance(layouts: Sequence[int], regime: str, palette_seeds: Sequence[int],
                       trajectories: int = 1, steps: int = 9, seed: int = 11,
                       policy: str = "uniform") -> list[Episode]:
    """Episodes under one appearance regime. Layout, policy and palette seeds are
    factored: the palette seed never touches the layout and vice versa."""
    from structured_calibration import DELTAS_BY_INDEX, DISPLACEMENTS

    gate = AuthorityGate(gate_id="o-appearance")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    rng = np.random.default_rng(seed)
    out: list[Episode] = []
    for layout in layouts:
        for palette_seed in palette_seeds:
            for _ in range(trajectories):
                adapter.reset(layout)
                level = adapter._require()
                goal = tuple(int(v) for v in level.markers[adapter._goal_marker])
                stripe = int(level.initial_polarity)
                bijection = (IDENTITY_BIJECTION if regime == "PHOTOMETRIC_JITTER"
                             else sample_bijection(palette_seed))
                frames, roles, actions, positions, polarity, blocked = [], [], [], [], [], []
                per_frame: list[np.ndarray] = []
                for step in range(steps):
                    truth = adapter.snapshot().reveal("evaluator")
                    position = tuple(int(v) for v in truth["position"])
                    grid = role_grid(level, position)
                    if regime == "PER_FRAME_PERMUTATION":
                        current = sample_bijection(palette_seed * 1_000 + step)
                    else:
                        current = bijection
                    per_frame.append(current)
                    jitter = (photometric_jitter(palette_seed * 100 + step)
                              if regime == "PHOTOMETRIC_JITTER" else None)
                    frames.append(render_roles(grid, current,
                                               stripe if step == 0 else None, jitter))
                    roles.append(grid)
                    positions.append(position)
                    polarity.append(int(truth["polarity"]))
                    blocked.append(float(truth["last_blocked"]))
                    if policy == "uniform":
                        action = int(rng.integers(0, len(ACTIONS)))
                    else:
                        scores = [abs(goal[0] - (position[0] + dr))
                                  + abs(goal[1] - (position[1] + dc))
                                  for dr, dc in DELTAS_BY_INDEX]
                        action = (int(rng.integers(0, len(ACTIONS)))
                                  if rng.random() < 0.25 else int(np.argmin(scores)))
                    actions.append(action)
                    if adapter.step(action,
                                    gate.authorize_evaluator(action, "o")).terminated:
                        break
                if len(frames) < 3:
                    continue
                length = len(frames)
                switches = {tuple(int(v) for v in c) for c in level.switches}
                displacement = np.zeros(length, dtype=np.int64)
                event = np.zeros(length, dtype=np.float32)
                for t in range(1, length):
                    delta = (positions[t][0] - positions[t - 1][0],
                             positions[t][1] - positions[t - 1][1])
                    displacement[t] = (DISPLACEMENTS.index(delta)
                                       if delta in DISPLACEMENTS else len(DISPLACEMENTS) - 1)
                    event[t] = float(positions[t] != positions[t - 1]
                                     and positions[t] in switches)
                out.append(Episode(
                    layout=layout, regime=regime, palette_seed=palette_seed,
                    bijection=bijection,
                    per_frame_bijections=(np.stack(per_frame)
                                          if regime == "PER_FRAME_PERMUTATION" else None),
                    frames=np.stack(frames), roles=np.stack(roles),
                    actions=np.array(actions[:length]), positions=np.array(positions),
                    polarity=np.array(polarity), event=event, displacement=displacement,
                    blocked=np.array(blocked, dtype=np.float32),
                    goal_marker=adapter._goal_marker, stripe=stripe))
    return out


def semantic_channels(episode: Episode, t: int) -> np.ndarray:
    """The SEMANTIC-ROLE ORACLE's input: role one-hots, no colour anywhere.

    Exact under every palette permutation by construction, because the palette never
    enters. If this arm ever moves when only the palette moves, the split leaks.
    """
    one_hot = np.zeros((GRID, GRID, N_ROLES), dtype=np.float32)
    rows, columns = np.indices((GRID, GRID))
    one_hot[rows, columns, episode.roles[t]] = 1.0
    return one_hot


def describe_projection() -> dict[str, Any]:
    """Section B: say exactly what the 'random projection' interface does.

    Calling it a random baseline understates it. It begins with an EXACT 2x2 pixel-block
    mean, and because the renderer paints every cell as a solid 2x2 block that step is
    lossless -- it recovers the environment's own 12x12 cell lattice byte for byte. The
    random projection that follows only mixes channels.
    """
    return {
        "spatial_aggregation": "exact 2x2 pixel-block mean, 24x24 -> 12x12",
        "aggregation_is_lossless": True,
        "why": ("the renderer emits each cell as a solid CELL x CELL block, so the block "
                "mean inverts the upsample exactly; only the reset stripe, which occupies "
                "half of row 0, is altered by it"),
        "projection_matrix": "frozen normal(0, 1/sqrt(in)) of shape (6, 64), seed 20002",
        "output_grid": [GRID, GRID],
        "channel_width": 64,
        "input_bytes_per_pair": 2 * FRAME * FRAME * 3,
        "aggregated_bytes_per_pair": 2 * GRID * GRID * 3,
        "compression_ratio": (2 * FRAME * FRAME * 3) / (2 * GRID * GRID * 3),
        "trainable_parameters": 0,
        "honest_name": "environment_aligned_pixel_aggregation_plus_frozen_projection",
    }


def digest_episodes(episodes: Iterable[Episode]) -> str:
    h = hashlib.sha256()
    for e in episodes:
        h.update(e.frames.tobytes())
        h.update(np.asarray(e.bijection).tobytes())
    return h.hexdigest()[:16]
