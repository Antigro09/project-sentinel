"""C. Appearance strata that remove the fixed-cardinality shortcut.

Phase O found SWITCH pinned from a single frame, and the reason was not perception: seven
switch cells is a generator constant, so counting cells identifies the role. Any model
that "solves" the hidden palette under that generator may only be doing a cardinality
lookup, and this module builds the strata that tell the two apart.

A DECOY role is introduced. It renders as its own colour and behaves EXACTLY like EMPTY --
walkable, no polarity flip -- so the semantic dynamics are untouched, as the specification
requires. What changes is only how informative a cell count is:

  COUNT_INFORMATIVE  no decoy; SWITCH is the unique role with seven cells. Diagnostic.
  COUNT_VARIED       decoy count drawn from 4..10, so "seven cells" no longer names a role.
  COUNT_COLLISION    decoy count is exactly the switch count, so cardinality NEVER
                     separates them and only interaction can -- crossing a switch flips
                     the polarity, crossing a decoy does nothing.

    imported by p_binding.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import o_core as O
from o_core import COLOUR_POOL, GRID, ROLES as BASE_ROLES

ROLES = BASE_ROLES + ("DECOY",)
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}
N_ROLES = len(ROLES)
STRATA = ("COUNT_INFORMATIVE", "COUNT_VARIED", "COUNT_COLLISION")
SWITCH_COUNT = 7


def decoy_count(stratum: str, rng) -> int:
    if stratum == "COUNT_INFORMATIVE":
        return 0
    if stratum == "COUNT_COLLISION":
        return SWITCH_COUNT
    return int(rng.integers(4, 11))


def sample_bijection(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(len(COLOUR_POOL))[:N_ROLES]


def add_decoy(grid: np.ndarray, count: int, rng) -> np.ndarray:
    """Repaint `count` EMPTY cells as DECOY. Semantics unchanged: decoy IS empty."""
    if count <= 0:
        return grid
    empty = np.argwhere(grid == ROLE_INDEX["EMPTY"])
    if len(empty) <= count:
        return grid
    picked = rng.choice(len(empty), size=count, replace=False)
    out = grid.copy()
    for index in picked:
        out[tuple(empty[index])] = ROLE_INDEX["DECOY"]
    return out


@dataclass
class StratumEpisode:
    base: O.Episode
    stratum: str
    decoy_cells: int
    roles: np.ndarray          # (T, 12, 12) with DECOY, evaluator-only
    frames: np.ndarray         # (T, 24, 24, 3)
    bijection: np.ndarray

    @property
    def length(self) -> int:
        return len(self.frames)


def build_stratum(episodes: Sequence[O.Episode], stratum: str, palette_seed: int,
                  seed: int = 11) -> list[StratumEpisode]:
    """Re-render existing episodes under a stratum. The trajectory, the layout and every
    semantic target are reused unchanged, so a stratum comparison varies appearance
    statistics and nothing else."""
    rng = np.random.default_rng(seed)
    out: list[StratumEpisode] = []
    for episode in episodes:
        count = decoy_count(stratum, rng)
        # The decoy cells are fixed for the episode: they are scenery, not events.
        placement = np.random.default_rng(seed * 1_000 + episode.layout)
        first = add_decoy(episode.roles[0], count, placement)
        decoy_mask = first == ROLE_INDEX["DECOY"]
        bijection = sample_bijection(palette_seed * 31 + episode.layout)
        roles, frames = [], []
        for t in range(episode.length):
            grid = episode.roles[t].copy()
            # Only repaint cells the agent is not standing on, so occlusion still wins.
            paint = decoy_mask & (grid == ROLE_INDEX["EMPTY"])
            grid[paint] = ROLE_INDEX["DECOY"]
            roles.append(grid)
            frames.append(O.render_roles(grid, bijection,
                                         episode.stripe if t == 0 else None))
        out.append(StratumEpisode(
            base=episode, stratum=stratum, decoy_cells=int(decoy_mask.sum()),
            roles=np.stack(roles), frames=np.stack(frames), bijection=bijection))
    return out


# ---- public per-colour tokens -------------------------------------------------------


def colour_tokens(episode: StratumEpisode, t: int) -> tuple[np.ndarray, np.ndarray]:
    """One token per distinct colour present, built from PUBLIC quantities only.

    Fields: normalised RGB, cell count, spatial mean and spread, whether the colour's
    cell set moved since the previous frame, and whether the agent-candidate motion was
    blocked. No role name, no palette id, no evaluator state.
    """
    frame = episode.frames[t][::2, ::2, :]              # (12,12,3) cell grid
    previous = episode.frames[max(t - 1, 0)][::2, ::2, :]
    flat = frame.reshape(-1, 3)
    colours, inverse = np.unique(flat, axis=0, return_inverse=True)
    tokens = []
    for k, colour in enumerate(colours):
        mask = (inverse == k).reshape(GRID, GRID)
        rows, cols = np.nonzero(mask)
        before = np.all(previous == colour, axis=-1)
        tokens.append(np.concatenate([
            colour.astype(np.float32) / 255.0,
            [mask.sum() / (GRID * GRID)],
            [rows.mean() / GRID, cols.mean() / GRID],
            [rows.std() / GRID, cols.std() / GRID],
            [float(mask.sum() != before.sum())],
            [float(np.logical_xor(mask, before).sum()) / (GRID * GRID)],
        ]).astype(np.float32))
    return np.stack(tokens), colours


TOKEN_WIDTH = 3 + 1 + 2 + 2 + 1 + 1
