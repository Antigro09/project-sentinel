"""Scale 1A-0R-O3. Semantic scenarios that can be re-rendered under any palette.

O2 measured one unseen palette at 1.0000 route parity and three at 0.5346 on the same
kind of semantic content, and treated it as a calibration question. Before that can be
asked, the pipeline has to be shown to be a function of the SEMANTICS and not of the
colours -- so this module separates the two.

A `Scenario` holds the palette-free part of an episode group: role grids, actions,
positions, events, stripe. Rendering is then a pure lookup, and the same semantic history
can be pushed through the whole pipeline under hundreds of palettes with nothing else
varying. That is what makes an orbit test possible at all.

    imported by o3_orbit.py, o3_population.py, o3_calibration.py
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import o2_core as C                                                      # noqa: E402
import o2_models as M                                                    # noqa: E402
from o2_core import COLOUR_POOL, GRID, MAX_COLOURS, N_ROLES, TOKEN_WIDTH  # noqa: E402

N_POOL = len(COLOUR_POOL)


@dataclass
class Scenario:
    """One semantic episode group, free of any appearance."""
    label: str
    calibration: list[C.O2Episode]
    transfer: list[C.O2Episode]

    @property
    def semantic_digest(self) -> str:
        h = hashlib.sha256()
        for episode in self.calibration + self.transfer:
            h.update(episode.roles.tobytes())
            h.update(np.asarray(episode.actions).tobytes())
            h.update(np.asarray(episode.event).tobytes())
        return h.hexdigest()[:16]


def build_scenario(label: str, calibration_layouts, transfer_layouts, seed: int,
                   stratum: str = "COUNT_COLLISION") -> Scenario:
    """Collected once under an arbitrary bijection; only the SEMANTICS are kept."""
    reference = C.sample_bijection(1)
    return Scenario(
        label=label,
        calibration=C.collect(calibration_layouts, reference, stratum, 9, seed=seed,
                              policy="uniform"),
        transfer=C.collect(transfer_layouts, reference, stratum, 9, seed=seed + 91,
                           policy="uniform"))


def render_cells(episode: C.O2Episode, bijection: np.ndarray) -> np.ndarray:
    """(T, 12, 12, 3) colour grids for one episode under one palette."""
    return np.stack([
        C.cells_from_roles(episode.roles[t], bijection,
                           episode.stripe if t == 0 else None)
        for t in range(episode.length)])


def episode_block(episode: C.O2Episode, bijection: np.ndarray,
                  registry: C.ColourRegistry):
    """Per-step tokens and colour-index grids, without ever building a 24x24 frame."""
    cells = render_cells(episode, bijection)
    tokens = np.zeros((episode.length, MAX_COLOURS, TOKEN_WIDTH), np.float32)
    index = np.zeros((episode.length, GRID, GRID), np.int64)
    for t in range(episode.length):
        index[t] = C.cell_index(cells[t], registry)
    for t in range(1, episode.length):
        tokens[t] = C.pair_tokens(cells[t - 1], cells[t], int(episode.actions[t - 1]),
                                  registry)
    return tokens, index


def scenario_block(scenario: Scenario, bijection: np.ndarray,
                   registry: C.ColourRegistry, view: str, history_steps: int = 32,
                   contested_only: bool = True) -> dict[str, np.ndarray]:
    """The complete memory input for one scenario under one palette."""
    steps = []
    for episode in scenario.calibration:
        tokens, _ = episode_block(episode, bijection, registry)
        steps.extend(tokens[t] for t in range(1, episode.length))
    history = np.zeros((history_steps, MAX_COLOURS, TOKEN_WIDTH), np.float32)
    mask = np.zeros(history_steps, np.float32)
    take = min(len(steps), history_steps)
    history[:take] = np.stack(steps[:take])
    mask[:take] = 1.0
    history = M.mask_view(history, view)

    tokens, before, after, event, meta = [], [], [], [], []
    for episode in scenario.transfer:
        block, index = episode_block(episode, bijection, registry)
        for t in range(1, episode.length):
            entered = int(episode.entered_role[t])
            if contested_only and entered not in (C.SWITCH, C.DECOY):
                continue
            tokens.append(block[t])
            before.append(index[t - 1])
            after.append(index[t])
            event.append(episode.event[t])
            meta.append((episode.layout, t, entered))
    if not tokens:
        raise ValueError("scenario has no contested transfer rows")
    pairs = {"tokens": M.mask_view(np.stack(tokens).astype(np.float32), view),
             "before_index": np.stack(before), "after_index": np.stack(after),
             "event": np.array(event, np.float32)}
    sequence, seq_mask, b, a, y = C.sequence_dataset(pairs, history, mask)
    return {"sequence": sequence, "mask": seq_mask, "before": b, "after": a,
            "event": y, "meta": np.array(meta, np.int64)}


# ---- palette orbits ------------------------------------------------------------------


def identity_bijection() -> np.ndarray:
    return np.arange(N_ROLES)


def base_role_orbit() -> list[np.ndarray]:
    """All 720 permutations of the six base roles over the first six pool colours, with
    DECOY held on a fixed seventh colour. An exhaustive subgroup, not a sample."""
    out = []
    for order in itertools.permutations(range(6)):
        bijection = np.zeros(N_ROLES, dtype=np.int64)
        bijection[:6] = order
        bijection[C.DECOY] = 6
        out.append(bijection)
    return out


def random_orbit(count: int, seed: int = 3_001) -> list[np.ndarray]:
    """Random injections of seven roles into the eight-colour pool."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(N_POOL)[:N_ROLES].astype(np.int64) for _ in range(count)]


def entropy_bits(assignment: np.ndarray) -> float:
    live = assignment[assignment.sum(axis=-1) > 1e-6]
    if not len(live):
        return float("nan")
    return float(np.mean([-np.sum(r * np.log2(np.maximum(r, 1e-12))) for r in live]))


def semantic_assignment(assignment: np.ndarray, bijection: np.ndarray,
                        registry: C.ColourRegistry) -> np.ndarray:
    """Map a colour-slot assignment back into SEMANTIC role space.

    Row r of the result is what the model believes about the colour that this palette
    gave to true role r. Two palettes agree semantically exactly when these agree, which
    is the only comparison an orbit test may make.
    """
    out = np.zeros((N_ROLES, N_ROLES), np.float32)
    for role in range(N_ROLES):
        out[role] = assignment[registry.of(COLOUR_POOL[bijection[role]])]
    return out


def digest_block(block: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("sequence", "mask", "before", "after", "event"):
        h.update(np.ascontiguousarray(block[key]).tobytes())
    return h.hexdigest()[:16]
