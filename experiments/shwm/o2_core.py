"""Scale 1A-0R-O2. Shared machinery: palettes that do not depend on the layout,
per-colour tokens with a public interaction block, and calibration/transfer episodes.

Three things phase O1 could not express are built here.

1. A palette that is a property of the EPISODE GROUP, not of the layout. `p_core`
   derived the bijection from `palette_seed * 31 + layout`, so two layouts never shared
   a palette and "the same hidden palette across calibration and transfer" was not
   representable. Here the bijection is passed in.

2. An INTERACTION block in the per-colour token. O1's token carried colour, count,
   spatial moments and two weak motion fields -- and `moved` was a COUNT-change flag, so
   an agent stepping from one empty cell to another set it to zero. Nothing in the token
   said which colour the agent stepped ONTO, which is half of the event definition. The
   three interaction fields say it, and they are public: the moving singleton is found
   by differencing two frames, no role name required.

3. Calibration and transfer as separate segments over disjoint layouts under one
   palette, with the transfer pairs restricted to the set that is PROVABLY ambiguous
   without the calibration history -- the agent stepped onto a cell whose colour, under
   COUNT_COLLISION, could be SWITCH or DECOY with identical pixels forever after.

    imported by every o2_*.py module
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import o_core as O                                                       # noqa: E402
from o_core import COLOUR_POOL, GRID, ROLES as BASE_ROLES                # noqa: E402
from structured_calibration import DELTAS_BY_INDEX                       # noqa: E402

ROLES = BASE_ROLES + ("DECOY",)
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}
N_ROLES = len(ROLES)
EMPTY, WALL, SWITCH, GOAL_ALPHA, GOAL_BETA, AGENT, DECOY = range(N_ROLES)
STRATA = ("COUNT_INFORMATIVE", "COUNT_VARIED", "COUNT_COLLISION")
SWITCH_COUNT = 7

# ---- token layout -------------------------------------------------------------------
# Every field is computable from the pixels and the action alone. Asserted, not assumed:
# `check_token_layout()` is called at import and pinned by a test.
RGB = slice(0, 3)
COUNT = slice(3, 4)
MOMENTS = slice(4, 8)
MOTION = slice(8, 10)
INTERACT = slice(10, 13)
GLOBAL = slice(13, 22)
TOKEN_WIDTH = 22
MAX_COLOURS = 10

FIELD_NAMES = {
    "RGB": RGB, "COUNT": COUNT, "MOMENTS": MOMENTS, "MOTION": MOTION,
    "INTERACT": INTERACT, "GLOBAL": GLOBAL,
}


def check_token_layout() -> None:
    covered = np.zeros(TOKEN_WIDTH, dtype=int)
    for part in FIELD_NAMES.values():
        covered[part] += 1
    assert (covered == 1).all(), f"token fields overlap or leave a gap: {covered}"


check_token_layout()


def decoy_count(stratum: str, rng, count_range: tuple[int, int] | None = None) -> int:
    """Cells repainted as DECOY. DECOY is semantically identical to EMPTY, so this
    changes how informative a cell count is and changes nothing else."""
    if stratum == "COUNT_INFORMATIVE":
        return 0
    if stratum == "COUNT_COLLISION":
        return SWITCH_COUNT
    low, high = count_range if count_range else (4, 11)
    return int(rng.integers(low, high))


def sample_bijection(seed: int) -> np.ndarray:
    """role -> COLOUR_POOL index. Injective, and drawn WITHOUT reference to the role."""
    rng = np.random.default_rng(seed)
    return rng.permutation(len(COLOUR_POOL))[:N_ROLES]


# ---- episodes -----------------------------------------------------------------------


def decoy_placement(level, start: tuple[int, int], n_decoy: int, rng) -> np.ndarray:
    """Where the decoys go, and why it decides whether the collision is real.

    DECOY must be exchangeable with SWITCH in every public statistic except behaviour.
    Three weaker rules were measured first and each failed a different way:

      uniform over empty cells   the switch generator clusters its seven cells within
                                 three steps of the start, so a test population had 51
                                 switch entries against 5 decoy entries. A count-only
                                 lookup then reads P(event | count 7) = 0.897 and scores
                                 0.975 -- not because counts name the role, but because
                                 the ambiguous case almost never happens.
      nearest empty cells        balances the visit rate but puts decoys CLOSER to the
                                 start than switches.
      the generator's own band   falls back to the whole candidate list whenever the
                                 band is exhausted, which it usually is once seven
                                 switches have taken it: measured mean distance from
                                 start 5.35 for decoys against 2.27 for switches.

    The rule below matches the DISTANCE MULTISET: for each switch at distance d it
    places a decoy at distance d, falling back to the nearest unused distance only when
    that shell is full. Cardinality was already exact; this makes the spatial marginal
    match too, so no count and no spatial moment separates the two roles.
    """
    mask = np.zeros((GRID, GRID), dtype=bool)
    if n_decoy <= 0:
        return mask
    walls = np.asarray(level.walls, dtype=bool)
    taken = {tuple(int(v) for v in level.start)}
    taken |= {tuple(int(v) for v in c) for c in level.markers.values()}
    switches = [tuple(int(v) for v in c) for c in level.switches]
    taken |= set(switches)

    def distance(cell):
        return abs(cell[0] - start[0]) + abs(cell[1] - start[1])

    available: dict[int, list] = {}
    for row in range(GRID):
        for column in range(GRID):
            cell = (row, column)
            if walls[cell] or cell in taken:
                continue
            available.setdefault(distance(cell), []).append(cell)
    for shell in available.values():
        rng.shuffle(shell)

    wanted = [distance(c) for c in switches[:n_decoy]]
    while len(wanted) < n_decoy:
        wanted.append(wanted[len(wanted) % max(len(switches), 1)]
                      if switches else 1)
    for target in sorted(wanted):
        shells = sorted(available, key=lambda d: (abs(d - target), d))
        for shell in shells:
            if available[shell]:
                mask[available[shell].pop()] = True
                break
    return mask


@dataclass
class O2Episode:
    layout: int
    stratum: str
    palette_id: int
    bijection: np.ndarray            # (N_ROLES,) role -> pool index
    frames: np.ndarray               # (T, 24, 24, 3) uint8   PUBLIC
    roles: np.ndarray                # (T, 12, 12)            evaluator-only
    actions: np.ndarray              # (T,)                   PUBLIC
    positions: np.ndarray            # (T, 2)                 evaluator-only
    polarity: np.ndarray             # (T,)                   evaluator-only
    event: np.ndarray                # (T,)                   PUBLIC label
    entered_role: np.ndarray         # (T,) role stepped onto, -1 if no move
    goal_marker: str
    goal_cells: dict[str, tuple[int, int]]
    stripe: int
    decoy_cells: int

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def cells(self) -> np.ndarray:
        """The 12x12 colour grid. The renderer paints solid CELL x CELL blocks, so the
        stride-2 subsample is exact -- except on row 0 of the reset frame, which the
        stripe overwrites. That is a public artifact and is left in."""
        return self.frames[:, ::2, ::2, :]


def collect(layouts: Sequence[int], bijection: np.ndarray, stratum: str,
            steps: int = 9, seed: int = 11, policy: str = "uniform",
            goal_draw: int | None = None, decoy_seed: int | None = None,
            count_range: tuple[int, int] | None = None,
            per_frame_seed: int | None = None) -> list[O2Episode]:
    """Episodes under ONE palette across every layout given."""
    from sentinel.env.adapters.procedural_visual_v2 import (
        ACTIONS, ProceduralVisualV2Adapter)
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="o2")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    rng = np.random.default_rng(seed)
    palette_id = int(hashlib.sha256(np.asarray(bijection).tobytes())
                     .hexdigest()[:8], 16)
    out: list[O2Episode] = []
    for layout in layouts:
        dynamic = "base" if goal_draw is None else f"goal:{goal_draw}"
        adapter.reset(layout, dynamic)
        level = adapter._require()
        goal_cells = {m: tuple(int(v) for v in level.markers[m])
                      for m in ("alpha", "beta")}
        target = goal_cells[adapter._goal_marker]
        stripe = int(level.initial_polarity)
        switches = {tuple(int(v) for v in c) for c in level.switches}

        placement = np.random.default_rng((decoy_seed if decoy_seed is not None
                                           else seed) * 1_000_003 + layout)
        n_decoy = decoy_count(stratum, placement, count_range)
        decoy_mask = decoy_placement(level, tuple(int(v) for v in level.start),
                                     n_decoy, placement)

        frames, roles, actions, positions, polarity = [], [], [], [], []
        for step in range(steps):
            truth = adapter.snapshot().reveal("evaluator")
            position = tuple(int(v) for v in truth["position"])
            grid = O.role_grid(level, position)
            grid[decoy_mask & (grid == EMPTY)] = DECOY
            current = (bijection if per_frame_seed is None
                       else sample_bijection(per_frame_seed * 1_000 + step))
            frames.append(O.render_roles(grid, current,
                                         stripe if step == 0 else None))
            roles.append(grid)
            positions.append(position)
            polarity.append(int(truth["polarity"]))
            if policy == "uniform":
                action = int(rng.integers(0, len(ACTIONS)))
            elif policy == "switch_seeking":
                action = _toward(position, switches, rng, len(ACTIONS))
            else:
                action = _toward(position, {target}, rng, len(ACTIONS))
            actions.append(action)
            if adapter.step(action, gate.authorize_evaluator(action, "o2")).terminated:
                break
        if len(frames) < 3:
            continue
        length = len(frames)
        event = np.zeros(length, dtype=np.float32)
        entered = np.full(length, -1, dtype=np.int64)
        for t in range(1, length):
            if positions[t] != positions[t - 1]:
                entered[t] = int(roles[t - 1][positions[t]])
                event[t] = float(positions[t] in switches)
        out.append(O2Episode(
            layout=layout, stratum=stratum, palette_id=palette_id,
            bijection=np.asarray(bijection), frames=np.stack(frames),
            roles=np.stack(roles), actions=np.array(actions[:length]),
            positions=np.array(positions), polarity=np.array(polarity), event=event,
            entered_role=entered, goal_marker=adapter._goal_marker,
            goal_cells=goal_cells, stripe=stripe,
            decoy_cells=int(decoy_mask.sum())))
    return out


def _toward(position, targets, rng, n_actions: int) -> int:
    if rng.random() < 0.25 or not targets:
        return int(rng.integers(0, n_actions))
    best, choice = None, 0
    for index, (dr, dc) in enumerate(DELTAS_BY_INDEX):
        cell = (position[0] + dr, position[1] + dc)
        score = min(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) for t in targets)
        if best is None or score < best:
            best, choice = score, index
    return choice


# ---- public per-colour tokens -------------------------------------------------------


class ColourRegistry:
    """A stable public index for each RGB triple, shared across every episode that uses
    the same palette. This is what lets a memory built during calibration be addressed
    during transfer: the colour value is public, so matching on it is not a leak."""

    def __init__(self) -> None:
        self.order: list[tuple[int, int, int]] = []
        self.index: dict[tuple[int, int, int], int] = {}

    def add(self, colour) -> int:
        key = tuple(int(v) for v in colour)
        if key not in self.index:
            if len(self.order) >= MAX_COLOURS:
                raise ValueError(f"more than {MAX_COLOURS} distinct colours")
            self.index[key] = len(self.order)
            self.order.append(key)
        return self.index[key]

    def of(self, colour) -> int:
        return self.index[tuple(int(v) for v in colour)]

    def scan(self, episodes: Iterable[O2Episode]) -> "ColourRegistry":
        for episode in episodes:
            for colour in np.unique(episode.cells.reshape(-1, 3), axis=0):
                self.add(colour)
        return self

    def permuted(self, seed: int) -> "ColourRegistry":
        """A registry whose indices are shuffled. Used by the wrong-pairing control:
        the memory is then attached to the wrong colours."""
        other = ColourRegistry()
        other.order = list(self.order)
        rng = np.random.default_rng(seed)
        values = rng.permutation(len(self.order))
        other.index = {c: int(values[i]) for i, c in enumerate(self.order)}
        return other


def moving_singleton(before_cells: np.ndarray, after_cells: np.ndarray):
    """(vacated, occupied) from two 12x12 colour grids, or None. Public.

    Two traps make the naive version wrong, and both were measured before this rule was
    written.

    1. `before[a] == after[b]` holds in BOTH directions whenever the agent steps from one
       empty cell to another: the agent colour arrives at b and the empty colour is
       restored at a. Reading the change set in scan order therefore reversed the move on
       roughly half the steps, which put the INTERACT `entered` flag on the wrong colour.
       The mover is disambiguated by cardinality instead: the agent is a colour with
       exactly one cell in BOTH frames, while the revealed colour is either plural or
       had zero cells before.

    2. The reset frame carries the polarity stripe across the whole of row 0, so a pair
       that straddles it has twelve changed cells, not two. Requiring exactly two changes
       silently dropped every reset pair. The rule below allows any number of changes and
       identifies the move by the singleton match, which the stripe cannot fake -- the
       stripe colour covers twelve cells and row 0 is wall border, so no agent is ever
       in it.
    """
    changed = [tuple(int(v) for v in c)
               for c in np.argwhere(np.any(before_cells != after_cells, axis=-1))]
    if len(changed) < 2:
        return None
    before_flat = before_cells.reshape(-1, 3)
    after_flat = after_cells.reshape(-1, 3)

    def count(flat, colour):
        return int(np.all(flat == colour, axis=-1).sum())

    found = []
    for a in changed:
        colour = before_cells[a]
        if count(before_flat, colour) != 1 or count(after_flat, colour) != 1:
            continue
        landing = [b for b in changed
                   if b != a and np.array_equal(after_cells[b], colour)]
        if len(landing) != 1:
            continue
        b = landing[0]
        if abs(b[0] - a[0]) + abs(b[1] - a[1]) != 1:
            continue
        found.append((a, b))
    return found[0] if len(found) == 1 else None


def is_reset_pair(before_cells: np.ndarray) -> bool:
    """Is this pair's BEFORE frame the reset frame? Derived from pixels, not from t.

    The polarity stripe paints the whole of pixel row 0, so the top cell row of a reset
    frame is a single colour that appears nowhere else in the scene. Without this flag a
    concatenated calibration history has a spurious direction reversal at every episode
    boundary, and the recurrent memory attributes it to whatever colour happened to be
    entered last. The flag is public: a viewer can see the stripe.
    """
    top = np.unique(before_cells[0].reshape(-1, 3), axis=0)
    if len(top) != 1:
        return False
    rest = np.unique(before_cells[1:].reshape(-1, 3), axis=0)
    return not any(np.array_equal(top[0], c) for c in rest)


def pair_tokens(before_cells: np.ndarray, after_cells: np.ndarray, action: int,
                registry: ColourRegistry) -> np.ndarray:
    """(MAX_COLOURS, TOKEN_WIDTH). Public quantities only.

    The three INTERACTION fields are the outcomes of the step, per colour:

        entered   this colour was under the cell the mover arrived at
        left      this colour was at the cell the mover departed from -- so it is the
                  mover's own colour, which is how AGENT becomes bindable at all
        revealed  this colour is at the departed cell AFTERWARDS -- what the mover had
                  been occluding

    None of them says whether the entered colour is a SWITCH. That is the whole point:
    interaction localises the event's ARGUMENTS and leaves its PREDICATE to the binding.
    """
    tokens = np.zeros((MAX_COLOURS, TOKEN_WIDTH), dtype=np.float32)
    move = moving_singleton(before_cells, after_cells)
    vacated, occupied = move if move else (None, None)
    delta = ((occupied[0] - vacated[0], occupied[1] - vacated[1]) if move else (0, 0))

    glob = np.zeros(9, dtype=np.float32)
    glob[action] = 1.0
    glob[4], glob[5] = float(delta[0]), float(delta[1])
    glob[6] = float(move is not None)
    glob[7] = float(is_reset_pair(before_cells))
    # The displacement SIGN relative to the action. A deterministic function of two
    # fields already in the block, added as its own scalar because the quantity a
    # persistent memory has to track is "did the sign change since this colour was
    # entered", and asking a recurrent cell to first multiply an action one-hot by a
    # displacement is a change of basis it should not have to discover.
    reference = DELTAS_BY_INDEX[action]
    if move is not None:
        glob[8] = (1.0 if delta == reference
                   else -1.0 if delta == (-reference[0], -reference[1]) else 0.0)

    for colour in np.unique(before_cells.reshape(-1, 3), axis=0):
        k = registry.of(colour)
        mask = np.all(before_cells == colour, axis=-1)
        after_mask = np.all(after_cells == colour, axis=-1)
        rows, cols = np.nonzero(mask)
        token = tokens[k]
        token[RGB] = colour.astype(np.float32) / 255.0
        token[COUNT] = mask.sum() / (GRID * GRID)
        token[MOMENTS] = [rows.mean() / GRID, cols.mean() / GRID,
                          rows.std() / GRID, cols.std() / GRID]
        token[MOTION] = [float(mask.sum() != after_mask.sum()),
                         float(np.logical_xor(mask, after_mask).sum()) / (GRID * GRID)]
        token[INTERACT] = [float(move is not None and mask[occupied]),
                           float(move is not None and mask[vacated]),
                           float(move is not None and after_mask[vacated])]
        token[GLOBAL] = glob
    # A colour present only in the AFTER frame still needs its GLOBAL block, so that a
    # view restricted to GLOBAL is not silently all-zero for it.
    for colour in np.unique(after_cells.reshape(-1, 3), axis=0):
        k = registry.of(colour)
        if not tokens[k].any():
            tokens[k][RGB] = colour.astype(np.float32) / 255.0
            tokens[k][GLOBAL] = glob
    return tokens


def cell_index(cells: np.ndarray, registry: ColourRegistry) -> np.ndarray:
    """(12, 12) colour index grid."""
    out = np.zeros((GRID, GRID), dtype=np.int64)
    for colour in np.unique(cells.reshape(-1, 3), axis=0):
        out[np.all(cells == colour, axis=-1)] = registry.of(colour)
    return out


def episode_stream(episode: O2Episode, registry: ColourRegistry):
    """Per-step tokens and colour-index grids for one episode."""
    cells = episode.cells
    length = episode.length
    tokens = np.zeros((length, MAX_COLOURS, TOKEN_WIDTH), dtype=np.float32)
    index = np.zeros((length, GRID, GRID), dtype=np.int64)
    for t in range(length):
        index[t] = cell_index(cells[t], registry)
    for t in range(1, length):
        tokens[t] = pair_tokens(cells[t - 1], cells[t], int(episode.actions[t - 1]),
                                registry)
    return tokens, index


def digest_episodes(episodes: Iterable[O2Episode]) -> str:
    h = hashlib.sha256()
    for e in episodes:
        h.update(e.frames.tobytes())
        h.update(np.asarray(e.actions).tobytes())
        h.update(np.asarray(e.bijection).tobytes())
    return h.hexdigest()[:16]


def manifest(episodes: Sequence[O2Episode], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "episodes": len(episodes),
        "layouts": sorted({e.layout for e in episodes}),
        "palette_ids": sorted({e.palette_id for e in episodes}),
        "strata": sorted({e.stratum for e in episodes}),
        "decoy_cells": sorted({e.decoy_cells for e in episodes}),
        "steps": sorted({e.length for e in episodes}),
        "event_rate": float(np.mean([e.event[1:].mean() for e in episodes])),
        "entered_role_counts": {
            ROLES[r]: int(sum(int((e.entered_role[1:] == r).sum()) for e in episodes))
            for r in range(N_ROLES)},
        "steps_without_a_move": int(sum(int((e.entered_role[1:] < 0).sum())
                                        for e in episodes)),
        "digest": digest_episodes(episodes),
    }


# ---- section K: the authored goal-grounding demonstration ---------------------------


def goal_demonstration(layout: int, bijection: np.ndarray, marker: str,
                       stratum: str = "COUNT_COLLISION", decoy_seed: int = 5,
                       max_steps: int = 64) -> dict[str, Any] | None:
    """A calibration episode that ENDS with the agent standing on the named marker, and
    that RETAINS that final frame.

    No ordinary episode in this project has ever depicted it. The adapter terminates the
    instant the agent reaches its goal marker, and every collector appends the frame
    BEFORE stepping, so the terminal frame is discarded. The public record therefore
    never shows the goal marker occupied, and language naming it can bind nothing. That
    is a property of the recording rule, not of language, and it is why section K has to
    author a demonstration rather than mine one.

    What is public here is only: the frame sequence, the action sequence, and the
    sentence naming the marker. The planner below reads hidden polarity, because a
    demonstration is authored by the environment, not inferred by the learner.
    """
    from collections import deque

    from sentinel.env.adapters.procedural_visual_v2 import (
        DELTAS, ProceduralVisualV2Adapter)
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="o2-demo")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    adapter.reset(layout)
    level = adapter._require()
    target = tuple(int(v) for v in level.markers[marker])
    switches = {tuple(int(v) for v in c) for c in level.switches}
    start = (tuple(int(v) for v in level.start), int(level.initial_polarity))

    # Breadth-first over (cell, polarity): the action delta is mirrored by polarity and
    # polarity flips on a switch, so the graph is not the plain grid.
    previous: dict[tuple, tuple] = {start: (None, None)}
    queue = deque([start])
    goal_state = None
    while queue:
        state = queue.popleft()
        if state[0] == target:
            goal_state = state
            break
        (row, column), polarity = state
        for action, (dr, dc) in DELTAS.items():
            if polarity:
                dr, dc = -dr, -dc
            cell = (row + dr, column + dc)
            if not (0 <= cell[0] < GRID and 0 <= cell[1] < GRID):
                continue
            if level.walls[cell]:
                continue
            nxt = (cell, polarity ^ 1 if cell in switches else polarity)
            if nxt not in previous:
                previous[nxt] = (state, action)
                queue.append(nxt)
    if goal_state is None:
        return None
    plan: list[int] = []
    state = goal_state
    while previous[state][0] is not None:
        state, action = previous[state]
        plan.append(action)
    plan.reverse()
    if not plan or len(plan) > max_steps:
        return None

    placement = np.random.default_rng(decoy_seed * 1_000_003 + layout)
    decoy_mask = decoy_placement(level, tuple(int(v) for v in level.start),
                                 decoy_count(stratum, placement), placement)

    adapter.reset(layout)
    stripe = int(level.initial_polarity)
    frames, roles, positions = [], [], []
    for step in range(len(plan) + 1):
        position = tuple(int(v) for v in
                         adapter.snapshot().reveal("evaluator")["position"])
        grid = O.role_grid(level, position)
        grid[decoy_mask & (grid == EMPTY)] = DECOY
        frames.append(O.render_roles(grid, bijection, stripe if step == 0 else None))
        roles.append(grid)
        positions.append(position)
        if step == len(plan):
            break
        action = plan[step]
        adapter.step(action, gate.authorize_evaluator(action, "o2-demo"))
    if positions[-1] != target:
        return None
    marker_colour = tuple(int(v) for v in
                          O.render_roles(roles[-2], bijection, None)[
                              target[0] * 2, target[1] * 2])
    return {
        "layout": layout, "marker": marker,
        "instruction": f"reach the {marker} marker",
        "frames": np.stack(frames), "roles": np.stack(roles),
        "actions": np.array(plan, dtype=np.int64),
        "positions": np.array(positions), "target_cell": target,
        "terminal_frame_retained": True,
        "named_marker_colour": marker_colour,
        "steps": len(plan),
    }


# ---- datasets -----------------------------------------------------------------------


def pair_dataset(episodes: Sequence[O2Episode], registry: ColourRegistry,
                 keep=None) -> dict[str, np.ndarray]:
    """One row per frame pair. `keep(episode, t)` selects a sub-population."""
    tokens, before, after, event, meta = [], [], [], [], []
    for episode in episodes:
        stream, index = episode_stream(episode, registry)
        for t in range(1, episode.length):
            if keep is not None and not keep(episode, t):
                continue
            tokens.append(stream[t])
            before.append(index[t - 1])
            after.append(index[t])
            event.append(episode.event[t])
            meta.append((episode.layout, episode.palette_id, t,
                         int(episode.entered_role[t])))
    if not tokens:
        return {"tokens": np.zeros((0, MAX_COLOURS, TOKEN_WIDTH), np.float32),
                "before_index": np.zeros((0, GRID, GRID), np.int64),
                "after_index": np.zeros((0, GRID, GRID), np.int64),
                "event": np.zeros(0, np.float32),
                "meta": np.zeros((0, 4), np.int64)}
    return {"tokens": np.stack(tokens).astype(np.float32),
            "before_index": np.stack(before), "after_index": np.stack(after),
            "event": np.array(event, np.float32),
            "meta": np.array(meta, dtype=np.int64)}


def as_block(data: dict[str, np.ndarray]) -> tuple:
    return (data["tokens"], data["before_index"], data["after_index"], data["event"])


def sequence_dataset(pairs: dict[str, np.ndarray], history: np.ndarray,
                     history_mask: np.ndarray) -> tuple:
    """Prepend a shared calibration history to every transfer pair.

    `history` is (T, K, TOKEN_WIDTH) and is the SAME for every row that shares a palette
    group, which is the construction the specification asks for: one hidden palette,
    calibrated once, reused across new layouts.
    """
    n = len(pairs["event"])
    steps = len(history) + 1
    sequence = np.zeros((n, steps, MAX_COLOURS, TOKEN_WIDTH), np.float32)
    mask = np.zeros((n, steps), np.float32)
    sequence[:, :len(history)] = history[None]
    mask[:, :len(history)] = history_mask[None]
    sequence[:, -1] = pairs["tokens"]
    mask[:, -1] = 1.0
    return (sequence, mask, pairs["before_index"], pairs["after_index"], pairs["event"])


# ---- exact identifiability over seven roles -----------------------------------------
# o_identifiability enumerates 720 permutations of six roles. DECOY makes it 5040, and
# the cardinality rule changes: DECOY's count is not a generator constant, so nothing
# bounds it, and under COUNT_COLLISION it satisfies SWITCH's own count constraint. That
# is what makes the collision exact rather than merely likely. Every other legality rule
# -- motion, static scene, switch consistency, wall grounding -- is reused unchanged from
# o_identifiability, because the role indices of the six base roles are identical.


def cardinality_legal7(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=N_ROLES)
    if counts[AGENT] != 1:
        return False
    if counts[GOAL_ALPHA] not in (0, 1) or counts[GOAL_BETA] not in (0, 1):
        return False
    if counts[SWITCH] not in (SWITCH_COUNT - 1, SWITCH_COUNT):
        return False
    return counts[WALL] >= 1 and counts[EMPTY] >= 1


def _pair_legal(before: np.ndarray, after: np.ndarray, action: int) -> bool:
    import o_identifiability as ident
    return (cardinality_legal7(after)
            and ident.static_scene_legal(before, after)
            and ident.motion_legal(before, after, action))


def survivors_over(episodes: Sequence[O2Episode], steps: int | None = None,
                   candidates: Sequence[tuple[int, ...]] | None = None
                   ) -> list[tuple[int, ...]]:
    """Every role permutation whose permuted trajectory is legal on ALL of `episodes`.

    Intersecting across episodes is the point: one hidden palette shared by several
    layouts is exactly a constraint that accumulates, which is what calibration means.
    """
    import itertools

    import o_identifiability as ident

    if candidates is None:
        candidates = list(itertools.permutations(range(N_ROLES)))
    keep = [pi for pi in candidates
            if cardinality_legal7(np.array(pi, np.int64)[episodes[0].roles[0]])]
    for episode in episodes:
        limit = episode.length - 1 if steps is None else min(steps, episode.length - 1)
        actions = episode.actions
        survivors = []
        for pi in keep:
            grids = np.array(pi, np.int64)[episode.roles]
            if not cardinality_legal7(grids[0]):
                continue
            ok = all(_pair_legal(grids[t - 1], grids[t], int(actions[t - 1]))
                     for t in range(1, limit + 1))
            if ok:
                ok = ident.switch_consistency_legal(grids[:limit + 1],
                                                    actions[:limit + 1])
            if ok:
                survivors.append(pi)
        keep = survivors
        if not keep:
            break
    return keep


def event_quotient_mass(survivors: Sequence[tuple[int, ...]]) -> float:
    """Posterior mass on the permutations that agree with the truth about the event."""
    if not survivors:
        return float("nan")
    return float(np.mean([pi[AGENT] == AGENT and pi[SWITCH] == SWITCH
                          for pi in survivors]))


def exchangeability(episodes: Sequence[O2Episode]) -> dict[str, Any]:
    """How close SWITCH and DECOY are to being the same thing to a viewer.

    Cardinality is exact by construction under COUNT_COLLISION. The spatial marginal is
    matched shell by shell but cannot be made exact: the switch generator claims the
    nearest cells first and a distance shell can run out. What remains is reported
    rather than assumed away, and the empirical consequence is the spatial-moments arm's
    accuracy on contested rows.
    """
    from collections import Counter

    counts = {"SWITCH": [], "DECOY": []}
    distance = {"SWITCH": [], "DECOY": []}
    for episode in episodes:
        start = tuple(int(v) for v in episode.positions[0])
        grid = episode.roles[0]
        for name, role in (("SWITCH", SWITCH), ("DECOY", DECOY)):
            cells = np.argwhere(grid == role)
            counts[name].append(len(cells))
            distance[name].extend(abs(int(c[0]) - start[0]) + abs(int(c[1]) - start[1])
                                  for c in cells)
    out: dict[str, Any] = {}
    for name in ("SWITCH", "DECOY"):
        out[name] = {
            "mean_cells": float(np.mean(counts[name])) if counts[name] else 0.0,
            "mean_distance_from_start": (float(np.mean(distance[name]))
                                         if distance[name] else float("nan")),
            "distance_histogram": {str(k): v for k, v in
                                   sorted(Counter(distance[name]).items())}}
    out["cardinality_identical"] = bool(
        abs(out["SWITCH"]["mean_cells"] - out["DECOY"]["mean_cells"]) < 1e-9)
    out["distance_gap"] = float(abs(out["SWITCH"]["mean_distance_from_start"]
                                    - out["DECOY"]["mean_distance_from_start"]))
    return out


def canonical_registry() -> ColourRegistry:
    """A fixed colour-slot order: the eight pool entries, then the two stripe colours.

    Scanning episodes assigns slots in first-seen order, which makes the slot index a
    property of the run rather than of the palette. That is harmless for a
    permutation-equivariant binder but it means two modules cannot share a memory
    without a remapping step, and a remapping step is somewhere for a bug to live.
    """
    registry = ColourRegistry()
    for colour in COLOUR_POOL:
        registry.add(colour)
    registry.add(np.array([0, 0, 0], np.uint8))
    registry.add(np.array([255, 255, 255], np.uint8))
    return registry


def cells_from_roles(grid: np.ndarray, bijection: np.ndarray,
                     stripe: int | None) -> np.ndarray:
    """The 12x12 colour grid a viewer would subsample, without building the frame.

    render_roles upsamples every cell to a CELL x CELL block and the consumers
    immediately subsample it back; on a route replay that is the dominant cost. The
    stripe is reproduced exactly: it paints pixel row 0, which is the row the stride-2
    subsample takes, so cell row 0 of a reset frame is the stripe colour.
    """
    out = COLOUR_POOL[bijection][grid]
    if stripe is not None:
        out[0, :] = 255 if stripe else 0
    return out.astype(np.uint8)
