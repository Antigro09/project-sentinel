"""Scale 1A-0R-N. Visual frame-pair datasets, public auxiliary targets, and splits.

Everything downstream consumes the SAME frozen frame pairs. A transition t is the triple
(frame_{t-1}, action_{t-1}, frame_t) and nothing else; the structured row that earlier
phases used is never handed to a visual model, only to the frozen structured ceilings.

The auxiliary targets are all deterministic functions of PUBLIC state. The one that
matters is `entered_switch`, and it is read from the frame BEFORE the move: the renderer
paints the agent over the switch beneath it, so the frame after the move cannot answer
what the agent is standing on. That occlusion is the whole reason the event is a
two-frame quantity rather than a one-frame one.

    imported by n_dataflow.py, n_interfaces.py, n_gauge.py, n_pathway.py
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m2d_core as m2d  # noqa: E402
from m2d_core import ARTIFACTS, write  # noqa: E402
from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS, CELL, GRID, ProceduralVisualV2Adapter)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from structured_calibration import DELTAS_BY_INDEX, DISPLACEMENTS, CLASSES  # noqa: E402

FRAME = GRID * CELL          # 24
N_DISPLACEMENT = len(DISPLACEMENTS)   # 5

# Layout ranges claimed by this phase. Every range used by M2C-M2F is avoided:
# 61000-61039, 63000+, 81000-81019, 83000-83019, 90000-90009, 91000-91009,
# 92000-92009, 95000-95019, 97000-97019.
TRAIN_LAYOUTS = tuple(range(110_000, 110_060))
SAME_LAYOUT_LAYOUTS = TRAIN_LAYOUTS                 # new trajectories, same layouts
HELD_OUT_LAYOUTS = tuple(range(111_000, 111_030))
APPEARANCE_SHIFT_SEED = 777_001                     # != CANONICAL_APPEARANCE_SEED
CROSSED_LAYOUTS = tuple(range(112_000, 112_030))    # held-out layout AND appearance

ALIAS_TRAIN = tuple(range(90_000, 90_010))          # frozen alias populations, reused
ALIAS_HELD_OUT = tuple(range(95_000, 95_010))


def goal_direction_policy(rng, position, goal):
    scores = [abs(goal[0] - (position[0] + dr)) + abs(goal[1] - (position[1] + dc))
              for dr, dc in DELTAS_BY_INDEX]
    return int(np.argmin(scores))


@dataclass
class VisualEpisode:
    layout: int
    appearance: int
    frames: np.ndarray        # (T, 24, 24, 3) uint8
    actions: np.ndarray       # (T,) int, actions[t] is the action taken FROM step t
    positions: np.ndarray     # (T, 2) evaluator-only, for building targets
    polarity: np.ndarray      # (T,) evaluator-only, never an input
    agent_mask: np.ndarray    # (T, 12, 12) float32
    switch_mask: np.ndarray   # (T, 12, 12) float32
    displacement: np.ndarray  # (T,) int, class of position[t]-position[t-1]; 0 at t=0
    entered_switch: np.ndarray  # (T,) float32
    event: np.ndarray         # (T,) float32, 0 at t=0
    stripe_state: np.ndarray  # (T,) float32, the rendered stripe value where drawn
    is_reset: np.ndarray      # (T,) float32
    goal_tokens: np.ndarray   # (G,) int, constant within an episode
    goal_text: str

    @property
    def length(self) -> int:
        return len(self.frames)


def displacement_class(previous: tuple[int, int], current: tuple[int, int]) -> int:
    delta = (current[0] - previous[0], current[1] - previous[1])
    return DISPLACEMENTS.index(delta) if delta in DISPLACEMENTS else N_DISPLACEMENT - 1


def collect_visual(layouts: Sequence[int], trajectories: int, steps: int,
                   appearance: int = CANONICAL_APPEARANCE_SEED, seed: int = 11,
                   policy: str = "uniform", epsilon: float = 0.25,
                   goal_draw: int | None = None) -> list[VisualEpisode]:
    """Frames plus every PUBLIC auxiliary target, and the evaluator fields the targets
    are built from. Nothing evaluator-only ever reaches a model input; it is carried so
    the masks and the event label can be constructed and so strata can be reported."""
    from sentinel.wm.packet import MAX_GOAL_TOKENS, tokenise_goal
    from alias_audit import VOCABULARY

    gate = AuthorityGate(gate_id="n-visual")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    out: list[VisualEpisode] = []
    for layout in layouts:
        for _ in range(trajectories):
            adapter.reset(layout, f"appearance:{appearance}")
            if goal_draw is not None:
                # The goal is a SEPARATE dynamic from layout/appearance/phase, so the
                # same rendered world can carry either language goal. That is what makes
                # section K instantiable rather than hypothetical.
                adapter.reset(layout, f"goal:{goal_draw}")
                adapter.reset(layout, f"appearance:{appearance}")
            level = adapter._require()
            switches = {tuple(int(v) for v in c) for c in level.switches}
            goal = tuple(int(v) for v in level.markers[adapter._goal_marker])
            text = adapter.goal_text()
            switch_grid = np.zeros((GRID, GRID), dtype=np.float32)
            for r, c in switches:
                switch_grid[r, c] = 1.0

            frames, actions, positions, polarity = [], [], [], []
            for step in range(steps):
                truth = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in truth["position"])
                frames.append(adapter.frame().copy())
                positions.append(position)
                polarity.append(int(truth["polarity"]))
                if policy == "uniform":
                    action = int(generator.integers(0, len(ACTIONS)))
                else:
                    action = (int(generator.integers(0, len(ACTIONS)))
                              if generator.random() < epsilon
                              else goal_direction_policy(generator, position, goal))
                actions.append(action)
                if adapter.step(action, gate.authorize_evaluator(action, "n")).terminated:
                    break
            if len(frames) < 3:
                continue

            length = len(frames)
            agent_mask = np.zeros((length, GRID, GRID), dtype=np.float32)
            for t, (r, c) in enumerate(positions):
                agent_mask[t, r, c] = 1.0
            # The switch layout is static, but the VISIBLE switch set is not: the agent
            # occludes the cell it stands on. The mask target is what the frame shows.
            switch_visible = np.repeat(switch_grid[None], length, axis=0).copy()
            for t, (r, c) in enumerate(positions):
                switch_visible[t, r, c] = 0.0

            displacement = np.zeros(length, dtype=np.int64)
            entered = np.zeros(length, dtype=np.float32)
            event = np.zeros(length, dtype=np.float32)
            for t in range(1, length):
                displacement[t] = displacement_class(positions[t - 1], positions[t])
                moved = positions[t] != positions[t - 1]
                entered[t] = float(positions[t] in switches)
                event[t] = float(moved and positions[t] in switches)

            stripe = np.zeros(length, dtype=np.float32)
            stripe[0] = float(level.initial_polarity)
            reset_flag = np.zeros(length, dtype=np.float32)
            reset_flag[0] = 1.0

            out.append(VisualEpisode(
                layout=layout, appearance=appearance,
                frames=np.stack(frames), actions=np.array(actions[:length]),
                positions=np.array(positions), polarity=np.array(polarity),
                agent_mask=agent_mask, switch_mask=switch_visible,
                displacement=displacement, entered_switch=entered, event=event,
                stripe_state=stripe, is_reset=reset_flag,
                goal_tokens=np.array(tokenise_goal(text, VOCABULARY)[:MAX_GOAL_TOKENS]),
                goal_text=text))
    return out


# ---- the tensor view every interface consumes ------------------------------------------


@dataclass
class PairBatch:
    """Frame pairs, flattened over (episode, t>=1). This is the ONLY thing a visual
    interface may read, plus the action one-hot."""
    before: np.ndarray        # (N, 24, 24, 3) float32 in [0,1]
    after: np.ndarray         # (N, 24, 24, 3) float32
    action: np.ndarray        # (N, 4) float32 one-hot of A_{t-1}
    event: np.ndarray         # (N,) float32
    displacement: np.ndarray  # (N,) int64
    entered: np.ndarray       # (N,) float32
    agent_before: np.ndarray  # (N, 144) float32
    agent_after: np.ndarray   # (N, 144) float32
    switch_before: np.ndarray  # (N, 144) float32
    stripe: np.ndarray        # (N,) float32, the reset stripe of the episode
    is_reset_pair: np.ndarray  # (N,) float32, 1 when `before` is the reset frame
    entered_map: np.ndarray   # (N, 144) float32, the entered-switch cell if any
    event_map: np.ndarray     # (N, 144) float32, the crossing cell if any
    episode: np.ndarray       # (N,) int
    step: np.ndarray          # (N,) int
    layout: np.ndarray        # (N,) int
    goal: np.ndarray          # (N, G) int

    def __len__(self) -> int:
        return len(self.before)

    def subset(self, mask: np.ndarray) -> "PairBatch":
        return PairBatch(**{k: v[mask] for k, v in self.__dict__.items()})

    def digest(self) -> str:
        h = hashlib.sha256()
        for name in ("before", "after", "action", "event", "layout", "step"):
            h.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return h.hexdigest()[:16]


def to_pairs(episodes: Sequence[VisualEpisode]) -> PairBatch:
    fields: dict[str, list] = {k: [] for k in (
        "before", "after", "action", "event", "displacement", "entered", "agent_before",
        "agent_after", "switch_before", "stripe", "is_reset_pair", "episode", "step",
        "layout", "goal")}
    for index, episode in enumerate(episodes):
        for t in range(1, episode.length):
            fields["before"].append(episode.frames[t - 1])
            fields["after"].append(episode.frames[t])
            one_hot = np.zeros(len(ACTIONS), dtype=np.float32)
            one_hot[episode.actions[t - 1]] = 1.0
            fields["action"].append(one_hot)
            fields["event"].append(episode.event[t])
            fields["displacement"].append(episode.displacement[t])
            fields["entered"].append(episode.entered_switch[t])
            fields["agent_before"].append(episode.agent_mask[t - 1].ravel())
            fields["agent_after"].append(episode.agent_mask[t].ravel())
            fields["switch_before"].append(episode.switch_mask[t - 1].ravel())
            fields["stripe"].append(episode.stripe_state[0])
            fields["is_reset_pair"].append(float(t == 1))
            fields["episode"].append(index)
            fields["step"].append(t)
            fields["layout"].append(episode.layout)
            fields["goal"].append(episode.goal_tokens)
    agent_after = np.stack(fields["agent_after"])
    entered = np.array(fields["entered"], dtype=np.float32)
    event = np.array(fields["event"], dtype=np.float32)
    return PairBatch(
        entered_map=agent_after * entered[:, None],
        event_map=agent_after * event[:, None],
        before=np.stack(fields["before"]).astype(np.float32) / 255.0,
        after=np.stack(fields["after"]).astype(np.float32) / 255.0,
        action=np.stack(fields["action"]),
        event=np.array(fields["event"], dtype=np.float32),
        displacement=np.array(fields["displacement"], dtype=np.int64),
        entered=np.array(fields["entered"], dtype=np.float32),
        agent_before=np.stack(fields["agent_before"]),
        agent_after=np.stack(fields["agent_after"]),
        switch_before=np.stack(fields["switch_before"]),
        stripe=np.array(fields["stripe"], dtype=np.float32),
        is_reset_pair=np.array(fields["is_reset_pair"], dtype=np.float32),
        episode=np.array(fields["episode"]), step=np.array(fields["step"]),
        layout=np.array(fields["layout"]), goal=np.stack(fields["goal"]))


def splits(trajectories: int = 3, steps: int = 9) -> dict[str, list[VisualEpisode]]:
    """Section G. Every split is kept separate and named for what it actually varies.

    No dynamics split appears here and none is claimed: v2 has one transition function --
    SWITCH_COUNT is a constant and the flip rule never varies.
    """
    return {
        "train": collect_visual(TRAIN_LAYOUTS, trajectories, steps,
                                CANONICAL_APPEARANCE_SEED, 11),
        "same_layout_new_trajectories": collect_visual(
            SAME_LAYOUT_LAYOUTS, 2, steps, CANONICAL_APPEARANCE_SEED, 4_242),
        "held_out_layouts": collect_visual(HELD_OUT_LAYOUTS, 2, steps,
                                           CANONICAL_APPEARANCE_SEED, 313),
        "appearance_shift": collect_visual(TRAIN_LAYOUTS[:30], 2, steps,
                                           APPEARANCE_SHIFT_SEED, 515),
        "visitation_policy_shift": collect_visual(
            HELD_OUT_LAYOUTS, 2, steps, CANONICAL_APPEARANCE_SEED, 616,
            policy="goal_directed"),
        "crossed_layout_and_appearance": collect_visual(
            CROSSED_LAYOUTS, 2, steps, APPEARANCE_SHIFT_SEED, 717),
    }
