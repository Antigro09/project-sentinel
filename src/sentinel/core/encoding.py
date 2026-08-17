"""Turning an episode into tensors the core can reason over.

Two decisions shape everything downstream.

**Nothing is pre-extracted.** The encoding hands over raw cropped grids and
the action taken, and nothing else. It does not mark which cell is the
agent, does not compute displacement, does not flag which cells are walls.
Identifying the agent *is* most of the problem — an encoding that pointed
at it would be solving the task in the preprocessor and then congratulating
the network for reading the answer.

**The signal lives in the diff.** What distinguishes one world's rules from
another's is how a grid changes when an action is taken, so each transition
is presented as (before, after, changed-mask) rather than as isolated
frames. The changed-mask is redundant — derivable from the other two — but
it is the single most informative channel and making the network rediscover
subtraction is a waste of its capacity.

The hardest label in the set is `charge_period`: a hidden counter that
makes every Nth move travel two cells instead of one. It is invisible in
any single frame and invisible in any single transition. It can only be
recovered from the *pattern across a sequence* of transitions, which is
precisely why it is the interesting test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.env.history import History
from sentinel.env.types import GRID_SIZE
from sentinel.gen.spec import Mechanics, WorldSpec
from sentinel.verify.region import active_region

CROP = 16
"""Active regions are 7-13 cells square, so 16 holds them with room to spare."""

MAX_TRANSITIONS = 32
"""Enough to see a period-3 or period-4 pattern many times over."""

N_CELL_VALUES = 16
N_ACTIONS = 5

CHANNELS = 3
"""before, after, changed."""


@dataclass(frozen=True, slots=True)
class MechanicLabels:
    """Ground truth for one world. Free, because the generator made it.

    This is what makes Phase 3 runnable without any LLM at all: every
    generated world knows its own rules, so labels are unlimited and exact.
    """

    step_distance: int
    """1 or 2 -> class 0 or 1."""
    charge_period: int
    """0 = none, 3, or 4 -> class 0, 1, 2. THE hidden-state label."""
    wrap_edges: int
    has_hazards: int
    has_switches: int
    ordered_targets: int

    @classmethod
    def from_mechanics(cls, mech: Mechanics) -> MechanicLabels:
        charge = {None: 0, 3: 1, 4: 2}.get(mech.charge_period, 0)
        return cls(
            step_distance=max(0, min(1, mech.step_distance - 1)),
            charge_period=charge,
            wrap_edges=int(mech.wrap_edges),
            has_hazards=int(mech.has_hazards),
            has_switches=int(mech.has_switches),
            ordered_targets=int(mech.ordered_targets),
        )

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.step_distance,
                self.charge_period,
                self.wrap_edges,
                self.has_hazards,
                self.has_switches,
                self.ordered_targets,
            ],
            dtype=np.int32,
        )


HEADS: tuple[tuple[str, int], ...] = (
    ("step_distance", 2),
    ("charge_period", 3),
    ("wrap_edges", 2),
    ("has_hazards", 2),
    ("has_switches", 2),
    ("ordered_targets", 2),
)
"""(name, n_classes) per prediction head, in label-array order."""


def crop_box(history: History) -> tuple[int, int]:
    """Top-left of the CROP x CROP window over the active region."""
    region = active_region(history)
    if not region:
        return 0, 0
    xs = [x for x, _ in region]
    ys = [y for _, y in region]
    x0 = max(0, min(min(xs), GRID_SIZE - CROP))
    y0 = max(0, min(min(ys), GRID_SIZE - CROP))
    return x0, y0


def encode_history(history: History) -> tuple[np.ndarray, np.ndarray]:
    """Encode an episode.

    Returns:
        grids:   (MAX_TRANSITIONS, CROP, CROP, CHANNELS) int8
        actions: (MAX_TRANSITIONS,) int32, -1 where padded
    """
    x0, y0 = crop_box(history)
    grids = np.zeros((MAX_TRANSITIONS, CROP, CROP, CHANNELS), dtype=np.int8)
    actions = np.full((MAX_TRANSITIONS,), -1, dtype=np.int32)

    previous = history.initial
    count = 0
    for step in history.steps:
        if count >= MAX_TRANSITIONS:
            break
        settled = step.settled

        before = np.array(
            [row[x0 : x0 + CROP] for row in previous.grid[y0 : y0 + CROP]], dtype=np.int8
        )
        after = np.array(
            [row[x0 : x0 + CROP] for row in settled.grid[y0 : y0 + CROP]], dtype=np.int8
        )
        # Pad if the crop ran off the edge of the frame.
        if before.shape != (CROP, CROP):
            padded = np.zeros((CROP, CROP), dtype=np.int8)
            padded[: before.shape[0], : before.shape[1]] = before
            before = padded
        if after.shape != (CROP, CROP):
            padded = np.zeros((CROP, CROP), dtype=np.int8)
            padded[: after.shape[0], : after.shape[1]] = after
            after = padded

        grids[count, :, :, 0] = before
        grids[count, :, :, 1] = after
        grids[count, :, :, 2] = (before != after).astype(np.int8)
        actions[count] = step.action.action_id

        previous = settled
        count += 1

    return grids, actions


def encode_world(spec: WorldSpec, history: History) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode an episode together with its ground-truth mechanics."""
    grids, actions = encode_history(history)
    labels = MechanicLabels.from_mechanics(spec.mechanics).as_array()
    return grids, actions, labels


def label_names() -> list[str]:
    return [name for name, _ in HEADS]
