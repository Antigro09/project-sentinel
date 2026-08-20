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

from sentinel.env.boundary import is_boundary
from sentinel.env.history import History
from sentinel.env.types import GRID_SIZE
from sentinel.gen.spec import Mechanics, WorldSpec
from sentinel.verify.region import active_region

CROP = 16
"""Active regions are 7-13 cells square, so 16 holds them with room to spare."""

MAX_TRANSITIONS = 32
"""How much of an episode the core sees.

Kept at 32 after testing 64. The window does force a trade between rules --
the hidden counter needs varied movement to reveal its period, while
`ordered_targets` needs deliberate walks onto targets -- but widening it did
not resolve that trade, it moved it. Measured over three seeds, 64
transitions with heavier probing gave charge_period 0.651 +/- 0.183 against
0.795 +/- 0.159 at 32, buying only +0.03 on ordered_targets. Charge is worth
more: it is the label that actually converts into solve rate."""

N_CELL_VALUES = 16
N_ACTIONS = 5

BOUNDARY_ACTION = -2
"""Marks a transition no action caused: a level change or an engine reset.

Distinct from the -1 used for padding, because the two mean opposite
things -- padding is "nothing here", a boundary is "something happened
that you must not attribute to the action"."""

CHANNELS = 3
"""before, after, changed."""


CHARGE_CLASSES: tuple[int | None, ...] = (None, 2, 3, 4, 5)
EDGE_CLASSES: tuple[str, ...] = ("block", "wrap", "bounce", "respawn")
HAZARD_CLASSES: tuple[str, ...] = ("none", "kill", "pushback", "respawn")
SWITCH_CLASSES: tuple[str, ...] = ("none", "toggle", "latch")


@dataclass(frozen=True, slots=True)
class MechanicLabels:
    """Ground truth for one world. Free, because the generator made it.

    This is what makes Phase 3 runnable without any LLM at all: every
    generated world knows its own rules, so labels are unlimited and exact.

    **Eight labels, not six.** The original six spanned 26 rule sets, and a
    space that small cannot produce a holdout whose labels vary
    independently -- measured at 1.00 confounding for every holdout size
    tried, meaning some label was always either constant or a restatement
    of another. The first version of this benchmark scored `charge_period`
    at 0.795 while `charge_period` was exactly `has_hazards` in the held-out
    set, so a model detecting coloured cells looked like a model inferring a
    hidden counter. These eight span 5,760 rule sets, which is enough for
    the holdout to be chosen rather than hoped for.
    """

    step_distance: int
    """1, 2 or 3 -> class 0, 1, 2."""
    charge_period: int
    """Index into CHARGE_CLASSES. THE hidden-state label."""
    edge_mode: int
    """Index into EDGE_CLASSES."""
    hazards: int
    """Index into HAZARD_CLASSES; class 0 means no hazards at all."""
    switches: int
    """Index into SWITCH_CLASSES; class 0 means no switches at all."""
    ordered_targets: int
    gates_start_open: int
    wait_advances_charge: int
    """Whether waiting ticks the hidden counter. Worlds where it does not
    are strictly easier, since the period can be pinned by waiting."""

    @classmethod
    def from_mechanics(cls, mech: Mechanics) -> MechanicLabels:
        charge = mech.charge_period if mech.charge_period in CHARGE_CLASSES else None
        edge = mech.effective_edge_mode()
        hazard = mech.hazard_effect if mech.has_hazards else "none"
        switch = mech.switch_mode if mech.has_switches else "none"
        return cls(
            step_distance=max(0, min(2, mech.step_distance - 1)),
            charge_period=CHARGE_CLASSES.index(charge),
            edge_mode=EDGE_CLASSES.index(edge) if edge in EDGE_CLASSES else 0,
            hazards=HAZARD_CLASSES.index(hazard) if hazard in HAZARD_CLASSES else 0,
            switches=SWITCH_CLASSES.index(switch) if switch in SWITCH_CLASSES else 0,
            ordered_targets=int(mech.ordered_targets),
            gates_start_open=int(mech.gates_start_open),
            wait_advances_charge=int(mech.wait_advances_charge),
        )

    def to_mechanics(self) -> Mechanics:
        return Mechanics(
            step_distance=self.step_distance + 1,
            charge_period=CHARGE_CLASSES[self.charge_period],
            wrap_edges=False,
            edge_mode=EDGE_CLASSES[self.edge_mode],
            has_hazards=self.hazards != 0,
            hazard_effect=HAZARD_CLASSES[self.hazards] if self.hazards else "kill",
            has_switches=self.switches != 0,
            switch_mode=SWITCH_CLASSES[self.switches] if self.switches else "toggle",
            ordered_targets=bool(self.ordered_targets),
            gates_start_open=bool(self.gates_start_open),
            wait_advances_charge=bool(self.wait_advances_charge),
        )

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.step_distance,
                self.charge_period,
                self.edge_mode,
                self.hazards,
                self.switches,
                self.ordered_targets,
                self.gates_start_open,
                self.wait_advances_charge,
            ],
            dtype=np.int32,
        )


HEADS: tuple[tuple[str, int], ...] = (
    ("step_distance", 3),
    ("charge_period", len(CHARGE_CLASSES)),
    ("edge_mode", len(EDGE_CLASSES)),
    ("hazards", len(HAZARD_CLASSES)),
    ("switches", len(SWITCH_CLASSES)),
    ("ordered_targets", 2),
    ("gates_start_open", 2),
    ("wait_advances_charge", 2),
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

        # A reset or a level change is not the effect of an action, and
        # encoding one as though it were teaches the network that a single
        # move can rebuild the board -- including reinstating collected
        # targets, which is the exact signature `ordered_targets` rests on.
        #
        # But DROPPING them is worse, and the measurement was brutal:
        # charge_period fell from 0.795 to 0.088, well below the 0.33 chance
        # line for three classes. The hidden counter is *periodic*. It keeps
        # ticking across a level change, so removing a transition leaves a
        # gap the network cannot see, and moves n, n+1, n+3 get read as
        # consecutive. Every phase it learns is then wrong, which is how
        # accuracy lands below chance rather than at it.
        #
        # So boundaries stay in the sequence, preserving the time axis, and
        # are MARKED instead. The action id says "no action caused this",
        # leaving the network free to discount the content without losing
        # count of when things happened.
        #
        # The check is `levels_completed`, NOT `level_index`: a step carries
        # the level it ENDED on, so the crossing step already reports the
        # new number and comparing consecutive `level_index` values misses
        # the crossing itself. See env/boundary.
        boundary = is_boundary(previous, settled)

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
        actions[count] = BOUNDARY_ACTION if boundary else step.action.action_id

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


def defined_mask(labels: np.ndarray, head: int) -> np.ndarray:
    """Which rows have a well-defined value for this head.

    Two labels are conditional, and scoring them where they are undefined
    reports a coin flip as if it were knowledge -- the same error that made
    the first benchmark meaningless.

    `wait_advances_charge` governs whether a non-move action ticks the
    hidden counter, so in a world with NO counter it changes nothing
    observable. Measured: the evidence determines it in 100% of charge=2,
    charge=3 and charge=5 worlds, 85% of charge=4, and **0%** of worlds
    without a counter.

    `gates_start_open` says whether gates begin passable, and gates exist
    only where switches do. With no switches there are no gates and the flag
    is unobservable -- which covers 46% of the held-out episodes, so nearly
    half of that label's reported accuracy was guessing.
    """
    names = [n for n, _ in HEADS]
    if names[head] == "wait_advances_charge":
        return labels[:, names.index("charge_period")] != 0
    if names[head] == "gates_start_open":
        return labels[:, names.index("switches")] != 0
    return np.ones(len(labels), dtype=bool)
