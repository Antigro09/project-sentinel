"""Telling a real transition from a discontinuity.

A level change or an engine reset rebuilds the board. It is not the effect
of the action that preceded it, and anything that treats it as one is being
taught that a single move can move every wall.

**`level_index` is not sufficient to detect this**, which cost two separate
bugs before it was noticed. A step carries the level it ended on, so the
step that *crosses* into a new level already reports the new number, and the
step after it reports the same number. Comparing consecutive `level_index`
values therefore misses the crossing itself. Observed directly on world
w2000007: the episode's initial frame is level 0, step 0 lands on level 1,
and steps 0 and 1 both report `level_index=1`.

`levels_completed` is carried on the frame rather than the step, so
comparing it across a transition detects the crossing exactly. `full_reset`
covers the other discontinuity.
"""

from __future__ import annotations

from sentinel.env.history import History
from sentinel.env.types import Observation


def is_boundary(before: Observation, after: Observation) -> bool:
    """Did the board discontinuously change between these two frames?"""
    return after.full_reset or after.levels_completed != before.levels_completed


def continuous_runs(history: History) -> list[tuple[Observation, list]]:
    """Split an episode into runs with no discontinuity inside them.

    Each run is (frame_before_the_run, steps). The frame is the state the
    first step of the run acted on, so a run is directly replayable through
    a world model built from that frame.
    """
    runs: list[tuple[Observation, list]] = []
    start = history.initial
    current: list = []
    previous = history.initial

    for step in history.steps:
        if is_boundary(previous, step.settled):
            if current:
                runs.append((start, current))
            # The crossing step's own result is the new run's starting frame.
            start = step.settled
            current = []
        else:
            current.append(step)
        previous = step.settled

    if current:
        runs.append((start, current))
    return runs


def longest_run(history: History) -> tuple[Observation, list]:
    """The longest stretch of the episode with no board rebuild in it.

    The most evidence available about one level, which is what a
    single-level hypothesis can actually be scored against.
    """
    runs = continuous_runs(history)
    if not runs:
        return history.initial, []
    return max(runs, key=lambda r: len(r[1]))
