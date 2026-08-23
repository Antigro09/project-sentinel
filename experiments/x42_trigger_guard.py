"""X42: tell "I need a new language" apart from "my inputs are garbage".

X35 grows its grammar when refutation of all 61,440 programs comes back
EMPTY. That signal is load-bearing and dangerously ambiguous: an empty
survivor set is the signature of a grammar that cannot express the world,
and equally the signature of corrupted evidence -- a misread layout, a level
boundary scored as an action effect, a non-deterministic replay. This
project has shipped exactly that bug twice, so the trigger firing on a
broken episode is not hypothetical.

Growing the grammar in response to garbage is the worst possible failure: it
would invent vocabulary to explain a bug, adopt it, and carry it forward as
knowledge.

THE DISCRIMINATOR IS PREFIX DEPTH. If the grammar is merely inadequate,
short prefixes are still explicable -- a single transition is consistent
with thousands of programs, and survivors collapse to zero only once the
world does something no program can express. If the EVIDENCE is corrupt, the
very first transition is already inexplicable, because no rule set produces
it. So:

    prefix_depth = 0     nothing explains even one transition -> CORRUPT
    prefix_depth > 0     the world was explicable until step d and then
                         was not -> genuine novelty, located at step d

Prefix depth is NOT sufficient on its own, and finding that out took two
failures worth recording:

    a record whose frames are SWAPPED still has an explicable prefix -- a
    few transposed steps are consistent with some program -- so the guard
    read scrambled evidence as "novelty at step 3" and would have grown the
    grammar to explain a bug.

    a cell-change bound does not separate them either. Measured over 480
    transitions, genuine and scrambled episodes both peak at 7 changed cells,
    means 0.8 and 1.1. Swapping ADJACENT steps keeps every frame locally
    plausible, so no per-transition statistic distinguishes them.

What does distinguish them is RE-EXECUTION. Replay the recorded actions
against the environment and compare frame for frame: genuine novelty
replays exactly, because the world is inexpressible rather than unreal,
while a corrupted record does not, because those frames were never what
those actions produced. Trust execution, not description -- the principle
the whole project rests on, applied to its own logs.

Three checks, then, plus the depth:

    layout round-trip    re-render the layout read from frame 0 and compare
                         to the frame; failure means the reader is wrong
    replay determinism   same actions twice, same digest
    record fidelity      the log matches what re-execution produces
    prefix depth         0 means nothing explains even one transition

MEASURED, on genuine ice worlds and on deliberately corrupted episodes,
because a guard that has never seen the thing it guards against is untested:

    genuine ice worlds        4/4 authenticated as novelty
                              (explicable through steps 4, 11, 13, 24)
    scrambled-frames          2/2 blocked, by record fidelity
    wrong-layout              2/2 blocked, by prefix depth 0
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x35_novelty_trigger import (
    EXPLORE_STEPS,
    FAMILY_BY_NAME,
    make_ice_world,
    refute_family,
    run_agent,
)
from sentinel.adapt.hypothesis import scorable_segment
from sentinel.core.agent import read_layout
from sentinel.env.types import Action
from sentinel.gen.grid import GridWorld, initial_state, render_grid
from sentinel.gen.spec import WorldSpec


@dataclass(frozen=True, slots=True)
class Verdict:
    kind: str            # "novelty" | "corrupt" | "expressible"
    prefix_depth: int
    layout_ok: bool
    deterministic: bool
    detail: str

    @property
    def may_grow(self) -> bool:
        """Growth is permitted only on an authenticated novelty signal."""
        return self.kind == "novelty" and self.layout_ok and self.deterministic


def layout_round_trip(world, spec, size: int) -> bool:
    """Does the layout read from frame 0 re-render to frame 0?

    If the reader mis-parses the board, every downstream refutation is
    against a world that was never observed -- and refutation would go
    empty for reasons having nothing to do with the grammar.
    """
    frame = world.history.initial.grid
    observed = read_layout(frame, size)
    probe = WorldSpec(world_id="rt", seed=0, field_size=size,
                      mechanics=spec.mechanics, levels=(observed,))
    return render_grid(initial_state(0, probe), probe) == frame


def replay_deterministic(spec, actions: list[int]) -> bool:
    """Same actions, same frames? Exact replay is meaningless otherwise."""
    digests = []
    for _ in range(2):
        world = GridWorld(spec)
        world.reset()
        for aid in actions:
            if world.done:
                break
            world.step(Action(aid))
        digests.append(world.history.digest())
    return digests[0] == digests[1]


def record_faithful(world, spec, actions: list[int]) -> bool:
    """Is the RECORD of what happened what actually happens?

    The decisive check, and it took two failures to find. Prefix depth
    catches a wholesale mismatch -- a layout from another world leaves not
    even one transition explicable -- but it does NOT catch a record whose
    frames were swapped, because a few transposed steps are still explicable
    by some program and the guard reads that as novelty at step 3.

    A cell-change bound does not catch it either: measured over 480
    transitions, genuine and scrambled episodes both peak at 7 changed cells
    with means of 0.8 and 1.1. Swapping ADJACENT steps keeps every frame
    locally plausible, so no per-transition statistic separates them.

    What does separate them is re-execution. Replay the recorded actions
    against the environment and compare frame for frame. Genuine novelty
    replays exactly -- the world is inexpressible, not unreal. A corrupted
    record does not, because the frames were never what the actions
    produced. This is the same principle the whole project rests on: trust
    execution, not description.
    """
    replay = GridWorld(spec)
    replay.reset()
    recorded = world.history.steps
    for i, aid in enumerate(actions):
        if replay.done or i >= len(recorded):
            break
        replay.step(Action(aid))
        if replay.history.last.grid != recorded[i].settled.grid:
            return False
    return True


def prefix_depth(steps, observed, size: int, limit: int = 24) -> int:
    """Longest prefix of the episode that SOME program still explains.

    Bisection would be faster; linear is used because the interesting
    values are small -- corruption shows at 0 and genuine novelty at the
    first slide -- and correctness matters more than speed in a guard.
    """
    base = FAMILY_BY_NAME["base"]
    depth = 0
    for d in range(1, min(len(steps), limit) + 1):
        if refute_family(base, steps[:d], observed, size):
            depth = d
        else:
            break
    return depth


def authenticate(world, spec, size: int, actions: list[int]) -> Verdict:
    """Decide whether an empty refutation licenses growing the grammar."""
    steps = scorable_segment(world.history).steps
    observed = read_layout(world.history.initial.grid, size)

    survivors = refute_family(FAMILY_BY_NAME["base"], steps, observed, size)
    if survivors:
        return Verdict("expressible", len(steps), True, True,
                       f"{len(survivors)} survivors; grammar suffices")

    layout_ok = layout_round_trip(world, spec, size)
    deterministic = replay_deterministic(spec, actions) and \
        record_faithful(world, spec, actions)
    depth = prefix_depth(steps, observed, size)

    if not layout_ok:
        return Verdict("corrupt", depth, layout_ok, deterministic,
                       "layout does not re-render to the observed frame")
    if not deterministic:
        return Verdict("corrupt", depth, layout_ok, deterministic,
                       "the record does not match what re-execution produces")
    if depth == 0:
        return Verdict("corrupt", depth, layout_ok, deterministic,
                       "not even one transition is explicable")
    return Verdict("novelty", depth, layout_ok, deterministic,
                   f"explicable through step {depth}, inexplicable after")


# ------------------------------------------------------- corrupt episodes


def corrupt_episode(spec, size: int, mode: str, seed: int = 1234):
    """Break an episode on purpose, the way real bugs have broken them here.

    A guard that has only ever seen valid input is not a guard.
    """
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    actions = []
    for _ in range(EXPLORE_STEPS):
        if world.done:
            break
        aid = int(rng.integers(1, 6))
        world.step(Action(aid))
        actions.append(aid)

    if mode == "scrambled-frames":
        # The failure mode of the boundary bug: frames that never followed
        # from the action recorded beside them.
        steps = world.history.steps
        for i in range(0, len(steps) - 1, 2):
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
    elif mode == "wrong-layout":
        # The failure mode of a misread board: an initial frame that does
        # not describe the world the steps came from.
        other, _ = make_ice_world(seed + 77, size)
        w2 = GridWorld(other)
        w2.reset()
        world.history.initial = w2.history.initial
    return world, actions


def main() -> int:
    print("X42: authenticating the novelty trigger\n")

    print("PART 1 -- genuine ice worlds (the trigger SHOULD fire)")
    print(f'{"world":>8} {"verdict":>12} {"prefix":>7} {"layout":>7} {"determ":>7}  detail')
    genuine_ok = 0
    for i, (seed, size) in enumerate(((0, 10), (100, 20), (7, 12), (23, 14))):
        spec, _ = make_ice_world(seed, size)
        world = GridWorld(spec)
        world.reset()
        rng = np.random.default_rng(1234)
        actions = []
        for _ in range(EXPLORE_STEPS):
            if world.done:
                break
            aid = int(rng.integers(1, 6))
            world.step(Action(aid))
            actions.append(aid)
        v = authenticate(world, spec, size, actions)
        genuine_ok += int(v.may_grow)
        print(f"{f'ice{seed}':>8} {v.kind:>12} {v.prefix_depth:>7} "
              f"{str(v.layout_ok):>7} {str(v.deterministic):>7}  {v.detail}")
    print(f"  authenticated as novelty: {genuine_ok}/4\n")

    print("PART 2 -- corrupted episodes (the trigger MUST NOT fire)")
    print(f'{"corruption":>18} {"verdict":>12} {"prefix":>7}  detail')
    caught = 0
    for mode in ("scrambled-frames", "wrong-layout"):
        for seed, size in ((0, 10), (100, 20)):
            spec, _ = make_ice_world(seed, size)
            world, actions = corrupt_episode(spec, size, mode)
            v = authenticate(world, spec, size, actions)
            blocked = not v.may_grow
            caught += int(blocked)
            print(f"{mode:>18} {v.kind:>12} {v.prefix_depth:>7}  {v.detail}")
    print(f"  growth correctly BLOCKED: {caught}/4\n")

    print("VERDICT")
    if genuine_ok == 4 and caught == 4:
        print("  the guard separates grammar inadequacy from corrupted evidence")
        print("  on every case tried: growth is licensed where the world is")
        print("  genuinely inexpressible and refused where the inputs are broken.")
    else:
        print(f"  INCOMPLETE: {genuine_ok}/4 genuine authenticated, "
              f"{caught}/4 corruptions blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
