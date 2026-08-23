"""X20: rebuild the explorer's committee from the DSL quotient.

X18's finding 1: the purposeful explorers (info-gain, plan-disagree) draw
their committees from the OLD LABEL SPACE -- 5,760 hypotheses that cannot
represent step=8 or charge=16. Their disagreement signal is purposeful
about the wrong hypothesis space, and they LOSE to random walk. The
narrowness migrated into the explorer exactly as it had migrated into the
core (X6).

The fix: a QbC explorer whose committee is a SAMPLED DSL QUOTIENT --
programs drawn from the full grammar, refuted against the episode so far,
steering toward actions the surviving behaviours most disagree about --
plus periodic hazard-SEEKING (rendered hazards are visible; a world's
hazard axis can only be exercised by encountering one) with no blanket
hazard-cell veto.

MEASURED (12 identifiability-aware worlds, same protocol as X19):

    policy     exact  bisimilar  avg steps
    random      2/12     11/12       172
    dsl-qbc     4/12     12/12       196

The DSL-committee explorer WINS on exact recovery (doubling it) and is
perfect at behavioural recovery. Two of its wins are diagnostic:

  - w02: truth had respawn hazards; random never stepped on the single
    hazard in 200 steps and selected 'no hazards' -- a CORRECT induction
    from evidence that never touched them. The seeker encounters the
    hazard and recovers the axis exactly.
  - w10: similar coverage win via disagreement steering.

Every component of the Level 4 loop now speaks DSL: generator
(X19), explorer (this), search and selection (X17). No label vocabulary
remains anywhere in the loop.
"""

from __future__ import annotations

from __future__ import annotations

import itertools
import sys
import time
from itertools import permutations as _perm

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import scorable_segment
from sentinel.core import load_split
from sentinel.env.types import Action
from sentinel.explore.version_space import state_key
from sentinel.gen.grid import GridWorld, initial_state, transition_state
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec

sys.path.insert(0, "experiments")
from x17_dsl_search import AXES, SpecCache, compile_program, complexity  # noqa: E402
from x19_identifiable_worlds import make_identifiable_level  # noqa: E402

N_WORLDS = 12
EPISODE_STEPS = 200
COMMITTEE = 300
EPSILON = 0.15


def refute_committee(live, segment, cache, order, field_size):
    """Drop committee members that mispredict any observed frame."""
    for step in segment.steps:
        if len(live) <= 1:
            break
        from x17_dsl_search import _frame_facts
        wa, wt, wg = _frame_facts(step.settled.grid, field_size)
        survivors = []
        for idx, state in live:
            nxt = transition_state(state, step.action, cache.get(idx, order))
            here = (nxt.x, nxt.y)
            visible = frozenset(t for t in nxt.remaining if t != here)
            if here != wa or visible != wt:
                continue
            if wg is not None and bool(nxt.gates_open) != wg:
                continue
            survivors.append((idx, nxt))
        live = survivors
    return live


def dsl_qbc_episode(mech: Mechanics, observed, field_size: int, seed: int,
                    steps: int = EPISODE_STEPS):
    spec = WorldSpec(world_id="syn", seed=0, field_size=field_size,
                     mechanics=mech, levels=(observed,))
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)

    progs = [tuple(a[int(rng.integers(0, len(a)))] for a in AXES)
             for _ in range(COMMITTEE)]
    cache = SpecCache(observed, field_size)
    idxs = [tuple(a.index(v) for a, v in zip(AXES, p)) for p in progs]
    order = observed.targets
    live = [(i, initial_state(0, cache.get(i, order))) for i in idxs]

    size = field_size
    spent = 0
    while spent < steps and not world.done:
        seg = scorable_segment(world.history)
        live = refute_committee(live, seg, cache, order, field_size)

        grid = world.history.last.grid
        here = None
        for y in range(size):
            for x in range(size):
                if grid[y][x] == 4:
                    here = (x, y)
                    break
                if here:
                    break

        # NOTE: no blanket hazard-cell veto. Hazards are only dangerous
        # under some hypotheses (kill), and encountering one is the only
        # way to exercise the hazard axis -- which is the point of the
        # seeker below. Random movement still avoids them implicitly via
        # the epsilon branch's own survival pressure.
        cands = [1, 2, 3, 4, 5]

        if len(live) <= 1 or rng.random() < EPSILON:
            world.step(Action(int(rng.choice(cands))))
            spent += 1
            continue

        # Periodic hazard-seeking: rendered hazards are visible in the
        # frame, and a world's hazard axis can only be exercised by
        # ENCOUNTERING one. Without this, 'no hazards' is a correct
        # induction from evidence that never touched them -- measured on
        # w02, where the single hazard was never stepped on in 200 steps.
        aid_choice = None
        hazards = [(x, y) for y in range(size) for x in range(size)
                   if grid[y][x] == 2]
        if hazards and here is not None and spent % 15 == 14:
            hx, hy = min(hazards,
                         key=lambda h: abs(h[0] - here[0]) + abs(h[1] - here[1]))
            best, best_d = None, abs(hx - here[0]) + abs(hy - here[1])
            for a, (dx, dy) in {1: (0, -1), 2: (0, 1), 3: (-1, 0),
                                4: (1, 0)}.items():
                d = abs(hx - (here[0] + dx)) + abs(hy - (here[1] + dy))
                if d < best_d:
                    best, best_d = a, d
            aid_choice = best

        if aid_choice is None:
            best_aid, best_split = None, -1
            for aid in cands:
                outs = set()
                for i, st in live:
                    try:
                        nxt = transition_state(st, Action(aid),
                                               cache.get(i, order))
                    except Exception:
                        continue
                    outs.add(state_key(nxt))
                if len(outs) > best_split:
                    best_aid, best_split = aid, len(outs)
            aid_choice = best_aid if best_aid else int(rng.choice(cands))
        world.step(Action(aid_choice))
        spent += 1
    return scorable_segment(world.history)


def random_episode(mech, observed, field_size, seed, steps=EPISODE_STEPS):
    spec = WorldSpec(world_id="syn", seed=0, field_size=field_size,
                     mechanics=mech, levels=(observed,))
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    size = field_size
    for _ in range(steps):
        if world.done:
            break
        grid = world.history.last.grid
        here = None
        for y in range(size):
            for x in range(size):
                if grid[y][x] == 4:
                    here = (x, y)
                    break
                if here:
                    break
        choices = [1, 2, 3, 4, 5]
        if here is not None:
            safe = []
            for aid in (1, 2, 3, 4):
                dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[aid]
                nx, ny = here[0] + dx, here[1] + dy
                if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] != 2:
                    safe.append(aid)
            if safe and rng.random() > 0.15:
                choices = safe
        world.step(int(rng.choice(choices)))
    return scorable_segment(world.history)


def refute_select(segment, observed, field_size: int):
    orders = ([tuple(p) for p in _perm(observed.targets)]
              if 2 <= len(observed.targets) <= 4 else [observed.targets])
    cache = SpecCache(observed, field_size)
    live = [(idx, order, initial_state(0, cache.get(idx, order)))
            for idx in itertools.product(*[range(len(a)) for a in AXES])
            for order in orders]
    sims = 0
    for step in segment.steps:
        if len(live) <= 1:
            break
        from x17_dsl_search import _frame_facts
        wa, wt, wg = _frame_facts(step.settled.grid, field_size)
        survivors = []
        for idx, order, state in live:
            sims += 1
            nxt = transition_state(state, step.action, cache.get(idx, order))
            here = (nxt.x, nxt.y)
            visible = frozenset(t for t in nxt.remaining if t != here)
            if here != wa or visible != wt:
                continue
            if wg is not None and bool(nxt.gates_open) != wg:
                continue
            survivors.append((idx, order, nxt))
        live = survivors
    if not live:
        return tuple(0 for _ in AXES), orders[0], sims, 0
    best = min(live, key=lambda t: complexity(t[0]))
    return best[0], best[1], sims, len(live)


def main() -> int:
    rng = np.random.default_rng(0)
    from sentinel.core.universal import PROBE_ACTIONS
    long_probe = tuple(list(PROBE_ACTIONS)
                       + [((i % 5) + 1) for i in range(32)])

    results = {p: {"exact": 0, "bisim": 0, "steps": 0} for p in ("random", "dsl-qbc")}
    t0 = time.perf_counter()

    print(f"{N_WORLDS} identifiability-aware worlds; "
          f"random vs DSL-committee QbC ({COMMITTEE} members)\n")
    for wi in range(N_WORLDS):
        charge_opts = (None, 6, 8, 10, 12, 14, 16, 18, 20)
        truth_mech = Mechanics(
            step_distance=int(rng.integers(1, 9)),
            charge_period=charge_opts[int(rng.integers(0, len(charge_opts)))],
            edge_mode=str(rng.choice(("block", "wrap", "bounce", "respawn"))),
            has_hazards=bool(rng.integers(0, 2)),
            hazard_effect=str(rng.choice(("kill", "pushback", "respawn"))),
            has_switches=True,
            switch_mode=str(rng.choice(("toggle", "latch"))),
            ordered_targets=bool(rng.integers(0, 2)),
            gates_start_open=bool(rng.integers(0, 2)),
            wait_advances_charge=bool(rng.integers(0, 2)),
        )
        size = int(rng.integers(9, 14))
        level = make_identifiable_level(rng, size, truth_mech)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )

        truth_prog = (truth_mech.step_distance, truth_mech.charge_period,
                      truth_mech.effective_edge_mode(),
                      (truth_mech.has_hazards,
                       truth_mech.hazard_effect if truth_mech.has_hazards
                       else "kill"),
                      (truth_mech.has_switches,
                       truth_mech.switch_mode if truth_mech.has_switches
                       else "toggle"),
                      truth_mech.ordered_targets, truth_mech.gates_start_open,
                      truth_mech.wait_advances_charge)
        truth_idx = tuple(a.index(v) for a, v in zip(AXES, truth_prog))

        line = f"  w{wi:02d}:"
        for policy in ("random", "dsl-qbc"):
            if policy == "random":
                seg = random_episode(truth_mech, observed, size, seed=2000 + wi)
            else:
                seg = dsl_qbc_episode(truth_mech, observed, size, seed=2000 + wi)
            best_idx, best_order, sims, n_surv = refute_select(
                seg, observed, size)
            prog = tuple(a[i] for a, i in zip(AXES, best_idx))
            best_mech = compile_program(prog)
            exact = best_idx == truth_idx
            ts = WorldSpec(world_id="ts", seed=0, field_size=size,
                           mechanics=truth_mech, levels=(observed,))
            bs = WorldSpec(world_id="bs", seed=0, field_size=size,
                           mechanics=best_mech, levels=(observed,))
            ts_s, bs_s = initial_state(0, ts), initial_state(0, bs)
            b = True
            for aid in long_probe:
                try:
                    ts_s = transition_state(ts_s, Action(aid), ts)
                    bs_s = transition_state(bs_s, Action(aid), bs)
                except Exception:
                    b = False
                    break
                if state_key(ts_s) != state_key(bs_s):
                    b = False
                    break
            r = results[policy]
            r["exact"] += exact
            r["bisim"] += b
            r["steps"] += len(seg.steps)
            line += f"  {policy}={'EXACT' if exact else ('bisim' if b else 'MISS')}"

        print(line)

    dt = time.perf_counter() - t0
    n = N_WORLDS
    print(f"\n{'policy':>10} {'exact':>8} {'bisimilar':>10} {'avg steps':>10}")
    for p in ("random", "dsl-qbc"):
        r = results[p]
        print(f"{p:>10} {r['exact']:5d}/{n} {r['bisim']:7d}/{n} "
              f"{r['steps'] / n:9.0f}")
    print(f"({dt:.0f}s total)")

    rq, rr = results["dsl-qbc"], results["random"]
    print("\nverdict:")
    if rq["exact"] > rr["exact"]:
        print("   DSL-COMMITTEE EXPLORER WINS on exact recovery. The last")
        print("   label-bound component of the loop is fixed: generator,")
        print("   explorer, search and selection all speak DSL now.")
    elif rq["bisim"] >= rr["bisim"] and rq["exact"] == rr["exact"]:
        print("   PARITY with random at behavioural recovery. With X19 already")
        print("   at 12/12 bisimilar, the remaining inexactness is coverage,")
        print("   not exploration policy; longer worlds are the lever.")
    else:
        print("   QbC still loses: one-step disagreement saturates even over")
        print("   the DSL quotient (X8's mechanism). Sequence-level planning")
        print("   with a DSL committee is the next attempt, or accept random.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
