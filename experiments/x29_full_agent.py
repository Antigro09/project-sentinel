"""X29: the full agent -- infer the world, then plan inside it to WIN.

Everything before this experiment measured INFERENCE quality. But an
inferred world model is not the product -- it is the instrument. The point
of building an executable world model is that you can search inside it,
spending simulated actions instead of real ones.

THE FULL AGENT, assembled entirely from measured components:

    1. EXPLORE   DSL-committee QbC + hazard-seeking        (X20)
    2. INFER     bulk refutation of all 368,640 programs    (X17)
    3. SELECT    simplicity among survivors                 (X17)
    4. PLAN      BFS inside the inferred model              (plan/search.py)
    5. EXECUTE   divergence-checked; on divergence,
                 re-infer from the extended evidence

MEASURED on 12 fresh held-out worlds:

    agent       solve rate   avg real actions
    random       2/12          529
    FULL         9/12          231
    oracle      10/12           12   <- plans inside the TRUE model

The agent wins 9 of the 10 worlds solvable from the opening state, at the
cost of exploration (~231 real actions vs the oracle's 12). Re-inference
on divergence fired 0.5 times per world: plans derived from the inferred
model mostly survive contact with reality.

THIS IS THE PROGRAMME'S PRODUCT DEMONSTRATION. Not "does it know the
rules" but "does knowing the rules let it win": a system that lands in an
unfamiliar environment, explores it purposefully, infers an executable
model of its dynamics beyond exhaustive reach, plans inside that model,
and wins -- adapting by re-inferring when reality diverges. Every
component speaks DSL; no label vocabulary exists anywhere in the loop.
"""

from __future__ import annotations

from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from itertools import permutations as _perm

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "experiments")
from x17_dsl_search import (  # noqa: E402
    AXES,
    SpecCache,
    compile_program,
    complexity,
)
from x19_identifiable_worlds import make_identifiable_level  # noqa: E402
from x21_derivation_core import mech_to_labels, sample_truth  # noqa: E402

from sentinel.adapt.hypothesis import scorable_segment  # noqa: E402
from sentinel.env.types import Action  # noqa: E402
from sentinel.explore.version_space import state_key  # noqa: E402
from sentinel.gen.grid import (  # noqa: E402
    GridWorld,
    initial_state,
    transition_state,
)
from sentinel.gen.spec import LevelSpec, WorldSpec  # noqa: E402

N_WORLDS = 12
EXPLORE_STEPS = 200
PLAN_LIMIT = 60_000


def dsl_qbc_explore(world: GridWorld, observed, field_size: int, cache,
                    order, live, rng, steps: int):
    """X20's explorer, driving a live world. Mutates `live` in place."""
    size = field_size
    spent = 0
    MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    while spent < steps and not world.done:
        seg = scorable_segment(world.history)
        # refute committee against evidence so far
        for step in seg.steps:
            if len(live) <= 1:
                break
            wa, wt, wg = _frame_facts(step.settled.grid, field_size)
            survivors = []
            for idx, state in live:
                nxt = transition_state(state, step.action,
                                       cache.get(idx, order))
                here = (nxt.x, nxt.y)
                visible = frozenset(t for t in nxt.remaining if t != here)
                if here != wa or visible != wt:
                    continue
                if wg is not None and bool(nxt.gates_open) != wg:
                    continue
                survivors.append((idx, nxt))
            live = survivors

        grid = world.history.last.grid
        here = None
        for y in range(size):
            for x in range(size):
                if grid[y][x] == 4:
                    here = (x, y)
                    break
                if here:
                    break

        aid_choice = None
        hazards = [(x, y) for y in range(size) for x in range(size)
                   if grid[y][x] == 2]
        if hazards and here is not None and spent % 15 == 14:
            hx, hy = min(hazards, key=lambda h: abs(h[0] - here[0])
                         + abs(h[1] - here[1]))
            best, best_d = None, abs(hx - here[0]) + abs(hy - here[1])
            for a, (dx, dy) in MOVES.items():
                d = abs(hx - (here[0] + dx)) + abs(hy - (here[1] + dy))
                if d < best_d:
                    best, best_d = a, d
            aid_choice = best

        if aid_choice is None:
            cands = [1, 2, 3, 4, 5]
            if len(live) <= 1 or rng.random() < 0.15:
                aid_choice = int(rng.choice(cands))
            else:
                best_aid, best_split = None, -1
                for aid in cands:
                    outs = set()
                    for idx, state in live:
                        try:
                            nxt = transition_state(state, Action(aid),
                                                   cache.get(idx, order))
                        except Exception:
                            continue
                        outs.add(state_key(nxt))
                    if len(outs) > best_split:
                        best_aid, best_split = aid, len(outs)
                aid_choice = (best_aid if best_aid
                              else int(rng.choice(cands)))
        world.step(Action(aid_choice))
        spent += 1


def _frame_facts(grid, field_size):
    from x17_dsl_search import _frame_facts as ff
    return ff(grid, field_size)


def bfs_plan(spec: WorldSpec, start_state, limit: int = PLAN_LIMIT):
    """Shortest clearing sequence from `start_state` inside `spec`."""
    if start_state.cleared:
        return []
    queue = deque([(start_state, [])])
    seen = {start_state}
    explored = 0
    while queue and explored < limit:
        state, path = queue.popleft()
        explored += 1
        for aid in (1, 2, 3, 4, 5):
            nxt = transition_state(state, Action(aid), spec)
            if nxt.dead or nxt in seen:
                continue
            if nxt.cleared:
                return [*path, aid]
            seen.add(nxt)
            queue.append((nxt, [*path, aid]))
    return None


def run_agent(truth_spec: WorldSpec, observed, size: int, seed: int,
              mode: str):
    """Run one agent episode. Returns (won, real_actions, reinf, exact).

    mode: 'full'    explore -> infer -> plan -> execute (with re-inference)
          'random'  no model, random actions
          'oracle'  plan inside the TRUE model (ceiling)
    """
    rng = np.random.default_rng(seed)
    world = GridWorld(truth_spec)
    world.reset()

    if mode == "random":
        actions = 0
        while not world.done and actions < 600:
            world.step(Action(int(rng.integers(1, 6))))
            actions += 1
        return world.done, actions, 0, None

    if mode == "oracle":
        plan = bfs_plan(truth_spec, initial_state(0, truth_spec))
        if plan is None:
            return False, 0, 0, True
        for aid in plan:
            if world.done:
                break
            world.step(Action(aid))
        return world.done, len(plan), 0, True

    # ---- full agent
    real_actions = 0
    reinf = 0
    inferred_exact = None
    truth_prog = _truth_program(truth_spec)

    for attempt in range(3):
        observed_now = _observed_from_world(world, size)

        # 1. EXPLORE
        cache = SpecCache(observed_now, size)
        order = observed_now.targets
        progs = [tuple(ax[int(rng.integers(0, len(ax)))] for ax in AXES)
                 for _ in range(300)]
        idxs = [tuple(ax.index(v) for ax, v in zip(AXES, p)) for p in progs]
        live = [(idx, initial_state(0, cache.get(idx, order))) for idx in idxs]
        explore_budget = EXPLORE_STEPS if attempt == 0 else 60
        before = len(world.history.steps)
        dsl_qbc_explore(world, observed_now, size, cache, order, live, rng,
                        explore_budget)
        real_actions += len(world.history.steps) - before

        # 2. INFER over ALL programs
        seg = scorable_segment(world.history)
        orders = ([tuple(p) for p in _perm(observed_now.targets)]
                  if 2 <= len(observed_now.targets) <= 4
                  else [observed_now.targets])
        live = [(idx, o, initial_state(0, cache.get(idx, o)))
                for idx in itertools.product(*[range(len(ax)) for ax in AXES])
                for o in orders]
        for step in seg.steps:
            if len(live) <= 1:
                break
            wa, wt, wg = _frame_facts(step.settled.grid, size)
            survivors = []
            for idx, o, state in live:
                nxt = transition_state(state, step.action, cache.get(idx, o))
                here = (nxt.x, nxt.y)
                visible = frozenset(t for t in nxt.remaining if t != here)
                if here != wa or visible != wt:
                    continue
                if wg is not None and bool(nxt.gates_open) != wg:
                    continue
                survivors.append((idx, o, nxt))
            live = survivors
        if not live:
            break
        best = min(live, key=lambda t: complexity(t[0]))
        best_idx, best_order = best[0], best[1]
        best_prog = tuple(ax[i] for ax, i in zip(AXES, best_idx))
        inferred_exact = (best_prog == truth_prog)
        if attempt > 0:
            reinf += 1
        inferred_spec = WorldSpec(
            world_id="inf", seed=0, field_size=size,
            mechanics=compile_program(best_prog),
            levels=(observed_now,),
        )

        # 3. PLAN inside the inferred model from the CURRENT state
        cur = _current_state(world, inferred_spec)
        plan = bfs_plan(inferred_spec, cur)
        if plan is None:
            continue  # believed stuck: gather more evidence

        # 4. EXECUTE with divergence checking
        state = cur
        diverged = False
        for aid in plan:
            if world.done:
                break
            predicted = transition_state(state, Action(aid), inferred_spec)
            world.step(Action(aid))
            real_actions += 1
            wa, wt, wg = _frame_facts(world.history.last.grid, size)
            here = (predicted.x, predicted.y)
            visible = frozenset(t for t in predicted.remaining if t != here)
            if here != wa or visible != wt or (
                    wg is not None and bool(predicted.gates_open) != wg):
                diverged = True
                break
            state = predicted
        if world.done:
            return True, real_actions, reinf, inferred_exact
        if not diverged:
            continue  # plan exhausted cleanly; model believes no win from here
    return world.done, real_actions, reinf, inferred_exact


def _observed_from_world(world: GridWorld, size: int) -> LevelSpec:
    """The layout as first seen -- layouts are static, so the initial frame
    of the episode is the layout for every inference round."""
    from sentinel.core.agent import read_layout
    return read_layout(world.history.initial.grid, size)


def _current_state(world: GridWorld, inferred_spec: WorldSpec, observed=None):
    st = initial_state(0, inferred_spec)
    for step in scorable_segment(world.history).steps:
        st = transition_state(st, step.action, inferred_spec)
    return st


def spent_cap(world):
    return 0  # exploration cost is counted inside dsl_qbc_explore's steps


def _truth_program(spec: WorldSpec) -> tuple:
    m = spec.mechanics
    return (m.step_distance, m.charge_period,
            m.effective_edge_mode(),
            (m.has_hazards, m.hazard_effect if m.has_hazards else "kill"),
            (m.has_switches, m.switch_mode if m.has_switches else "toggle"),
            m.ordered_targets, m.gates_start_open,
            m.wait_advances_charge)


def main() -> int:
    rng = np.random.default_rng(0)
    print(f"building {N_WORLDS} fresh held-out worlds...\n")
    worlds = []
    for wi in range(N_WORLDS):
        truth = sample_truth(rng)
        size = int(rng.integers(9, 14))
        level = make_identifiable_level(rng, size, truth)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )
        spec = WorldSpec(world_id=f"w{wi:02d}", seed=0, field_size=size,
                         mechanics=truth, levels=(observed,))
        worlds.append((spec, observed, size))

    results = {m: {"won": 0, "actions": 0, "reinf": 0, "exact": 0, "n": 0}
               for m in ("random", "full", "oracle")}
    t0 = time.perf_counter()

    for wi, (spec, observed, size) in enumerate(worlds):
        for mode in ("random", "full", "oracle"):
            won, actions, reinf, exact = run_agent(spec, observed, size,
                                                   seed=300 + wi, mode=mode)
            r = results[mode]
            r["won"] += won
            r["actions"] += actions
            r["reinf"] += reinf
            r["n"] += 1
            if exact:
                r["exact"] += 1
        print(f"  w{wi:02d}: done")

    print(f"\n{'agent':>8} {'solve rate':>11} {'avg actions':>12} "
          f"{'re-inferences':>14}")
    for mode in ("random", "full", "oracle"):
        r = results[mode]
        n = max(r["n"], 1)
        print(f"{mode:>8} {r['won']:6d}/{n} {r['actions'] / n:11.0f} "
              f"{r['reinf'] / n:13.1f}")

    full_won = results["full"]["won"]
    rand_won = results["random"]["won"]
    oracle_won = results["oracle"]["won"]
    print(f"\n(oracle ceiling: {oracle_won}/{N_WORLDS} -- worlds solvable "
          f"from the opening state within the plan limit)")
    print(f"(total {time.perf_counter() - t0:.0f}s)")

    print("\nverdict:")
    if full_won > rand_won and full_won >= oracle_won * 0.5:
        print("   THE AGENT WINS: inferring the world and planning inside it")
        print("   beats model-less action by a wide margin and approaches")
        print("   the oracle ceiling. This is the programme's product")
        print("   demonstration: not 'does it know the rules' but 'does")
        print("   knowing the rules let it win'.")
    elif full_won > rand_won:
        print("   The full agent beats random but sits below the oracle")
        print("   ceiling -- inference errors or exploration cost. Both are")
        print("   measured quantities with known levers.")
    else:
        print("   No advantage over random yet; inspect divergence rates and")
        print("   inference quality on the losing worlds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
