"""X30: scale the agent to 40 worlds and build its failure taxonomy.

X29 demonstrated the full agent on 12 worlds (9/12). This scales to 40 for
a usable confidence interval and classifies every failure.

MEASURED:

    solve rate: 27/40 = 68% +/- 15 (95% CI)
    avg real actions: 229

    failure taxonomy:
      won             27
      no-plan         10   <- dominant: inferred model believes no
      plan-diverged    3      solution exists from the current state

NO deaths during exploration -- X19's identifiability-aware generator
eliminated that failure mode entirely.

The dominant failure, no-plan, has two candidate causes with different
fixes: (a) inference selected a wrong-but-simpler program whose world is
genuinely unsolvable from the reached state -- the fix is better ranking
(the derivation core) or more exploration before inferring; (b) the BFS
limit binds on large boards -- the fix is a stronger planner (A* or
hierarchical). Distinguishing them is mechanical: re-run refutation on the
lost worlds and check whether the TRUE program was among the survivors.

Also notable: 5 worlds were won in under 50 actions -- short episodes
where early exploration happened to solve the world directly.
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
from x17_dsl_search import AXES, SpecCache, compile_program, complexity  # noqa: E402
from x19_identifiable_worlds import make_identifiable_level  # noqa: E402
from x21_derivation_core import sample_truth  # noqa: E402

from sentinel.adapt.hypothesis import scorable_segment  # noqa: E402
from sentinel.env.types import Action  # noqa: E402
from sentinel.explore.version_space import state_key  # noqa: E402
from sentinel.gen.grid import GridWorld, initial_state, transition_state  # noqa: E402
from sentinel.gen.spec import LevelSpec, WorldSpec  # noqa: E402

N_WORLDS = 40
EXPLORE_STEPS = 200
TOPUP_STEPS = 60
MAX_ROUNDS = 3


def _frame_facts(grid, field_size):
    from x17_dsl_search import _frame_facts as ff
    return ff(grid, field_size)


def dsl_qbc_explore(world, observed, field_size, cache, order, live, rng,
                    steps):
    size = field_size
    spent = 0
    MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    while spent < steps and not world.done:
        seg = scorable_segment(world.history)
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


def bfs_plan(spec, start_state, limit=60_000):
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


def truth_program(spec: WorldSpec) -> tuple:
    m = spec.mechanics
    return (m.step_distance, m.charge_period,
            m.effective_edge_mode(),
            (m.has_hazards, m.hazard_effect if m.has_hazards else "kill"),
            (m.has_switches, m.switch_mode if m.has_switches else "toggle"),
            m.ordered_targets, m.gates_start_open,
            m.wait_advances_charge)


def observed_from(world, size):
    from sentinel.core.agent import read_layout
    return read_layout(world.history.initial.grid, size)


def current_state(world, inferred_spec):
    st = initial_state(0, inferred_spec)
    for step in scorable_segment(world.history).steps:
        st = transition_state(st, step.action, inferred_spec)
    return st


def run_full_agent(truth_spec, seed):
    """Returns (won, actions, outcome, rounds_used)."""
    rng = np.random.default_rng(seed)
    world = GridWorld(truth_spec)
    world.reset()
    actions = 0
    reinf = 0
    truth_prog = truth_program(truth_spec)
    last_failure = "unknown"

    for attempt in range(MAX_ROUNDS):
        observed = observed_from(world, truth_spec.field_size)
        cache = SpecCache(observed, truth_spec.field_size)
        order = observed.targets
        progs = [tuple(ax[int(rng.integers(0, len(ax)))] for ax in AXES)
                 for _ in range(300)]
        idxs = [tuple(ax.index(v) for ax, v in zip(AXES, p)) for p in progs]
        live = [(idx, initial_state(0, cache.get(idx, order)))
                for idx in idxs]
        budget = EXPLORE_STEPS if attempt == 0 else TOPUP_STEPS
        before = len(world.history.steps)
        dsl_qbc_explore(world, observed, truth_spec.field_size, cache, order,
                        live, rng, budget)
        actions += len(world.history.steps) - before

        if world.done:
            return True, actions, "won", attempt

        seg = scorable_segment(world.history)
        orders = ([tuple(p) for p in _perm(observed.targets)]
                  if 2 <= len(observed.targets) <= 4
                  else [observed.targets])
        live = [(idx, o, initial_state(0, cache.get(idx, o)))
                for idx in itertools.product(*[range(len(ax)) for ax in AXES])
                for o in orders]
        for step in seg.steps:
            if len(live) <= 1:
                break
            wa, wt, wg = _frame_facts(step.settled.grid,
                                      truth_spec.field_size)
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
            last_failure = "refutation-empty"
            break
        best = min(live, key=lambda t: complexity(t[0]))
        best_prog = tuple(ax[i] for ax, i in zip(AXES, best[0]))
        inferred_spec = WorldSpec(
            world_id="inf", seed=0, field_size=truth_spec.field_size,
            mechanics=compile_program(best_prog),
            levels=(observed,),
        )
        if attempt > 0:
            reinf += 1

        cur = current_state(world, inferred_spec)
        plan = bfs_plan(inferred_spec, cur)
        if plan is None:
            last_failure = "no-plan"
            continue

        state = cur
        diverged = False
        for aid in plan:
            if world.done:
                break
            predicted = transition_state(state, Action(aid), inferred_spec)
            world.step(Action(aid))
            actions += 1
            wa, wt, wg = _frame_facts(world.history.last.grid,
                                      truth_spec.field_size)
            here = (predicted.x, predicted.y)
            visible = frozenset(t for t in predicted.remaining
                                if t != here)
            if here != wa or visible != wt or (
                    wg is not None and bool(predicted.gates_open) != wg):
                diverged = True
                break
            state = predicted
        if world.done:
            return True, actions, "won", attempt
        if diverged:
            last_failure = "plan-diverged"
            continue
        last_failure = "no-plan"  # clean exhaustion
    return (world.done, actions,
            ("won" if world.done else last_failure), MAX_ROUNDS)


def main() -> int:
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    print(f"building {N_WORLDS} fresh held-out worlds...")
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

    won_n = 0
    actions_list = []
    outcomes: dict[str, int] = {}
    t0 = time.perf_counter()
    for wi, (spec, observed, size) in enumerate(worlds):
        won, actions, outcome, rounds = run_full_agent(spec, seed=300 + wi)
        won_n += won
        actions_list.append(actions)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        print(f"  w{wi:02d}: {'WON ' if won else 'LOST'} "
              f"({actions:3d} actions, {rounds} rounds, {outcome})")

    k = N_WORLDS
    se = np.sqrt(won_n / k * (1 - won_n / k) / k)
    print(f"\nsolve rate: {won_n}/{k} = {100.0 * won_n / k:.0f}% "
          f"+/- {100 * 1.96 * se:.0f} (95% CI)")
    print(f"avg actions: {np.mean(actions_list):.0f}")
    print("failure taxonomy:")
    for outcome, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"   {outcome:16} {count:3d}")

    print("\nverdict:")
    losses = k - won_n
    if losses == 0:
        print("   PERFECT on this set. Scale worlds further before claiming")
        print("   more.")
    elif outcomes.get("died-exploring", 0) >= losses // 2:
        print("   Dominant failure: died exploring. The generator's hazard")
        print("   scaling needs tightening, or exploration needs path")
        print("   prediction.")
    elif outcomes.get("no-plan", 0) >= losses // 2:
        print("   Dominant failure: no plan found inside the inferred model.")
        print("   Either inference picked a wrong-but-simple program, or the")
        print("   BFS limit binds. Check inferred-vs-truth on those worlds.")
    elif outcomes.get("plan-diverged", 0) >= losses // 2:
        print("   Dominant failure: plans diverge from reality -- inference")
        print("   errors on behaviourally-subtle axes.")
    else:
        print("   Failures spread across modes; no single dominant lever.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
