"""X31: diagnose the no-plan losses -- the answer changes the architecture.

X30's taxonomy: 10 of 13 losses were `no-plan`. This experiment re-runs
the agent while tracking, at every plan failure: whether the TRUE program
was still among refutation survivors, whether BFS exhausted its space or
hit the node limit, and whether the TRUE model has a solution from the
reached state.

MEASURED (same 40 worlds):

    solve rate: 27/40 = 68% +/- 15 (unchanged)

    loss causes:
      unsolvable-from-here   6   <- DOMINANT
      plan-diverged          3
      ranking-picked-wrong   2
      no-plan-clean          2

THE ANSWER IS NEITHER OF THE TWO HYPOTHESES. The inferred programs were
mostly RIGHT (only 2 ranking errors), and the planner never hit its limit.
The agent loses because it has already MOVED SOMEWHERE the remaining
targets cannot be collected from -- under ordered rules especially,
collecting in the wrong order dead-ends the world. Infer-then-solve paints
itself into corners: exploration gathers evidence without regard for
solvability of the remaining task.

THE ARCHITECTURAL FIX: interleave planning with exploration. Plan
continuously from the current evidence; explore only when no plan exists;
and among exploratory moves prefer those that keep the world solvable
under the surviving hypotheses. That is solvability-aware exploration --
a constraint on the explorer, not a new component. It composes with
everything built so far.
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
PLAN_LIMIT = 60_000


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


def bfs_plan_tracked(spec, start_state, limit=PLAN_LIMIT):
    """BFS that also reports whether it exhausted the search space."""
    if start_state.cleared:
        return [], False
    queue = deque([(start_state, [])])
    seen = {start_state}
    explored = 0
    exhausted = True
    while queue and explored < limit:
        state, path = queue.popleft()
        explored += 1
        for aid in (1, 2, 3, 4, 5):
            nxt = transition_state(state, Action(aid), spec)
            if nxt.dead or nxt in seen:
                continue
            if nxt.cleared:
                return [*path, aid], False
            seen.add(nxt)
            queue.append((nxt, [*path, aid]))
    if queue:
        exhausted = False  # hit the node limit with frontier remaining
    return None, exhausted


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

    causes: dict[str, int] = {}
    lost_detail = []
    won_n = 0

    for wi, (spec, observed, size) in enumerate(worlds):
        truth_prog = truth_program(spec)
        truth_idx = tuple(ax.index(v) for ax, v in zip(AXES, truth_prog))
        rng = np.random.default_rng(300 + wi)
        world = GridWorld(spec)
        world.reset()
        actions = 0
        diagnosis = "won"

        for attempt in range(MAX_ROUNDS):
            obs_now = observed_from(world, size)
            cache = SpecCache(obs_now, size)
            order = obs_now.targets
            progs = [tuple(ax[int(rng.integers(0, len(ax)))] for ax in AXES)
                     for _ in range(300)]
            idxs = [tuple(ax.index(v) for ax, v in zip(AXES, p))
                    for p in progs]
            live_c = [(idx, initial_state(0, cache.get(idx, order)))
                      for idx in idxs]
            budget = EXPLORE_STEPS if attempt == 0 else TOPUP_STEPS
            before = len(world.history.steps)
            dsl_qbc_explore(world, obs_now, size, cache, order, live_c, rng,
                            budget)
            actions += len(world.history.steps) - before

            if world.done:
                break

            seg = scorable_segment(world.history)
            orders = ([tuple(p) for p in _perm(obs_now.targets)]
                      if 2 <= len(obs_now.targets) <= 4
                      else [obs_now.targets])
            live = [(idx, o, initial_state(0, cache.get(idx, o)))
                    for idx in itertools.product(
                        *[range(len(ax)) for ax in AXES])
                    for o in orders]
            for step in seg.steps:
                if len(live) <= 1:
                    break
                wa, wt, wg = _frame_facts(step.settled.grid, size)
                survivors = []
                for idx, o, state in live:
                    nxt = transition_state(state, step.action,
                                           cache.get(idx, o))
                    here = (nxt.x, nxt.y)
                    visible = frozenset(t for t in nxt.remaining
                                        if t != here)
                    if here != wa or visible != wt:
                        continue
                    if wg is not None and bool(nxt.gates_open) != wg:
                        continue
                    survivors.append((idx, o, nxt))
                live = survivors
            if not live:
                diagnosis = "refutation-empty"
                break

            surv_idxs = [idx for idx, _, _ in live]
            truth_in = truth_idx in surv_idxs

            best = min(live, key=lambda t: complexity(t[0]))
            best_prog = tuple(ax[i] for ax, i in zip(AXES, best[0]))
            inferred_spec = WorldSpec(
                world_id="inf", seed=0, field_size=size,
                mechanics=compile_program(best_prog),
                levels=(obs_now,),
            )

            cur = current_state(world, inferred_spec)
            plan, exhausted = bfs_plan_tracked(inferred_spec, cur)

            if plan is None:
                # THE DIAGNOSTIC MOMENT: was the truth still alive here?
                if not truth_in:
                    diagnosis = "ranking-picked-wrong"
                elif exhausted:
                    diagnosis = "planner-limit"
                else:
                    # truth alive, planner exhausted space: the inferred
                    # program's world genuinely has no solution from here
                    # (wrong program whose world is smaller/deader), OR the
                    # true world also lacks one -- check the true model.
                    true_cur = current_state(world, spec)
                    true_plan = bfs_plan_tracked(spec, true_cur)[0]
                    if true_plan is None:
                        diagnosis = "unsolvable-from-here"
                    else:
                        diagnosis = "ranking-picked-wrong"
                # top up evidence and retry
                continue

            state = cur
            diverged = False
            for aid in plan:
                if world.done:
                    break
                predicted = transition_state(state, Action(aid),
                                             inferred_spec)
                world.step(Action(aid))
                actions += 1
                wa, wt, wg = _frame_facts(world.history.last.grid, size)
                here = (predicted.x, predicted.y)
                visible = frozenset(t for t in predicted.remaining
                                    if t != here)
                if here != wa or visible != wt or (
                        wg is not None and bool(predicted.gates_open) != wg):
                    diverged = True
                    break
                state = predicted
            if world.done:
                diagnosis = "won"
                break
            if diverged:
                diagnosis = "plan-diverged"
                continue
            diagnosis = "no-plan-clean"

        if diagnosis == "won":
            won_n += 1
        else:
            causes[diagnosis] = causes.get(diagnosis, 0) + 1
            lost_detail.append((wi, diagnosis, actions))
        print(f"  w{wi:02d}: {'WON' if diagnosis == 'won' else diagnosis}")

    k = N_WORLDS
    se = np.sqrt(won_n / k * (1 - won_n / k) / k)
    print(f"\nsolve rate: {won_n}/{k} = {100.0 * won_n / k:.0f}% "
          f"+/- {100 * 1.96 * se:.0f} (95% CI)")
    print("loss causes:")
    for cause, count in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"   {cause:24} {count:3d}")
    print(f"(total {time.perf_counter() - t0:.0f}s)")

    print("\nverdict:")
    rw = causes.get("ranking-picked-wrong", 0)
    pl = causes.get("planner-limit", 0)
    uf = causes.get("unsolvable-from-here", 0)
    if rw >= pl and rw >= uf and rw > 0:
        print("   DOMINANT: ranking picked wrong. The derivation-core ranker")
        print("   (or more exploration before inferring) is the highest-value")
        print("   next investment.")
    elif pl > rw:
        print("   DOMINANT: planner node limit. A* or hierarchical planning")
        print("   is the next investment.")
    elif uf > 0:
        print("   DOMINANT: worlds genuinely unsolvable from the reached")
        print("   state -- exploration must solve-as-it-goes rather than")
        print("   infer-then-solve.")
    else:
        print("   Losses spread across causes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
