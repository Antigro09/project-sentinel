"""X32: solvability-aware exploration -- plan continuously, explore only
when stuck, and never take an exploratory action that dead-ends the
inferred world.

MEASURED (40 fresh held-out worlds):

    solvability-aware: 26/40 = 65% +/- 15, avg 223 actions
    infer-then-solve:  27/40 = 68% +/- 15, avg 229 actions

NO LIFT on solve rate -- statistically identical. But the cost profile
transformed: wins now cost 4-33 actions (median ~16) versus 200+ under
infer-then-solve, because the agent solves as soon as it can instead of
exploring a fixed 200 steps first. Losses burn the full 600-step cap with
repeated divergences (up to 19) -- worlds where the inferred model keeps
being wrong.

THE HONEST READING: X31's diagnosis was right about the mechanism
(painting yourself into corners) but the cure does not lift solve rate,
because the binding constraint was never exploration order -- it is
INFERENCE QUALITY on behaviourally-subtle axes. The high-divergence losses
are worlds where refutation cannot pin the true program from the available
evidence; no amount of solvability-awareness fixes a wrong model.

ALSO FIXED EN ROUTE: the first version of this experiment re-refuted the
ENTIRE episode on every agent step (~360M transitions per world -- it ran
for hours). Incremental refutation (keep survivors and states persistent,
refute only the new transition) cut that to seconds. The lesson mirrors
adapt/search.equivalence_search: never replay what you can advance.

WHERE THIS LEAVES THE LOOP: 65-68% solve rate is the current ceiling, and
the binding constraint is inference quality on subtle axes -- the same
conclusion as the charge arc. The next lever is better survivor ranking
(the derivation core, X21-X23) or more informative exploration, not
planning changes.
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
MAX_STEPS = 600
PLAN_LIMIT = 60_000


def _frame_facts(grid, field_size):
    from x17_dsl_search import _frame_facts as ff
    return ff(grid, field_size)


def bfs_plan(spec, start_state, limit=PLAN_LIMIT):
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


def refute_all(observed, size, cache, order):
    """All programs, unrefuted (initial state); refutation happens live."""
    return [(idx, o, initial_state(0, cache.get(idx, o)))
            for idx in itertools.product(*[range(len(ax)) for ax in AXES])
            for o in ([tuple(p) for p in _perm(observed.targets)]
                      if 2 <= len(observed.targets) <= 4
                      else [observed.targets])]


def refute_step(live, step, cache, order, field_size):
    wa, wt, wg = _frame_facts(step.settled.grid, field_size)
    survivors = []
    for idx, o, state in live:
        try:
            nxt = transition_state(state, step.action, cache.get(idx, o))
        except Exception:
            continue
        here = (nxt.x, nxt.y)
        visible = frozenset(t for t in nxt.remaining if t != here)
        if here != wa or visible != wt:
            continue
        if wg is not None and bool(nxt.gates_open) != wg:
            continue
        survivors.append((idx, o, nxt))
    return survivors


def solvable_from(spec, state, limit=2_000):
    """Does ANY clearing sequence exist from `state` inside `spec`?"""
    if state.cleared:
        return True
    queue = deque([state])
    seen = {state}
    explored = 0
    while queue and explored < limit:
        s = queue.popleft()
        explored += 1
        for aid in (1, 2, 3, 4, 5):
            nxt = transition_state(s, Action(aid), spec)
            if nxt.dead or nxt in seen:
                continue
            if nxt.cleared:
                return True
            seen.add(nxt)
            queue.append(nxt)
    return False


def run_solvability_agent(truth_spec: WorldSpec, seed: int):
    """Solvability-aware agent: plan when possible, explore safely when not."""
    rng = np.random.default_rng(seed)
    world = GridWorld(truth_spec)
    world.reset()
    actions = 0
    reinf = 0
    last_failure = "unknown"

    # THE PERFORMANCE FIX. The first version rebuilt `live` from scratch and
    # re-refuted the ENTIRE episode on every agent step: 600 steps x ~600k
    # transitions = ~360M transitions per world, which is why it hung for
    # hours. Incremental refutation keeps survivors and their states
    # persistent; each new real transition costs ONE pass over the current
    # survivor set (which collapses fast), not a full replay.
    observed = observed_from(world, truth_spec.field_size)
    size = truth_spec.field_size
    cache = SpecCache(observed, size)
    order = observed.targets
    live = [(idx, o, initial_state(0, cache.get(idx, o)))
            for idx in itertools.product(*[range(len(ax)) for ax in AXES])
            for o in ([tuple(p) for p in _perm(observed.targets)]
                      if 2 <= len(observed.targets) <= 4
                      else [observed.targets])]

    while not world.done and actions < MAX_STEPS:
        if not live:
            last_failure = "refutation-empty"
            break

        # Select the simplest survivor and check for a plan.
        best = min(live, key=lambda t: complexity(t[0]))
        best_prog = tuple(ax[i] for ax, i in zip(AXES, best[0]))
        inferred_spec = WorldSpec(
            world_id="inf", seed=0, field_size=size,
            mechanics=compile_program(best_prog),
            levels=(observed,),
        )
        cur = current_state(world, inferred_spec)
        plan = bfs_plan(inferred_spec, cur)

        if plan:
            # Follow ONE step of the plan, divergence-checked.
            aid = plan[0]
            predicted = transition_state(cur, Action(aid), inferred_spec)
            world.step(Action(aid))
            actions += 1
            wa, wt, wg = _frame_facts(world.history.last.grid, size)
            here = (predicted.x, predicted.y)
            visible = frozenset(t for t in predicted.remaining
                                if t != here)
            diverged = (here != wa or visible != wt or (
                wg is not None and bool(predicted.gates_open) != wg))
            if diverged:
                reinf += 1
            # INCREMENTAL refutation: one new transition, one pass over
            # survivors. Survivors' states advance with reality.
            live = refute_step(live, world.history.steps[-1], cache, order,
                               size)
            continue

        # No plan: explore, but SOLVABILITY-AWARE.
        # Simulate each candidate action under the simplest survivor; keep
        # actions after which the inferred world still has a solution.
        cands = []
        for aid in (1, 2, 3, 4, 5):
            nxt = transition_state(cur, Action(aid), inferred_spec)
            if nxt.dead:
                continue
            if solvable_from(inferred_spec, nxt):
                cands.append(aid)
        if not cands:
            cands = [1, 2, 3, 4, 5]  # nothing safe under the model; gamble
        # Among safe candidates prefer max disagreement across survivors.
        # Each survivor's CURRENT state is already tracked in `live`.
        scored = []
        for idx, o, state in live[:200]:
            sp = WorldSpec(world_id="t", seed=0, field_size=size,
                           mechanics=compile_program(
                               tuple(ax[i] for ax, i in zip(AXES, idx))),
                           levels=(observed,))
            scored.append((sp, state))
        best_aid, best_split = None, -1
        for aid in cands:
            outs = set()
            for sp, st in scored:
                nxt = transition_state(st, Action(aid), sp)
                outs.add(state_key(nxt))
            if len(outs) > best_split:
                best_aid, best_split = aid, len(outs)
        world.step(Action(best_aid if best_aid else int(rng.choice(cands))))
        actions += 1
        # Incremental refutation of the exploratory action too.
        live = refute_step(live, world.history.steps[-1], cache, order, size)

    if world.done:
        return True, actions, "won", reinf
    return False, actions, f"step-limit ({last_failure})", reinf


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
    for wi, (spec, observed, size) in enumerate(worlds):
        won, actions, outcome, reinf = run_solvability_agent(spec,
                                                             seed=300 + wi)
        won_n += won
        actions_list.append(actions)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        print(f"  w{wi:02d}: {'WON ' if won else 'LOST'} ({actions:3d} "
              f"actions, {reinf} divergences)")

    k = N_WORLDS
    se = np.sqrt(won_n / k * (1 - won_n / k) / k)
    print(f"\nsolve rate: {won_n}/{k} = {100.0 * won_n / k:.0f}% "
          f"+/- {100 * 1.96 * se:.0f} (95% CI)")
    print(f"avg actions: {np.mean(actions_list):.0f}")
    print("outcomes:")
    for outcome, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        print(f"   {outcome:24} {count:3d}")
    print(f"(total {time.perf_counter() - t0:.0f}s)")

    base = 27 / 40
    new = won_n / k
    print("\nverdict:")
    if new > base + 0.08:
        print(f"   SOLVABILITY-AWARE EXPLORATION WINS: {base:.0%} -> "
              f"{new:.0%}. The X31 fix works: solving continuously and")
        print("   exploring only when stuck keeps the agent out of dead")
        print("   ends. This is the production configuration.")
    elif new > base:
        print(f"   Modest lift ({base:.0%} -> {new:.0%}); direction right,")
        print("   magnitude limited by remaining failure modes.")
    else:
        print("   No lift over infer-then-solve; inspect whether the")
        print("   solvability filter itself misleads (wrong inferred model).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
