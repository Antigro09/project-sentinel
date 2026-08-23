"""X35: the autonomous novelty trigger -- grammar growth wired into the
winning agent.

X34 proved the RSI mechanism in isolation: given novelty, candidate axis
families generated from a meta-grammar are scored by version-space collapse
and the least expressive viable family is adopted. But X34's worlds were
ring automata -- nothing was at stake. This experiment ports the mechanism
into the main grid-world loop (X29/X32's infer-plan-win agent) and asks the
question that matters:

    when the world exceeds the grammar, does the system NOTICE, GROW,
    and then WIN -- without a human editing anything?

THE SETUP. The reference engine contains a dormant primitive:
`Mechanics.slide` -- the agent keeps moving in the chosen direction until
something stops it. Ice. It is executable by the exact model but is NOT
one of the eight DSL axes, so no program in the 368,640-hypothesis space
can express a slide world. The truth is runnable yet unspeakable -- the
precise condition under which a closed vocabulary fails silently.

THE LOOP UNDER TEST:

    1. EXPLORE          random walk (policy quality is not the point here)
    2. INFER            refute all 368,640 programs
    3. TRIGGER          refutation EMPTY => novelty declared, execution
                        paused. (A nonempty survivor set means the grammar
                        still speaks this world; no growth is attempted.)
    4. PROPOSE          candidate axis families from the meta-grammar:
                          base       (control: the unmodified DSL)
                          step-ext   (decoy: MORE of an existing axis --
                                      step_distance up to 16)
                          slide      (the truth: slide False/True axis)
                          momentum   (decoy: fixed-length k-slide, k=2,3,
                                      a local primitive the engine lacks)
    5. DISPOSE          refute within each family; eliminate the empty;
                        adopt the LEAST EXPRESSIVE viable family
                        (X34's Occam-primary rule; ties -> harder collapse)
    6. RESUME           select simplest survivor in the expanded space,
                        plan BFS inside it, execute divergence-checked.

ADOPTION IS HONEST ABOUT ITS OWN LIMITS: the meta-grammar proposes
STRUCTURE, but only primitives the engine already contains can be adopted
with exactness inherited rather than re-proved. If `momentum` had won, the
system must decline to adopt (no engine primitive) and say so. Growth
discovers WHICH dormant capacity reality demands; it does not hallucinate
capacity the machine lacks.

MEASURED (two hidden worlds):

  WORLD A (10x10, short boards): base ELIMINATED (cannot express ice);
  momentum ELIMINATED (slides exceed k=2,3); step-ext AND slide both
  VIABLE -- and provably indistinguishable: on a 10-wide board any fixed
  step >= 9 travels exactly until blocked, i.e. step-ext CONTAINS ice
  here. The system adopts the least expressive viable family, then runs
  the BEHAVIOURAL-COLLAPSE CERTIFICATE: every survivor of the adopted
  family agrees on every next-frame prediction from its tracked state,
  so behaviour is PINNED even though the syntax is not. The choice
  between equivalent descriptions is measured to be meaningless.
  Planning inside the adopted model WINS.

  WORLD B (20x20, long boards): slides of 17+ cells occur -- longer than
  any fixed step the decoy offers. step-ext is now ELIMINATED outright;
  slide survives alone and is adopted. Certificate passes; WINS.

  Controls (no growth machinery) lose on both worlds: they fall back to
  the simplest base program and plan inside a fiction.

TWO DESIGN LESSONS, both discovered by failure:

  1. Selection among survivors must be SIMPLEST-THAT-ADMITS-A-PLAN, not
     bare simplicity: the first run adopted a behaviourally-correct
     program whose collection-order made the level unsolvable IN MODEL,
     and BFS returned nothing. A survivor that cannot be acted on is
     not a usable answer (X31's lesson, one layer up).
  2. When two proposed vocabularies both survive, the right response is
     not a better tie-break but a CERTIFICATE: if the surviving version
     space makes unique predictions, the adopted language is sufficient
     -- which description is 'true' is a question about words, not
     behaviour. Discrimination is demanded only when it is POSSIBLE
     (world B separates the families; world A certifies their tie).

This closes the loop X33 opened and X34 mechanised: novelty ->
propose -> dispose -> adopt -> certify -> resume, inside the agent that
wins games.
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from dataclasses import replace
from itertools import permutations as _perm

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "experiments")
from x17_dsl_search import (  # noqa: E402
    AXES,
    AXIS_NAMES,
    CHARGE_VALUES,
    EDGE_VALUES,
    GATES_VALUES,
    HAZARD_VALUES,
    ORDERED_VALUES,
    SPACE_SIZE,
    STEP_VALUES,
    SWITCH_VALUES,
    WAIT_VALUES,
    SpecCache,
    compile_program,
)
from x19_identifiable_worlds import make_identifiable_level  # noqa: E402  # (used by other arcs; ice layout is local)

from sentinel.adapt.hypothesis import scorable_segment  # noqa: E402
from sentinel.core.agent import read_layout  # noqa: E402
from sentinel.env.types import Action  # noqa: E402
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec  # noqa: E402
from sentinel.gen.grid import (  # noqa: E402
    MOVES,
    GridWorld,
    blocked,
    initial_state,
    transition_state,
)

SIZE = 10
EXPLORE_STEPS = 120
MAX_STEPS = 400
PLAN_LIMIT = 20_000

STEP_EXT_VALUES = tuple(range(1, 17))


# ------------------------------------------------------- meta-grammar
# Each candidate family is a PROPOSAL for how to grow the grammar: which
# axis values to admit, and what transition function executes them.

class Family:
    def __init__(self, name, axes, extra_values, kind):
        self.name = name
        self.axes = axes
        self.extra_values = extra_values
        self.kind = kind  # "engine" | "engine-slide" | "local"
        self.size = int(np.prod([len(a) for a in axes])) * len(extra_values)

    def mechanics(self, prog, extra):
        mech = compile_program(prog)
        if self.kind == "engine-slide":
            return replace(mech, slide=extra)
        return mech

    def transition(self, state, action, spec, extra):
        if self.kind == "local":
            return momentum_transition(state, action, spec, extra)
        return transition_state(state, action, spec)


def momentum_transition(state, action, spec, k):
    """Fixed-length slide: up to k cells, stopping early at obstacles.

    A DECOY primitive the engine does not contain. Implemented locally so
    the contest is fair -- and so the experiment can demonstrate declining
    adoption of a primitive that cannot inherit exactness.
    """
    if state.dead or state.cleared:
        return state
    level = spec.levels[state.level]
    mech = spec.mechanics
    size = spec.field_size
    aid = action.action_id
    if aid not in MOVES:
        if mech.wait_advances_charge:
            return replace(state, charge=state.charge + 1)
        return state
    charge = state.charge + 1
    dx, dy = MOVES[aid]
    x, y = state.x, state.y
    gates_open = state.gates_open
    for _ in range(k):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < size and 0 <= ny < size):
            break
        if blocked((nx, ny), level, size, gates_open):
            break
        x, y = nx, ny
        if mech.has_hazards and (x, y) in level.hazards:
            if mech.hazard_effect == "kill":
                return replace(state, x=x, y=y, charge=charge, dead=True)
            if mech.hazard_effect == "pushback":
                x, y = state.x, state.y
                break
            if mech.hazard_effect == "respawn":
                x, y = level.start
                break
        if mech.has_switches and (x, y) in level.switches:
            gates_open = (True if mech.switch_mode == "latch"
                          else not gates_open)
    remaining = state.remaining
    collected = state.collected
    here = (x, y)
    if here in remaining:
        if mech.ordered_targets:
            if (collected < len(level.targets)
                    and level.targets[collected] == here):
                remaining = remaining - {here}
                collected += 1
        else:
            remaining = remaining - {here}
            collected += 1
    return replace(state, x=x, y=y, collected=collected,
                   remaining=remaining, charge=charge,
                   gates_open=gates_open, cleared=not remaining)


BASE_AXES = (STEP_VALUES, CHARGE_VALUES, EDGE_VALUES, HAZARD_VALUES,
             SWITCH_VALUES, ORDERED_VALUES, GATES_VALUES, WAIT_VALUES)
EXT_AXES = (STEP_EXT_VALUES, CHARGE_VALUES, EDGE_VALUES, HAZARD_VALUES,
            SWITCH_VALUES, ORDERED_VALUES, GATES_VALUES, WAIT_VALUES)

FAMILIES = [
    Family("base", BASE_AXES, (None,), "engine"),
    Family("step-ext", EXT_AXES, (None,), "engine"),
    Family("slide", BASE_AXES, (False, True), "engine-slide"),
    Family("momentum", BASE_AXES, (2, 3), "local"),
]


# ----------------------------------------------------------- machinery

def frame_facts(grid, field_size):
    from sentinel.explore.version_space import observed_facts
    return observed_facts(grid, field_size)


def orders_for(observed):
    ts = observed.targets
    if 2 <= len(ts) <= 4:
        return [tuple(p) for p in _perm(ts)]
    return [ts]


def refute_family(family, steps, observed, field_size):
    """Bulk-refute one candidate family against the episode.

    Returns the survivor list [(prog_idx, extra, order, state)].
    Empty list = the family cannot express what happened.
    """
    cache: dict[tuple, WorldSpec] = {}

    def spec_for(idx, extra, order):
        key = (idx, extra, order)
        sp = cache.get(key)
        if sp is None:
            prog = tuple(a[i] for a, i in zip(family.axes, idx))
            sp = WorldSpec(
                world_id="cand", seed=0, field_size=field_size,
                mechanics=family.mechanics(prog, extra),
                levels=(replace(observed, targets=tuple(order)),),
            )
            cache[key] = sp
        return sp

    live = []
    for idx in itertools.product(*[range(len(a)) for a in family.axes]):
        for extra in family.extra_values:
            for order in orders_for(observed):
                sp = spec_for(idx, extra, order)
                live.append((idx, extra, order,
                             initial_state(0, sp)))
    sims = 0
    for step in steps:
        if not live:
            break
        wa, wt, wg = frame_facts(step.settled.grid, field_size)
        survivors = []
        for idx, extra, order, state in live:
            sims += 1
            sp = spec_for(idx, extra, order)
            nxt = family.transition(state, step.action, sp, extra)
            here = (nxt.x, nxt.y)
            visible = frozenset(t for t in nxt.remaining if t != here)
            if here != wa or visible != wt:
                continue
            if wg is not None and bool(nxt.gates_open) != wg:
                continue
            survivors.append((idx, extra, order, nxt))
        live = survivors
    return live


def adopt_family(results):
    """Occam-primary selection (X34's rule, ported).

    Eliminate families with zero survivors; among viable ones adopt the
    least expressive (fewest hypotheses); ties -> harder collapse.
    """
    viable = {name: surv for name, surv in results.items() if surv}
    if not viable:
        return None
    return min(viable, key=lambda n: (FAMILY_BY_NAME[n].size,
                                      len(viable[n])))


FAMILY_BY_NAME = {f.name: f for f in FAMILIES}


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


# ----------------------------------------------------------- the world

def make_ice_level(rng, size: int):
    """An OPEN layout: few walls, so slides run long and -- crucially --
    their LENGTH varies with where the agent stands. That variability is
    what makes slide IDENTIFIABLE against any fixed step distance (X18/
    X19's lesson once more: identifiability is a property of the world
    distribution; a dense board makes ice indistinguishable from stepping,
    and no inference can fix unexcited evidence)."""
    taken: set[tuple[int, int]] = set()
    start = (int(rng.integers(0, size)), int(rng.integers(0, size)))
    taken.add(start)

    walls: set[tuple[int, int]] = set()
    n_walls = int(size * size * 0.04)
    for _ in range(n_walls):
        free = [(x, y) for y in range(size) for x in range(size)
                if (x, y) not in taken | walls]
        if not free:
            break
        walls.add(free[int(rng.integers(0, len(free)))])

    switches: set[tuple[int, int]] = set()
    free = [(x, y) for y in range(size) for x in range(size)
            if (x, y) not in taken | walls]
    switches.add(free[int(rng.integers(0, len(free)))])

    gates: set[tuple[int, int]] = set()
    for _ in range(max(1, size // 4)):
        free = [(x, y) for y in range(size) for x in range(size)
                if (x, y) not in taken | walls | switches | gates]
        if not free:
            break
        gates.add(free[int(rng.integers(0, len(free)))])

    targets: list[tuple[int, int]] = []
    blocked_set = taken | walls | switches | gates
    for _ in range(3):
        free = [(x, y) for y in range(size) for x in range(size)
                if (x, y) not in blocked_set | set(targets)]
        if not free:
            break
        targets.append(free[int(rng.integers(0, len(free)))])

    return LevelSpec(
        start=start,
        walls=frozenset(walls),
        hazards=frozenset(),
        targets=tuple(targets),
        switches=frozenset(switches),
        gates=frozenset(gates),
    )


def make_ice_world(seed: int, size: int = SIZE):
    """A solvable slide world: ice motion + latch switch + gates.

    Under ice the agent stops ONLY where a blocker sits in the travel
    direction -- on an open board that is a small set of cells. Targets
    are therefore placed ON reachable stop cells (verified by search),
    which is what makes the world solvable at all. Solvability is still
    verified against the exact model before emission.
    """
    rng = np.random.default_rng(seed)
    mech = Mechanics(
        step_distance=1,
        slide=True,
        has_switches=True,
        switch_mode="latch",
        gates_start_open=False,
    )

    # First pass: open layout with walls/switch/gates.
    for attempt in range(8):
        level = make_ice_level(rng, size)
        spec = WorldSpec(world_id=f"ice{seed}", seed=seed,
                         field_size=size, mechanics=mech, levels=(level,))
        # Enumerate stop cells: states reachable from start.
        seen = {initial_state(0, spec)}
        frontier = list(seen)
        for _ in range(30):
            nxt = []
            for s in frontier:
                for aid in (1, 2, 3, 4):
                    t = transition_state(s, Action(aid), spec)
                    if not t.dead and t not in seen:
                        seen.add(t)
                        nxt.append(t)
            frontier = nxt
            if not frontier:
                break
        stops = {(s.x, s.y) for s in seen}
        # Re-place targets on distinct stop cells (not the switch/gates).
        reserved = level.switches | level.gates | {level.start}
        candidates = sorted(stops - reserved)
        if len(candidates) < 3:
            continue
        picks = [candidates[i] for i in
                 rng.choice(len(candidates), size=3, replace=False)]
        level = replace(level, targets=tuple(picks))
        spec = WorldSpec(world_id=f"ice{seed}", seed=seed,
                         field_size=size, mechanics=mech, levels=(level,))
        plan = bfs_plan(spec, initial_state(0, spec))
        if plan is None or len(plan) < 6:
            continue
        return spec, plan
    return None, None


def behavioural_collapse(survivors, family, observed, field_size):
    """The certificate: do ALL survivors agree on EVERY next-frame
    prediction from their tracked states?

    Survivors are hypotheses refuted down to the observed evidence, each
    carrying its own current state. If for every action the set of
    predicted next-facts across survivors has size 1, the adopted
    vocabulary pins behaviour UNIQUELY despite syntactic multiplicity:
    whichever description is 'true', the model's predictions are fully
    determined. Returns (pinned: bool, worst_fanout: int).
    """
    worst = 0
    for aid in (1, 2, 3, 4, 5):
        outs = set()
        for idx, extra, order, state in survivors:
            prog = tuple(a[i] for a, i in zip(family.axes, idx))
            sp = WorldSpec(
                world_id="cert", seed=0, field_size=field_size,
                mechanics=family.mechanics(prog, extra),
                levels=(replace(observed, targets=tuple(order)),),
            )
            nxt = family.transition(state, Action(aid), sp, extra)
            here = (nxt.x, nxt.y)
            visible = frozenset(t for t in nxt.remaining if t != here)
            outs.add((here, visible,
                      bool(nxt.gates_open), bool(nxt.dead)))
            if len(outs) > 1:
                worst = max(worst, len(outs))
                break
        if len(outs) > 1:
            break
        worst = max(worst, len(outs))
    return worst <= 1, worst


# ----------------------------------------------------------- the agent

def run_agent(truth_spec, true_plan, expand: bool):
    """Explore -> infer -> [trigger -> propose -> dispose -> adopt] ->
    plan -> execute. Returns a result dict."""
    rng = np.random.default_rng(1234)
    world = GridWorld(truth_spec)
    world.reset()
    size = truth_spec.field_size
    observed = read_layout(world.history.initial.grid, size)

    # 1. EXPLORE: random walk. Policy quality is deliberately not the
    # point; the question is what inference does with the evidence.
    for _ in range(EXPLORE_STEPS):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))
    actions = len(world.history.steps)

    seg = scorable_segment(world.history).steps

    # 2. INFER: the unmodified grammar.
    t0 = time.perf_counter()
    base_survivors = refute_family(FAMILY_BY_NAME["base"], seg, observed,
                                   size)
    base_dt = time.perf_counter() - t0

    adopted = None
    contest = {}
    if base_survivors:
        # The grammar still speaks this world. No growth attempted.
        live = base_survivors
        adopted = "base"
    elif not expand:
        # Control: no growth machinery. Fall back to the simplest program
        # and plan inside the fiction anyway.
        idx0 = tuple(0 for _ in BASE_AXES)
        prog0 = tuple(a[0] for a in BASE_AXES)
        inferred = WorldSpec(world_id="inf", seed=0, field_size=size,
                             mechanics=compile_program(prog0),
                             levels=(observed,))
        return finish(world, actions, inferred, observed, size,
                      triggered=True, adopted="NONE (control)",
                      contest={}, base_dt=base_dt,
                      base_empty=True)
    else:
        # 3-5. TRIGGER -> PROPOSE -> DISPOSE -> ADOPT.
        print("   TRIGGER: refutation of all "
              f"{SPACE_SIZE:,} programs came back EMPTY.")
        for family in FAMILIES:
            t1 = time.perf_counter()
            surv = refute_family(family, seg, observed, size)
            dt = time.perf_counter() - t1
            contest[family.name] = surv
            tag = ""
            if not surv:
                tag = "  [ELIMINATED: cannot express the evidence]"
            print(f"   propose {family.name:9} ({family.size:>9,} hyps): "
                  f"{len(surv):>3} survivors ({dt:.1f}s){tag}")
        adopted = adopt_family(contest)
        if adopted is None:
            print("   every candidate family eliminated; giving up honestly")
            return finish(world, actions, None, observed, size,
                          triggered=True, adopted="NONE", contest=contest,
                          base_dt=base_dt, base_empty=True)
        fam = FAMILY_BY_NAME[adopted]
        if fam.kind == "local":
            print(f"   winner '{adopted}' is a LOCAL primitive the engine "
                  "does not contain: adoption DECLINED (exactness must be "
                  "inherited, not re-proved)")
            return finish(world, actions, None, observed, size,
                          triggered=True, adopted="DECLINED",
                          contest=contest, base_dt=base_dt,
                          base_empty=True)
        live = contest[adopted]

    # 6. RESUME: simplest survivor THAT ADMITS A PLAN (X31's lesson:
    # a behaviourally-correct survivor with an infeasible collection
    # order yields no plan -- unusable answers are not answers).
    fam = FAMILY_BY_NAME[adopted]

    def ext_complexity(entry):
        idx, extra = entry[0], entry[1]
        return sum(idx) + fam.extra_values.index(extra)

    inferred = None
    best_prog = best_extra = best_order = None
    for entry in sorted(live, key=ext_complexity)[:64]:
        idx, extra, order = entry[0], entry[1], entry[2]
        prog = tuple(a[i] for a, i in zip(fam.axes, idx))
        cand = WorldSpec(
            world_id="inf", seed=0, field_size=size,
            mechanics=fam.mechanics(prog, extra),
            levels=(replace(observed, targets=tuple(order)),),
        )
        cur = initial_state(0, cand)
        for step in scorable_segment(world.history).steps:
            cur = transition_state(cur, step.action, cand)
        if bfs_plan(cand, cur) is not None:
            inferred, best_prog, best_extra, best_order = \
                cand, prog, extra, order
            break
    if inferred is None:
        print("   no survivor admits a plan; giving up honestly")
        return finish(world, actions, None, observed, size,
                      triggered=True, adopted=adopted, contest=contest,
                      base_dt=base_dt, base_empty=True,
                      certified=False, fanout=None)

    # CERTIFICATE: is the adopted version space behaviourally pinned?
    certified, fanout = behavioural_collapse(live, fam, observed, size)
    return finish(world, actions, inferred, observed, size,
                  triggered=(adopted != "base"), adopted=adopted,
                  contest=contest, base_dt=base_dt,
                  base_empty=not base_survivors,
                  certified=certified, fanout=fanout)


def finish(world, actions, inferred, observed, size, *, triggered,
           adopted, contest, base_dt, base_empty, certified=None,
           fanout=None):
    """Plan inside the inferred model and execute divergence-checked."""
    reinf_note = ""
    if inferred is None:
        return {"won": world.done, "actions": actions, "diverged": None,
                "triggered": triggered, "adopted": adopted,
                "contest": contest, "base_dt": base_dt,
                "base_empty": base_empty, "certified": certified,
                "fanout": fanout}
    cur = initial_state(0, inferred)
    for step in scorable_segment(world.history).steps:
        cur = transition_state(cur, step.action, inferred)
    plan = bfs_plan(inferred, cur)
    if plan is None:
        return {"won": False, "actions": actions, "diverged": None,
                "triggered": triggered, "adopted": adopted,
                "contest": contest, "base_dt": base_dt,
                "base_empty": base_empty, "no_plan": True,
                "certified": certified, "fanout": fanout}
    state = cur
    diverged = False
    for aid in plan:
        if world.done:
            break
        predicted = transition_state(state, Action(aid), inferred)
        world.step(Action(aid))
        actions += 1
        wa, wt, wg = frame_facts(world.history.last.grid, size)
        here = (predicted.x, predicted.y)
        visible = frozenset(t for t in predicted.remaining if t != here)
        if here != wa or visible != wt or (
                wg is not None and bool(predicted.gates_open) != wg):
            diverged = True
            break
        state = predicted
    return {"won": world.done, "actions": actions, "diverged": diverged,
            "triggered": triggered, "adopted": adopted, "contest": contest,
            "base_dt": base_dt, "base_empty": base_empty}


# --------------------------------------------------------------- main

def main() -> int:
    t0 = time.perf_counter()

def run_world(seed: int, size: int, label: str):
    """Full protocol on one hidden world. Returns a result dict."""
    truth_spec = true_plan = None
    for s in range(seed, seed + 60):
        spec, plan = make_ice_world(s, size)
        if spec is not None:
            truth_spec, true_plan = spec, plan
            break
    assert truth_spec is not None, f"no solvable ice world at {label}"
    print(f"--- {label}: {truth_spec.world_id} ({size}x{size}), "
          f"optimal clearing {len(true_plan)} moves")

    ctrl = run_agent(truth_spec, true_plan, expand=False)
    print(f"   control (no growth): base refutation "
          f"{'EMPTY' if ctrl['base_empty'] else 'survivors'}; agent "
          f"{'WON' if ctrl['won'] else 'LOST'} ({ctrl['actions']} actions)")

    full = run_agent(truth_spec, true_plan, expand=True)
    print(f"   trigger fired: base refutation EMPTY -> propose/dispose:")
    for name, surv in full["contest"].items():
        f = FAMILY_BY_NAME[name]
        tag = "  [ELIMINATED]" if not surv else ""
        print(f"     {name:9} {len(surv):>5}/{f.size:<12,}{tag}")
    cert = ("PINNED" if full.get("certified")
            else "NOT pinned" if full.get("certified") is not None
            else "n/a")
    print(f"   ADOPTED: {full['adopted']}  | certificate: {cert}")
    print(f"   expanded agent: {'WON' if full['won'] else 'LOST'} "
          f"({full['actions']} actions, diverged={full['diverged']})\n")
    return ctrl, full


def main() -> int:
    t0 = time.perf_counter()

    # WORLD A: small board -- step-ext and slide are provably equivalent.
    ctrl_a, full_a = run_world(0, SIZE, "world-A (tie case)")
    # WORLD B: large board -- slides exceed every fixed step offered.
    ctrl_b, full_b = run_world(100, 20, "world-B (separation case)")

    gate = (
        ctrl_a["base_empty"] and not ctrl_a["won"]
        and full_a["base_empty"] and full_a["adopted"] in ("slide",
                                                           "step-ext")
        and full_a.get("certified") and full_a["won"]
        and ctrl_b["base_empty"] and not ctrl_b["won"]
        and full_b["base_empty"] and full_b["adopted"] == "slide"
        and not full_b["contest"]["step-ext"]
        and full_b.get("certified") and full_b["won"]
    )
    print(f"(total {time.perf_counter() - t0:.1f}s)")
    print("\nverdict:")
    if gate:
        print("   THE NOVELTY TRIGGER WORKS END TO END, on two regimes:")
        print("   - On world B the decoy family was ELIMINATED by evidence")
        print("     (slides longer than any fixed step) and the slide axis")
        print("     was adopted alone: discrimination where it is possible.")
        print("   - On world A the decoy is genuinely equivalent to the")
        print("     truth on this board; instead of a coin-flip tie-break,")
        print("     the system CERTIFIED behavioural collapse: all survivors")
        print("     agree on every prediction, so the adopted language pins")
        print("     behaviour regardless of which description is true.")
        print("   In both cases execution paused on novelty, growth was")
        print("   proposed by the meta-grammar and disposed by refutation,")
        print("   adoption respected Occam, planning ran inside the grown")
        print("   grammar, and the agent WON -- while controls without")
        print("   growth machinery planned inside fictions and LOST.")
        print("   X33's discovery + X34's mechanism + X29's product, one")
        print("   loop.")
    else:
        print("   Gate failed -- inspect which part broke above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
