"""X44: invent the movement rule, instead of selecting a dormant primitive.

X43 closed the n=2 caveat -- 20/20 worlds, adoption tracking the evidence.
It did not close the substrate caveat. In the grid loop, growth picks among
hand-written families (base, step-ext, slide, momentum), and `slide` is a
primitive the ENGINE already implements. The system discovers which dormant
capability reality demands; it never invents one. X38 and X40 broke that
bound on strings. This carries it to the grid.

A movement rule is a function from (position, direction, board) to a travel
distance, and it is naturally recursive: how far you go is one step plus how
far you go FROM THE NEXT CELL. So the atoms are:

    ZERO            travel no further
    ONE             travel exactly one more
    FREE            is the next cell along the ray passable?
    ADD1(e)         one more than e
    REC             travel distance from the next cell along the ray
    COND(FREE, t, e)

Nothing here slides, steps-k, or stops at a wall. Those are SHAPES:

    step 1      COND(FREE, ONE, ZERO)
    step 2      COND(FREE, ADD1(COND(FREE, ONE, ZERO)), ZERO)
    slide       COND(FREE, ADD1(REC), ZERO)

`slide` is three atoms deep and nobody wrote it down. Refutation finds it
from observed displacements.

WHY THE RECURSION TERMINATES, which is the whole trick and the same one
X40 used: REC advances strictly along the ray, and a ray leaves the board
in at most `size` steps. Evaluating positions from the far end backwards
makes REC a LOOKUP of an answer already computed, so no descent, no
termination check, and no unbounded search.

MEASURED (6 ice worlds, sizes 10-20, 120 exploration steps each):

    125 behaviourally-distinct movement rules up to size 7
    every world: 1 survivor, and the same one --

        (COND FREE (ADD1 REC) ZERO)

    which IS slide, recovered uniquely from displacement evidence alone.

The rule is not selected from a list. `slide` is not a symbol in this
file's vocabulary; it is a shape built from ZERO, ADD1, REC and a test on
whether the next cell is free -- the same way X40 builds filtration from
CONS and COND. Refutation reduces 125 candidates to exactly one on every
board, which means the evidence determines the rule rather than merely
permitting it.

A CHECK THAT WAS WRONG BEFORE IT WAS RIGHT: the first version decided
whether the recovered rule was ice by probing travel from cell (0,0) going
right, and reported "slide 3, other 3" for six worlds that had all
recovered the IDENTICAL expression. It was measuring whether (0,0) happened
to sit against a wall. A test of a rule must not depend on where you stand;
the check now compares behaviour against a reference expression.

WHAT THIS DOES AND DOES NOT CLOSE. It removes the substrate caveat for
MOTION: the grid loop can now invent its movement rule rather than awaken a
dormant engine primitive. It does not make the whole grid grammar
generative -- hazards, switches, gates and collection order are still eight
hand-written axes, and a world whose hazard behaviour is unlike any of them
remains unlearnable. Motion was chosen first because it is where X35's ice
worlds actually broke.
"""

from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x35_novelty_trigger import EXPLORE_STEPS, make_ice_world
from sentinel.env.types import Action
from sentinel.gen.grid import MOVES, GridWorld, blocked

Expr = tuple | str
ATOMS = ("ZERO", "ONE", "REC")


# ------------------------------------------------------------ evaluation


def travel_table(expr: Expr, level, size: int, gates_open: bool,
                 start: tuple[int, int], step: tuple[int, int]) -> dict:
    """Travel distance from every cell along one ray, far end first.

    Positions are visited in reverse ray order, so when `REC` is reached the
    answer for the next cell is already in the table. That is what makes an
    unrestricted recursive movement rule enumerable rather than divergent.
    """
    dx, dy = step
    ray = []
    x, y = start
    for _ in range(size + 1):
        ray.append((x, y))
        x, y = x + dx, y + dy
        if not (0 <= x < size and 0 <= y < size):
            break

    table: dict[tuple[int, int], int | None] = {}
    for pos in reversed(ray):
        nxt = (pos[0] + dx, pos[1] + dy)
        free = not blocked(nxt, level, size, gates_open)
        table[pos] = evaluate(expr, free, table.get(nxt))
    return table


def evaluate(expr: Expr, free: bool, rec: int | None) -> int | None:
    """One node. `rec` is the already-computed answer for the next cell."""
    if expr == "ZERO":
        return 0
    if expr == "ONE":
        return 1
    if expr == "REC":
        return rec
    head = expr[0]
    if head == "ADD1":
        inner = evaluate(expr[1], free, rec)
        return None if inner is None else inner + 1
    if head == "COND":
        return evaluate(expr[1] if free else expr[2], free, rec)
    return None


SLIDE_REF: Expr = ("COND", ("ADD1", "REC"), "ZERO")
"""What ice looks like in this vocabulary: one more than the rest of the
ray while the way is clear, nothing when blocked."""

PROBE_RECS = (None, 0, 1, 2, 5)


def behaviour_of(expr: Expr) -> tuple:
    """What a rule returns across free/blocked and possible rest-of-ray values."""
    return tuple(
        evaluate(expr, free, rec)
        for free in (True, False)
        for rec in PROBE_RECS
    )


def render(expr: Expr) -> str:
    if isinstance(expr, str):
        return expr
    if expr[0] == "COND":
        return f"(COND FREE {render(expr[1])} {render(expr[2])})"
    return f"({expr[0]} {render(expr[1])})"


def size_of(expr: Expr) -> int:
    if isinstance(expr, str):
        return 1
    return 1 + sum(size_of(p) for p in expr[1:])


# --------------------------------------------------------- enumeration


def enumerate_rules(max_size: int) -> list[Expr]:
    """Every movement rule up to `max_size`, deduped by (free, rec) behaviour.

    A rule's behaviour is what it returns for each combination of "is the
    next cell free" and "what does the rest of the ray give" -- a tiny
    signature, and enough to collapse syntactic duplicates.
    """
    seen: dict[tuple, Expr] = {}
    by_size: dict[int, list[Expr]] = {}
    def add(expr: Expr, n: int) -> None:
        sig = behaviour_of(expr)
        if sig in seen:
            return
        seen[sig] = expr
        by_size.setdefault(n, []).append(expr)

    for atom in ATOMS:
        add(atom, 1)
    for n in range(2, max_size + 1):
        for e in by_size.get(n - 1, []):
            add(("ADD1", e), n)
        for i in range(1, n):
            for t in by_size.get(i, []):
                for e in by_size.get(n - 1 - i, []):
                    if t is not e:
                        add(("COND", t, e), n)
    return [e for n in sorted(by_size) for e in by_size[n]]


# ---------------------------------------------------------- the evidence


def observed_moves(world, spec, size: int):
    """(level, gates, start, direction, actual distance) per real transition.

    Only moves are usable: a wait teaches nothing about travel. Gate state
    is read from the frame rather than assumed.
    """
    from sentinel.explore.version_space import observed_facts

    out = []
    prev = world.history.initial
    level = spec.levels[0]
    for step in world.history.steps:
        aid = step.action.action_id
        if aid not in MOVES:
            prev = step.settled
            continue
        before_agent, _, before_gates = observed_facts(prev.grid, size)
        after_agent, _, _ = observed_facts(step.settled.grid, size)
        if before_agent is None or after_agent is None:
            prev = step.settled
            continue
        dx, dy = MOVES[aid]
        dist = abs(after_agent[0] - before_agent[0]) + abs(after_agent[1] - before_agent[1])
        # Off-axis motion means something other than travel happened
        # (a respawn, a wrap); those transitions are not evidence about
        # how far a move goes.
        on_axis = (after_agent[0] - before_agent[0]) * dy == 0 and \
                  (after_agent[1] - before_agent[1]) * dx == 0
        if on_axis:
            out.append((level, bool(before_gates), before_agent, (dx, dy), dist))
        prev = step.settled
    return out


def refute(rules: list[Expr], evidence, size: int) -> list[Expr]:
    survivors = []
    for rule in rules:
        ok = True
        for level, gates, start, step, actual in evidence:
            table = travel_table(rule, level, size, gates, start, step)
            if table.get(start) != actual:
                ok = False
                break
        if ok:
            survivors.append(rule)
    return survivors


def main() -> int:
    print("X44: synthesising the movement rule from atoms\n")
    print(f"atoms: {'  '.join(ATOMS)}   ADD1(_)   COND(FREE, _, _)")
    print("nothing here slides, steps-k, or stops at a wall.\n")

    t0 = time.perf_counter()
    rules = enumerate_rules(max_size=7)
    print(f"{len(rules)} behaviourally-distinct movement rules "
          f"({time.perf_counter()-t0:.2f}s)\n")

    print(f'{"world":>10} {"moves":>6} {"survivors":>10} {"synthesised rule":38} {"slide?":>7}')
    found = Counter()
    for seed, size in ((0, 10), (7, 12), (23, 14), (100, 20), (0, 20), (7, 16)):
        made = make_ice_world(seed, size)
        if made is None or made[0] is None:
            continue
        spec, _ = made
        world = GridWorld(spec)
        world.reset()
        rng = np.random.default_rng(1234)
        for _ in range(EXPLORE_STEPS):
            if world.done:
                break
            world.step(Action(int(rng.integers(1, 6))))

        evidence = observed_moves(world, spec, size)
        survivors = refute(rules, evidence, size)
        if not survivors:
            print(f"{f'ice{seed}/{size}':>10} {len(evidence):>6} {0:>10} "
                  f"{'-- no rule explains the evidence --':38} {'-':>7}")
            found["none"] += 1
            continue
        chosen = min(survivors, key=lambda e: (size_of(e), render(e)))
        # Is the chosen rule ice? Compare BEHAVIOUR against the reference
        # slide expression.
        #
        # The first version probed one board cell -- travel from (0,0) going
        # right -- and reported "slide 3, other 3" for six worlds that had
        # all recovered the identical rule. It was measuring whether (0,0)
        # happened to be against a wall, not what the rule does. A check on
        # a rule must not depend on where you stand.
        is_slide = behaviour_of(chosen) == behaviour_of(SLIDE_REF)
        found["slide (invented)" if is_slide else "other rule"] += 1
        print(f"{f'ice{seed}/{size}':>10} {len(evidence):>6} {len(survivors):>10} "
              f"{render(chosen)[:38]:38} {'yes' if is_slide else 'no':>7}")

    print(f"\nrules recovered: {dict(found)}")
    print("\nThe ice worlds' true mechanic is `slide`, a primitive the ENGINE")
    print("implements and the 8-axis DSL cannot express. Here it is not selected")
    print("from a list -- it is built from ZERO, ONE, FREE, ADD1 and REC, the")
    print("same way X40 builds filtration from CONS and COND.")
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
