"""X46: one substrate -- motion and hazards as programs in the same fabric.

X44 made motion generative with atoms ZERO/ONE/ADD1/REC/FREE. X45 made
hazards generative with atoms STAY/BACK/START/DIE/ON/NEAR. Both were real
progress, and together they exposed the remaining anchor: the two atom sets
do not overlap, so eight rule-menus had become eight ATOM-menus. Slicing a
menu finer is not lifting it.

Here there are no axes. A transition is ONE program over a shared state,
built from primitives that know nothing about movement or danger:

    STEP                 advance one cell along the action's direction
    DIE / HOME / UNDO    set the agent dead, at the entrance, at where the
                         move began
    NOP                  do nothing
    SEQ(a, b)            do a, then b
    IF(p, a, b)          branch
    LOOP(a)              repeat a until it stops changing anything

    GET(prop, offset)    read the board at a relative offset
    OR(p, q)             either

`NEAR` is deliberately NOT a primitive. Making it one would smuggle the old
hazard vocabulary back in; it has to be built as a disjunction over shifts,
which is the test of whether both domains really read the world the same
way. What used to be two vocabularies is now:

    step 1          IF(GET(WALK, AHEAD), STEP, NOP)
    slide           LOOP(IF(GET(WALK, AHEAD), STEP, NOP))
    kill            IF(GET(HAZ, HERE), DIE, NOP)
    proximity mine  IF(OR(GET(HAZ,N), OR(GET(HAZ,S), ...)), DIE, NOP)
    slide + kill    SEQ(LOOP(...), IF(GET(HAZ, HERE), DIE, NOP))

The last line is the point. A world with ice AND a hazard rule is not two
axis settings any more; it is one program, and the composite is expressible
in the same breath as its parts.

WHAT WOULD FALSIFY THE UNIFICATION, stated before the numbers exist: if
motion rules are recovered but hazard rules are not, or the reverse, the
substrate is really one domain's vocabulary with the other bolted on. Both
must fall out of the same pool, and so must their composition.

MEASURED (291 predicates, 106k behaviourally-distinct programs to size 9):

  true rule           survivors  synthesised                            exact
  step 1                      1  (IF WALK@AHEAD STEP NOP)                 yes
  slide                       1  (LOOP (IF WALK@AHEAD STEP NOP))          yes
  step + kill                 1  (SEQ (IF ...) (IF HAZ@HERE DIE NOP))     yes
  step + pushback             1  (IF WALK@AHEAD (IF HAZ@AHEAD NOP STEP))  yes
  slide + kill                1  (LOOP (IF WALK@AHEAD STEP (IF HAZ@...))) yes
  step + proximity            0  15 nodes -- beyond the size budget        no

  5/6, each a unique survivor, all from one pool.

TWO RECOVERED PROGRAMS ARE BETTER THAN THE ONES WRITTEN. `step + pushback`
came back as "do not step onto a hazard" rather than "step, then undo it",
and `slide + kill` folded the hazard test inside the loop. Both are
behaviourally identical to the truth and shorter, which is what selecting on
behaviour rather than syntax is for.

TWO FLAWS IN THE FIRST VERSION OF THIS EXPERIMENT, both of which made the
substrate look better than it was:

  `STEP` refused to enter a wall, so motion was half-hardcoded and `step 1`
  was recovered as bare `STEP` -- the primitive was doing the collision test
  the program was supposed to express. STEP now moves unconditionally and
  the guard has to be synthesised.

  hazard rules were tested WITHOUT motion, so the agent never moved, never
  landed on anything, and every hazard rule collapsed to NOP. `pushback`
  scored correct for exactly that reason. Hazard truths are now composed
  with motion.

THE COST OF UNIFYING, and the real headline. X45 expressed a proximity mine
in 3 nodes using hazard-specific atoms. Here the same rule, composed with
motion, is 15 -- and enumeration to size 9 already yields ~10^5 programs, so
it is out of reach by search alone. Unification buys expressiveness and
charges search depth. That is exactly the regime where enumeration stops
being the answer and a learned proposer would have to earn its place, which
is the argument for the core rather than against the substrate.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

Expr = tuple | str
Pred = tuple

DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
OFFSETS = {
    "HERE": (0, 0),
    "N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0),
}
PROPS = ("WALK", "HAZ")


@dataclass(frozen=True, slots=True)
class Board:
    size: int
    walls: frozenset
    hazards: frozenset
    start: tuple[int, int]


@dataclass(frozen=True, slots=True)
class State:
    pos: tuple[int, int]
    began: tuple[int, int]
    dir: tuple[int, int]
    alive: bool = True


def read(board: Board, pos, offset: str, prop: str, direction) -> bool:
    """The single way any program looks at the world.

    AHEAD is the action's own direction, which is what lets motion and
    danger share one query form: `is the next cell walkable` and `is the
    cell to my north dangerous` are the same operation with different
    arguments.
    """
    dx, dy = direction if offset == "AHEAD" else OFFSETS[offset]
    cell = (pos[0] + dx, pos[1] + dy)
    if prop == "WALK":
        return (0 <= cell[0] < board.size and 0 <= cell[1] < board.size
                and cell not in board.walls)
    return cell in board.hazards


def test(pred: Pred, board: Board, st: State) -> bool:
    if pred[0] == "GET":
        return read(board, st.pos, pred[1], pred[2], st.dir)
    return test(pred[1], board, st) or test(pred[2], board, st)


def run(expr: Expr, board: Board, st: State, fuel: int = 64) -> tuple[State, int]:
    if fuel <= 0 or not st.alive:
        return st, fuel
    if expr == "NOP":
        return st, fuel - 1
    if expr == "STEP":
        # Moves UNCONDITIONALLY, except off the board. It does not know
        # about walls.
        #
        # The first version refused to enter a wall, which quietly made
        # motion half-hardcoded: `step 1` was then recovered as bare `STEP`,
        # because the primitive was already doing the collision test the
        # program was supposed to express. A primitive that knows the domain
        # is a menu item wearing a lowercase name.
        dx, dy = st.dir
        cell = (st.pos[0] + dx, st.pos[1] + dy)
        if 0 <= cell[0] < board.size and 0 <= cell[1] < board.size:
            return replace(st, pos=cell), fuel - 1
        return st, fuel - 1
    if expr == "DIE":
        return replace(st, alive=False), fuel - 1
    if expr == "HOME":
        return replace(st, pos=board.start), fuel - 1
    if expr == "UNDO":
        return replace(st, pos=st.began), fuel - 1
    head = expr[0]
    if head == "SEQ":
        st, fuel = run(expr[1], board, st, fuel)
        return run(expr[2], board, st, fuel)
    if head == "IF":
        return run(expr[2] if test(expr[1], board, st) else expr[3],
                   board, st, fuel - 1)
    if head == "LOOP":
        # Bounded, and stops when a pass changes nothing. Unbounded
        # iteration would make enumeration undecidable; the board is finite
        # so `size` passes is all any spatial loop can need.
        for _ in range(board.size + 1):
            nxt, fuel = run(expr[1], board, st, fuel)
            if nxt == st or fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


def transition(prog: Expr, board: Board, pos, aid: int):
    st = State(pos=pos, began=pos, dir=DIRS[aid])
    out, _ = run(prog, board, st)
    return None if not out.alive else out.pos


# --------------------------------------------------------- enumeration

ACTIONS = ("NOP", "STEP", "DIE", "HOME", "UNDO")


def enumerate_preds(max_size: int) -> list[Pred]:
    """Predicates deduped by TRUTH over probe situations, not by syntax.

    OR is commutative and associative, so a syntactic enumeration produces
    the same test many times over and multiplies the program search by a
    constant that buys nothing. Deduping here is what makes the program
    layer affordable.
    """
    situations = [
        (board, (x, y), aid)
        for board in PROBE_BOARDS
        for y in range(board.size) for x in range(board.size)
        for aid in DIRS
    ]

    def truth_vector(pred):
        return tuple(
            test(pred, b, State(pos=pos, began=pos, dir=DIRS[a]))
            for b, pos, a in situations
        )

    seen: dict[tuple, Pred] = {}
    by_size: dict[int, list[Pred]] = {1: []}
    for prop in PROPS:
        for off in ("HERE", "AHEAD", "N", "S", "W", "E"):
            pred = ("GET", off, prop)
            v = truth_vector(pred)
            if v not in seen:
                seen[v] = pred
                by_size[1].append(pred)
    for n in range(2, max_size + 1):
        out = []
        for i in range(1, n):
            for a in by_size.get(i, []):
                for b in by_size.get(n - i, []):
                    if a >= b:
                        continue
                    pred = ("OR", a, b)
                    v = truth_vector(pred)
                    if v in seen:
                        continue
                    seen[v] = pred
                    out.append(pred)
        by_size[n] = out
    return [p for n in sorted(by_size) for p in by_size[n]]


PROBE_BOARDS: list[Board] = []


def behaviour_of(prog: Expr) -> tuple:
    """What a program does across a fixed battery of situations."""
    out = []
    for board in PROBE_BOARDS:
        for y in range(board.size):
            for x in range(board.size):
                for aid in DIRS:
                    out.append(transition(prog, board, (x, y), aid))
    return tuple(out)


def enumerate_programs(max_size: int, preds: list[Pred]) -> list[Expr]:
    seen: dict[tuple, Expr] = {}
    by_size: dict[int, list[Expr]] = {}

    def add(expr: Expr, n: int) -> None:
        sig = behaviour_of(expr)
        if sig in seen:
            return
        seen[sig] = expr
        by_size.setdefault(n, []).append(expr)

    for a in ACTIONS:
        add(a, 1)
    for n in range(2, max_size + 1):
        for e in by_size.get(n - 1, []):
            add(("LOOP", e), n)
        for i in range(1, n):
            for a in by_size.get(i, []):
                for b in by_size.get(n - 1 - i, []):
                    add(("SEQ", a, b), n)
        for p in preds:
            ps = size_of(p)
            for i in range(1, n):
                rest = n - 1 - ps - i
                if rest < 1:
                    continue
                for a in by_size.get(i, []):
                    for b in by_size.get(rest, []):
                        if a is not b:
                            add(("IF", p, a, b), n)
    return [e for n in sorted(by_size) for e in by_size[n]]


def render(expr: Expr) -> str:
    if isinstance(expr, str):
        return expr
    if expr[0] == "GET":
        return f"{expr[2]}@{expr[1]}"
    if expr[0] == "OR":
        return f"({render(expr[1])}|{render(expr[2])})"
    if expr[0] == "IF":
        return f"(IF {render(expr[1])} {render(expr[2])} {render(expr[3])})"
    if expr[0] == "LOOP":
        return f"(LOOP {render(expr[1])})"
    return f"(SEQ {render(expr[1])} {render(expr[2])})"


def size_of(expr: Expr) -> int:
    if isinstance(expr, str):
        return 1
    if expr[0] == "GET":
        return 1
    if expr[0] == "OR":
        return 1 + size_of(expr[1]) + size_of(expr[2])
    if expr[0] == "IF":
        return 1 + size_of(expr[1]) + size_of(expr[2]) + size_of(expr[3])
    if expr[0] == "LOOP":
        return 1 + size_of(expr[1])
    return 1 + size_of(expr[1]) + size_of(expr[2])


# ------------------------------------------------------------- worlds

def make_board(seed: int, size: int = 7) -> Board:
    rng = np.random.default_rng(seed)
    cells = [(x, y) for y in range(size) for x in range(size)]
    rng.shuffle(cells)
    walls = frozenset(cells[:size])
    hazards = frozenset(cells[size:size + 3])
    start = next(c for c in cells[size + 3:])
    return Board(size, walls, hazards, start)


NEAR = ("OR", ("GET", "N", "HAZ"),
        ("OR", ("GET", "S", "HAZ"),
         ("OR", ("GET", "W", "HAZ"), ("GET", "E", "HAZ"))))

STEP1 = ("IF", ("GET", "AHEAD", "WALK"), "STEP", "NOP")
SLIDE = ("LOOP", STEP1)

TRUTHS = {
    # Motion alone.
    "step 1": STEP1,
    "slide": SLIDE,
    # Hazard rules are COMPOSED with motion, because a hazard rule with no
    # motion is unreachable: the agent never moves, never lands on anything,
    # and every hazard rule collapses to NOP. The first version tested them
    # bare and scored `pushback` correct for exactly that reason.
    "step + kill": ("SEQ", STEP1, ("IF", ("GET", "HERE", "HAZ"), "DIE", "NOP")),
    "step + pushback": ("SEQ", STEP1, ("IF", ("GET", "HERE", "HAZ"), "UNDO", "NOP")),
    "step + proximity": ("SEQ", STEP1, ("IF", NEAR, "DIE", "NOP")),
    "slide + kill": ("SEQ", SLIDE, ("IF", ("GET", "HERE", "HAZ"), "DIE", "NOP")),
}


def discriminating_probe(survivors, boards):
    for board in boards:
        for y in range(board.size):
            for x in range(board.size):
                for aid in DIRS:
                    outs = {transition(r, board, (x, y), aid) for r in survivors}
                    if len(outs) > 1:
                        return board, (x, y), aid
    return None


def main() -> int:
    global PROBE_BOARDS
    PROBE_BOARDS = [make_board(11, 5)]
    boards = [make_board(s) for s in range(1, 7)]

    print("X46: one substrate for motion AND hazards\n")
    print("primitives: NOP STEP DIE HOME UNDO  SEQ IF LOOP  GET(prop,offset) OR")
    print("no primitive is called FREE, NEAR, slide or kill.\n")

    t0 = time.perf_counter()
    preds = enumerate_preds(4)
    progs = enumerate_programs(9, preds)
    print(f"{len(preds)} predicates, {len(progs):,} behaviourally-distinct "
          f"programs ({time.perf_counter()-t0:.1f}s)\n")

    print("size of each truth in this substrate, against the X44/X45 "
          "axis-specific atoms:")
    for name, truth in TRUTHS.items():
        print(f"    {name:20} {size_of(truth):>3} nodes")
    print()
    print(f'{"true rule":20} {"kind":>8} {"survivors":>10} {"synthesised":36} '
          f'{"exact":>6} {"probes":>7}')
    got = Counter()
    for name, truth in TRUTHS.items():
        kind = "both" if "+" in name else "motion"
        survivors = [p for p in progs
                     if all(transition(p, b, (x, y), a) == transition(truth, b, (x, y), a)
                            for b in boards[:2]
                            for y in range(b.size) for x in range(b.size)
                            for a in DIRS)]
        probes = 0
        while len(survivors) > 1 and probes < 8:
            found = discriminating_probe(survivors, boards)
            if found is None:
                break
            b, pos, aid = found
            want = transition(truth, b, pos, aid)
            survivors = [r for r in survivors if transition(r, b, pos, aid) == want]
            probes += 1

        if not survivors:
            print(f"{name:20} {kind:>8} {0:>10} "
                  f"{'-- not expressible at this size --':36} {'no':>6} {probes:>7}")
            got["missing"] += 1
            continue
        chosen = min(survivors, key=lambda e: (size_of(e), render(e)))
        exact = behaviour_of(chosen) == behaviour_of(truth)
        got["exact" if exact else "wrong"] += 1
        print(f"{name:20} {kind:>8} {len(survivors):>10} {render(chosen)[:36]:36} "
              f"{'yes' if exact else 'NO':>6} {probes:>7}")

    print(f"\nrecovered: {dict(got)}")
    print("\nEvery rule recovered above came out of ONE pool of primitives.")
    print("Not one names a domain: STEP moves without knowing about walls, GET")
    print("knows nothing about danger, and `near` is a disjunction over shifts")
    print("rather than a handed-over primitive.")
    print("\nTHE COST OF UNIFYING, which is the honest headline here:")
    print("  X45's hazard atoms expressed a proximity mine in 3 nodes.")
    print("  In this substrate the same rule, composed with motion, is 15.")
    print("  Enumeration to size 9 already yields ~10^5 programs, so the")
    print("  composite proximity rule is out of reach by search alone.")
    print("  Unification buys expressiveness and charges search depth --")
    print("  which is precisely where a learned proposer would have to earn")
    print("  its place, and where pure enumeration stops being the answer.")
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
