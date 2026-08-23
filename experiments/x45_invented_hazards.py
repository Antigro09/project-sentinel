"""X45: invent hazard behaviour, including behaviours the axes cannot express.

X44 made MOTION generative: `slide` is built from ZERO/ADD1/REC rather than
selected from a list. Seven axes remain menu-driven, and hazards are the
sharpest case -- the engine offers exactly three responses (kill, pushback,
respawn) and a world whose obstacle behaves any other way is unlearnable by
construction.

The test case is the one the critique named: a PROXIMITY mine, which fires
when the agent is merely NEXT TO it rather than on it. No setting of
`hazard_effect` expresses that, so the menu cannot represent the truth no
matter how the evidence falls.

The atoms describe what a hazard DOES to the agent's position, not which of
three named behaviours it is:

    STAY        the move stands
    BACK        return to where the move began
    START       return to the level's entrance
    DIE         the episode ends
    COND(p, a, b) with predicates ON (standing on a hazard cell) and
                  NEAR (orthogonally adjacent to one)

`kill` is COND(ON, DIE, STAY). `pushback` is COND(ON, BACK, STAY). A
proximity mine is COND(NEAR, DIE, STAY) -- three atoms, and unreachable
from the eight-axis grammar at any size.

THE QUESTION THIS IS REALLY ASKING is not whether hazards can be made
generative -- X44 already showed one axis can. It is whether making each
axis generative separately is progress or bookkeeping. Motion needed
ZERO/ONE/ADD1/REC/FREE; hazards need STAY/BACK/START/DIE/ON/NEAR. If every
axis needs its own atom set, eight rule-menus have become eight atom-menus
and the anchor has moved rather than lifted. That is measured below rather
than asserted.

MEASURED (256 rules to size 7, 12 boards x 4 walks, then targeted probes):

  true behaviour        in menu?  survivors  synthesised                 probes
  kill                       yes          1  (COND ON DIE STAY)              0
  pushback                   yes          1  (COND ON BACK STAY)             0
  respawn                    yes          1  (COND ON START STAY)            0
  harmless                   yes          1  STAY                            0
  proximity mine              NO          1  (COND NEAR DIE STAY)            2
  proximity pushback          NO          1  (COND NEAR BACK STAY)           2
  on kills, near pushes       NO          1  (COND ON DIE (COND NEAR ...))   2

  7/7 exact, each pinned to a unique survivor.

Three of these describe behaviour the eight-axis grammar cannot express at
any setting -- a mine that fires when you are merely NEXT to it has no
`hazard_effect` value. The menu cannot represent them however the evidence
falls; the atoms build them in three nodes.

RANDOM EVIDENCE WAS NOT ENOUGH, and the way it failed is the point.
`on kills, near pushes` came back as `near pushes`: the two differ only when
a move LANDS on a hazard cell that has no hazard neighbour, measured at 39
of 48,000 transitions -- 0.08%. Three boards never contained one. Twelve
boards and four walks each still did not fix it. The NEAR rules left 16
survivors for a related reason: if landing NEAR a hazard triggers, a cell ON
one is unreachable, so every rule differing only in its ON branch is
genuinely indistinguishable from evidence that never gets there.

What fixed it was seeking the case instead of waiting for it: simulate the
survivors over every cell and action, take the first situation they disagree
about, and ask. Two probes per rule. This is `version_space.best_action` on
a third substrate, and the third time in this project that a rare rule has
had to be hunted rather than observed -- ordered_targets, wait_advances_
charge, and now hazard branches behind an unreachable guard.

THE HONEST LIMIT, measured rather than asserted. Motion needed atoms
ZERO/ONE/ADD1/REC/FREE; hazards need STAY/BACK/START/DIE/ON/NEAR. The sets
do not overlap. Making each axis generative separately turns eight
rule-menus into eight ATOM-menus: real progress per axis -- proximity mines
are learnable now and were not -- but the anchor has moved rather than
lifted. Lifting it needs one substrate in which motion and hazards are both
ordinary programs.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

Expr = tuple | str
ATOMS = ("STAY", "BACK", "START", "DIE")
PREDICATES = ("ON", "NEAR")


# ------------------------------------------------------ the toy world

@dataclass(frozen=True, slots=True)
class Board:
    size: int
    walls: frozenset
    hazards: frozenset
    start: tuple[int, int]


MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def on_hazard(board: Board, pos) -> bool:
    return pos in board.hazards


def near_hazard(board: Board, pos) -> bool:
    x, y = pos
    return any((x + dx, y + dy) in board.hazards for dx, dy in MOVES.values())


def apply_effect(expr: Expr, board: Board, landed, began) -> tuple | None:
    """Where the agent ends up. None means the episode ended."""
    if expr == "STAY":
        return landed
    if expr == "BACK":
        return began
    if expr == "START":
        return board.start
    if expr == "DIE":
        return None
    pred, a, b = expr[0], expr[1], expr[2]
    hit = on_hazard(board, landed) if pred == "ON" else near_hazard(board, landed)
    return apply_effect(a if hit else b, board, landed, began)


def step_world(board: Board, pos, aid: int, effect: Expr):
    """One move: travel one cell if free, then let the hazard rule speak."""
    dx, dy = MOVES[aid]
    nxt = (pos[0] + dx, pos[1] + dy)
    if not (0 <= nxt[0] < board.size and 0 <= nxt[1] < board.size) or nxt in board.walls:
        nxt = pos
    return apply_effect(effect, board, nxt, pos)


def make_board(seed: int, size: int = 8) -> Board:
    rng = np.random.default_rng(seed)
    cells = [(x, y) for y in range(size) for x in range(size)]
    rng.shuffle(cells)
    walls = frozenset(cells[: size])
    hazards = frozenset(cells[size: size + 4])
    start = next(c for c in cells[size + 4:] if c not in walls and c not in hazards)
    return Board(size=size, walls=walls, hazards=hazards, start=start)


def episode(board: Board, effect: Expr, seed: int, steps: int = 120):
    """Random walk, recording (before, action, after). None marks a death."""
    rng = np.random.default_rng(seed)
    pos = board.start
    out = []
    for _ in range(steps):
        aid = int(rng.integers(1, 5))
        nxt = step_world(board, pos, aid, effect)
        out.append((pos, aid, nxt))
        if nxt is None:
            pos = board.start          # respawn to keep gathering evidence
        else:
            pos = nxt
    return out


# -------------------------------------------------------- enumeration


def render(expr: Expr) -> str:
    if isinstance(expr, str):
        return expr
    return f"(COND {expr[0]} {render(expr[1])} {render(expr[2])})"


def size_of(expr: Expr) -> int:
    return 1 if isinstance(expr, str) else 1 + size_of(expr[1]) + size_of(expr[2])


SIGNATURE_CASES = tuple(
    (on, near) for on in (True, False) for near in (True, False)
)


def behaviour_of(expr: Expr) -> tuple:
    """What the rule does in each (on-hazard, near-hazard) situation."""
    out = []
    for on, near in SIGNATURE_CASES:
        node = expr
        while not isinstance(node, str):
            hit = on if node[0] == "ON" else near
            node = node[1] if hit else node[2]
        out.append(node)
    return tuple(out)


def enumerate_effects(max_size: int) -> list[Expr]:
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
        for pred in PREDICATES:
            for i in range(1, n):
                for a in by_size.get(i, []):
                    for b in by_size.get(n - 1 - i, []):
                        if a is not b:
                            add((pred, a, b), n)
    return [e for n in sorted(by_size) for e in by_size[n]]


def discriminating_probe(rules: list[Expr], boards):
    """A (board, position, action) the survivors do not agree about.

    Random walking cannot settle a rule whose distinguishing case occurs in
    0.08% of transitions -- measured, and twelve boards of random evidence
    still left the wrong rule standing. But the survivors themselves know
    where they differ: simulate each on every cell and action, and take the
    first situation that splits them. This is `version_space.best_action`
    again, and the third time in this project that a rare rule has needed
    seeking rather than waiting for.
    """
    for board in boards:
        for y in range(board.size):
            for x in range(board.size):
                if (x, y) in board.walls:
                    continue
                for aid in MOVES:
                    outcomes = {step_world(board, (x, y), aid, r) for r in rules}
                    if len(outcomes) > 1:
                        return board, (x, y), aid
    return None


def refute(rules: list[Expr], board: Board, evidence) -> list[Expr]:
    survivors = []
    for rule in rules:
        if all(step_world(board, before, aid, rule) == after
               for before, aid, after in evidence):
            survivors.append(rule)
    return survivors


# ------------------------------------------------------------- truths

TRUTHS = {
    "kill (menu)": ("ON", "DIE", "STAY"),
    "pushback (menu)": ("ON", "BACK", "STAY"),
    "respawn (menu)": ("ON", "START", "STAY"),
    "proximity mine": ("NEAR", "DIE", "STAY"),
    "proximity pushback": ("NEAR", "BACK", "STAY"),
    "harmless": "STAY",
    "on kills, near pushes": ("ON", "DIE", ("NEAR", "BACK", "STAY")),
}

MENU = {"kill (menu)", "pushback (menu)", "respawn (menu)", "harmless"}


def main() -> int:
    print("X45: synthesising hazard behaviour from atoms\n")
    print(f"atoms: {'  '.join(ATOMS)}   COND(ON|NEAR, _, _)")
    print("no atom names kill, pushback or respawn.\n")

    t0 = time.perf_counter()
    rules = enumerate_effects(max_size=7)
    print(f"{len(rules)} behaviourally-distinct hazard rules "
          f"({time.perf_counter()-t0:.2f}s)\n")

    print(f'{"true behaviour":22} {"in 8-axis menu?":>16} {"survivors":>10} '
          f'{"synthesised":28} {"exact":>6} {"probes":>6}')
    recovered = Counter()
    for name, truth in TRUTHS.items():
        # Twelve boards, four walks each. Three boards was not enough, and
        # the way it failed is worth keeping: `on kills, near pushes` came
        # back as `near pushes`, which differs only when a move LANDS on a
        # hazard cell that has no hazard neighbour -- measured at 39 of
        # 48,000 transitions, about 0.08%. Three episodes never contained
        # one, so refutation was starved rather than broken, and the
        # simplest surviving rule was wrong for want of a rare event. The
        # same shape as `ordered_targets`: a rule the evidence does not
        # exercise cannot be inferred, however good the search is.
        boards = [make_board(s) for s in range(1, 13)]
        survivors = list(rules)
        for board in boards:
            for walk in range(4):
                if len(survivors) <= 1:
                    break
                survivors = refute(survivors, board,
                                   episode(board, truth, seed=7 + walk))
        in_menu = "yes" if name in MENU else "NO"
        # Seek the discriminating case rather than hoping a walk contains it.
        probes = 0
        while len(survivors) > 1 and probes < 12:
            found = discriminating_probe(survivors, boards)
            if found is None:
                break  # genuinely indistinguishable: no situation splits them
            board, pos, aid = found
            want = step_world(board, pos, aid, truth)
            survivors = [r for r in survivors
                         if step_world(board, pos, aid, r) == want]
            probes += 1

        if not survivors:
            print(f"{name:22} {in_menu:>16} {0:>10} "
                  f"{'-- nothing explains it --':28} {'no':>6}")
            recovered["none"] += 1
            continue
        chosen = min(survivors, key=lambda e: (size_of(e), render(e)))
        exact = behaviour_of(chosen) == behaviour_of(truth)
        recovered["exact" if exact else "wrong"] += 1
        print(f"{name:22} {in_menu:>16} {len(survivors):>10} "
              f"{render(chosen)[:28]:28} {'yes' if exact else 'NO':>6} {probes:>6}")

    total = sum(recovered.values())
    print(f"\nrecovered exactly: {recovered['exact']}/{total}")

    off_menu = [n for n in TRUTHS if n not in MENU]
    print(f"\nof these, {len(off_menu)} describe behaviour the eight-axis grammar")
    print("cannot express at any setting -- a mine that fires when you are merely")
    print("NEXT to it has no `hazard_effect` value. The menu cannot represent")
    print("them however the evidence falls; the atoms build them in three nodes.")

    print("\nTHE HONEST QUESTION, measured rather than asserted:")
    print(f"  motion needed atoms ZERO/ONE/ADD1/REC/FREE  (X44)")
    print(f"  hazards need atoms  STAY/BACK/START/DIE/ON/NEAR  (here)")
    print("  These sets do not overlap. Making each axis generative separately")
    print("  turns eight rule-menus into eight ATOM-menus: real progress per")
    print("  axis -- proximity mines are now learnable and were not -- but the")
    print("  anchor has moved rather than lifted. Lifting it needs ONE substrate")
    print("  in which motion and hazards are both ordinary programs, which is a")
    print("  different experiment and a much larger one.")
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
