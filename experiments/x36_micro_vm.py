"""X36: swap the grid for a universal substrate -- programs over strings.

Every experiment up to X35 inferred SETTINGS of a fixed simulator. The
hypothesis was a `Mechanics` struct and the interpreter was
`gen/grid.transition_state`, hand-written and closed: a world whose physics
the eight axes cannot express is unlearnable, and X33 measured exactly that.
X35's growth mechanism picks which DORMANT ENGINE PRIMITIVE reality demands,
which is real but bounded -- it cannot invent a rule the engine cannot
already execute.

This replaces the substrate. A hypothesis is now an S-expression program
over strings, built from a tiny universal vocabulary:

    x              the input
    (DUP a)        a + a
    (REV a)        a reversed
    (CAT a b)      a + b
    (IF c t e)     t if |c| is even, else e

Composition is unbounded, so the space is generated rather than enumerated
from a fixed cross-product: programs are built by size, and any function
these primitives can compose is expressible. That is the difference between
a parameter space and a program space.

WHAT CARRIES OVER UNCHANGED, and this is the point of the experiment:

    refute in bulk        run every program on the evidence, keep exact matches
    select by Occam       smallest surviving program wins
    disagreement probes   survivors that agree teach nothing; the informative
                          test is the input they disagree about

That last line is Pillar 4 ("the agent synthesises its own adversarial test
suite") and it needs no new idea -- it is `explore/version_space.best_action`
with an input string in place of an action. A self-generated test is just an
experiment, and this project already knows how to choose experiments.

WHAT SELF-TESTING BUYS, AND WHAT IT DOES NOT. Generating tests at all is
decisive: from one example the Occam pick is right on 3 of 8 tasks, and
after self-testing on 8 of 8, with no extra data supplied. CHOOSING the
test by disagreement is not: X39 measures disagreement-selection and
fixed-order selection at 1.00 probes each, across spaces from 64 to 23,713
programs. An earlier 2x here was an artefact of a probe pool that began
with the training input, so the control wasted its first test by
construction.

MEASURED: see the summary printed by main(). Negative results are reported
in the same table as positive ones.
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass

sys.stdout.reconfigure(line_buffering=True)

# --------------------------------------------------------------- the VM

Program = tuple | str


class VMError(RuntimeError):
    """A program that cannot be evaluated. Crashes are evidence too."""


MAX_STRING = 512
"""Programs compose, so `(DUP (DUP (DUP x)))` grows exponentially. A cap
keeps refutation bounded; exceeding it is a refutation, not a crash."""


def evaluate(prog: Program, x: str, depth: int = 0) -> str:
    """Run one program. Deterministic, total, and cheap."""
    if depth > 32:
        raise VMError("recursion")
    if prog == "x":
        return x
    head = prog[0]
    if head == "DUP":
        a = evaluate(prog[1], x, depth + 1)
        if len(a) * 2 > MAX_STRING:
            raise VMError("overflow")
        return a + a
    if head == "REV":
        return evaluate(prog[1], x, depth + 1)[::-1]
    if head == "CAT":
        a = evaluate(prog[1], x, depth + 1)
        b = evaluate(prog[2], x, depth + 1)
        if len(a) + len(b) > MAX_STRING:
            raise VMError("overflow")
        return a + b
    if head == "IF":
        c = evaluate(prog[1], x, depth + 1)
        return evaluate(prog[2] if len(c) % 2 == 0 else prog[3], x, depth + 1)
    raise VMError(f"unknown head {head!r}")


def render(prog: Program) -> str:
    if isinstance(prog, str):
        return prog
    return "(" + " ".join(render(p) for p in prog) + ")"


def size(prog: Program) -> int:
    if isinstance(prog, str):
        return 1
    return 1 + sum(size(p) for p in prog[1:])


# -------------------------------------------------------- the space

UNARY = ("DUP", "REV")
BINARY = ("CAT",)


def enumerate_programs(max_size: int, use_if: bool = False) -> list[Program]:
    """All programs up to `max_size`, smallest first.

    Generated rather than enumerated from a cross-product: this is a
    program space, so its size is a consequence of the grammar and the
    budget, not a number written down in advance.
    """
    by_size: dict[int, list[Program]] = {1: ["x"]}
    for n in range(2, max_size + 1):
        out: list[Program] = []
        for head in UNARY:
            for a in by_size.get(n - 1, []):
                out.append((head, a))
        for head in BINARY:
            for i in range(1, n - 1):
                for a in by_size.get(i, []):
                    for b in by_size.get(n - 1 - i, []):
                        out.append((head, a, b))
        if use_if:
            for i in range(1, n - 2):
                for j in range(1, n - 1 - i):
                    k = n - 1 - i - j
                    for c in by_size.get(i, []):
                        for t in by_size.get(j, []):
                            for e in by_size.get(k, []):
                                if t != e:
                                    out.append(("IF", c, t, e))
        by_size[n] = out
    return [p for n in sorted(by_size) for p in by_size[n]]


# --------------------------------------------------------- refutation


@dataclass(frozen=True, slots=True)
class Evidence:
    """Input/output pairs the program must reproduce exactly."""

    pairs: tuple[tuple[str, str], ...]


def refute(programs: list[Program], evidence: Evidence) -> list[Program]:
    """Keep programs that reproduce every pair exactly.

    The same rule as every earlier experiment: a hypothesis is kept only
    while it has not been proven wrong, and a crash is a refutation rather
    than an exception to handle.
    """
    survivors = []
    for prog in programs:
        ok = True
        for x, want in evidence.pairs:
            try:
                if evaluate(prog, x) != want:
                    ok = False
                    break
            except VMError:
                ok = False
                break
        if ok:
            survivors.append(prog)
    return survivors


def behaviour(prog: Program, probes: tuple[str, ...]) -> tuple:
    """What a program DOES on a probe set -- its behavioural signature.

    X6 measured that the behavioural quotient stays far smaller than the
    hypothesis space, and the same holds here: syntactically distinct
    programs collapse onto few behaviours, which is what makes selecting
    among survivors tractable.
    """
    out = []
    for p in probes:
        try:
            out.append(evaluate(prog, p))
        except VMError:
            out.append(None)
    return tuple(out)


def _agrees(prog: Program, fn, probe: str) -> bool:
    """Does a program match the true function on one input, without crashing."""
    try:
        return evaluate(prog, probe) == fn(probe)
    except VMError:
        return False


# ------------------------------------------------- Pillar 4: self-tests


def adversarial_probe(survivors: list[Program], pool: tuple[str, ...]) -> tuple[str, int]:
    """The input the survivors most disagree about.

    Pillar 4 asks the system to write its own test suite. It needs no new
    machinery: a test that every surviving hypothesis answers identically
    cannot change what is believed, however sensible it looks, so the
    informative test is the one that splits the survivors -- which is
    `explore.version_space.best_action` with a string in place of an action.

    Returns the probe and how many distinct answers it provokes. A count of
    1 means the survivors are behaviourally pinned and no further test can
    separate them.
    """
    best, best_split = pool[0], 1
    for candidate in pool:
        answers = set()
        for prog in survivors:
            try:
                answers.add(evaluate(prog, candidate))
            except VMError:
                answers.add(None)
        if len(answers) > best_split:
            best, best_split = candidate, len(answers)
    return best, best_split


# ------------------------------------------------------------- tasks

TASKS: dict[str, callable] = {
    "identity": lambda s: s,
    "double": lambda s: s + s,
    "reverse": lambda s: s[::-1],
    "reverse-double": lambda s: (s + s)[::-1],
    "double-reverse": lambda s: s[::-1] + s[::-1],
    "quad": lambda s: s * 4,
    "palindrome": lambda s: s + s[::-1],
    "rev-palindrome": lambda s: s[::-1] + s,
}

TRAIN_INPUTS = ("A",)
"""ONE example, on a DEGENERATE input, deliberately.

Pillar 4 only means something when the evidence leaves genuine ambiguity,
and arranging that took two corrections. Two examples pinned every task
outright (measured: zero probes needed on all eight). One example on "AB"
was no better, because distinct functions still disagree there. On "A" they
collide -- s+s, s+reverse(s) and reverse(s)+s all yield "AA" -- so
refutation leaves programs that are genuinely DIFFERENT functions, and the
system has to separate them by testing itself."""
PROBE_POOL = ("ABC", "ABCD", "XY", "Q", "MNOP", "ZZ", "PQRS", "K", "ABAB", "B")
"""The training input is NOT in the pool, and that correction mattered.

An earlier version began the pool with "A" -- the very input the evidence
came from. The arm that takes probes in order therefore spent its first
test on data it already had, which is guaranteed uninformative, and the
resulting 2x advantage for disagreement-selection measured a rigged control
rather than a better strategy. With the overlap removed, X39 measures both
arms at 1.00 probes across space sizes from 64 to 23,713: at this scale
almost any fresh input separates the survivors, so the value is in TESTING
AT ALL, not in choosing the test cleverly."""


def main() -> int:
    print("X36: universal substrate -- S-expression programs over strings\n")

    t0 = time.perf_counter()
    programs = enumerate_programs(max_size=9)
    gen_dt = time.perf_counter() - t0
    print(f"hypothesis space: {len(programs):,} programs up to size 9 "
          f"({gen_dt:.1f}s to generate)")
    print("  generated from a grammar, not a cross-product of fixed axes\n")

    print(f'{"task":16} {"survive":>9} {"pick from 1 example":20} {"ok":>5} '
          f'{"pick after self-test":20} {"ok":>5} {"adv":>4} {"seq":>4}')
    solved = 0
    solved_after = 0
    pinned_counts = []
    sequential_counts = []
    for name, fn in TASKS.items():
        evidence = Evidence(tuple((x, fn(x)) for x in TRAIN_INPUTS))
        t1 = time.perf_counter()
        survivors = refute(programs, evidence)
        dt = time.perf_counter() - t1
        if not survivors:
            print(f"{name:16} {0:>10} {'-- NO PROGRAM EXPRESSES THIS --':28} "
                  f"{'no':>6} {'-':>7}")
            continue

        chosen = min(survivors, key=lambda p: (size(p), render(p)))
        # Does the chosen program agree with the true function everywhere we
        # can check, not merely on the evidence it was selected against?
        exact = all(_agrees(chosen, fn, q) for q in PROBE_POOL)
        solved += int(exact)

        # Pillar 4, with the control that makes it a claim rather than a
        # demonstration: does CHOOSING the test by disagreement beat taking
        # tests in arbitrary order? Both arms get the same budget and the
        # same pool; only the selection rule differs.
        def run_probes(pick_adversarial: bool):
            """Probes until behaviour is pinned. Both arms stop the SAME way.

            The first version stopped the adversarial arm on behavioural
            pinning and the sequential arm on how many programs remained --
            and since syntactically distinct programs can compute the same
            function, that count never falls to one. The control therefore
            spent its whole budget by construction and reported a
            meaningless ten-billion-fold win. Both arms now stop when no
            remaining probe can split the live set.
            """
            live = list(survivors)
            order = list(PROBE_POOL)
            used = 0
            while order:
                probe, split = adversarial_probe(live, tuple(order))
                if split <= 1:
                    break  # behaviourally pinned: no test can separate these
                if not pick_adversarial:
                    probe = order[0]
                order = [q for q in order if q != probe]
                truth_answer = fn(probe)
                live = [p for p in live
                        if behaviour(p, (probe,)) == (truth_answer,)]
                used += 1
            return used, live

        adv_used, adv_live = run_probes(True)
        seq_used, _ = run_probes(False)
        pinned_counts.append(adv_used)
        sequential_counts.append(seq_used)

        # The measurement that matters: does SELF-TESTING repair the answer?
        #
        # `exact` before probing is the Occam pick from one degenerate
        # example, and it is often wrong -- on "A" the identity and the
        # reverse agree, so `reverse` is answered with `x`. Pillar 4's whole
        # claim is that a system generating its own adversarial tests fixes
        # this without being given more data. Measuring only the first pick
        # would hide the entire effect.
        after = min(adv_live, key=lambda p: (size(p), render(p)))
        exact_after = all(_agrees(after, fn, q) for q in PROBE_POOL)
        solved_after += int(exact_after)

        print(f"{name:16} {len(survivors):>9,} {render(chosen)[:20]:20} "
              f"{'yes' if exact else 'NO':>5} {render(after)[:20]:20} "
              f"{'yes' if exact_after else 'NO':>5} {adv_used:>4} {seq_used:>4}")

    print(f"\nsolved exactly on held-out inputs:")
    print(f"  from one example, no self-testing : {solved}/{len(TASKS)}")
    print(f"  after self-generated adversarial tests: {solved_after}/{len(TASKS)}")
    if pinned_counts:
        adv = sum(pinned_counts) / len(pinned_counts)
        seq = sum(sequential_counts) / len(sequential_counts)
        print(f"self-generated tests to pin behaviour:")
        print(f"  chosen by disagreement : mean {adv:.2f}  max {max(pinned_counts)}")
        print(f"  taken in order (control): mean {seq:.2f}  max {max(sequential_counts)}")
        if adv < seq - 1e-9:
            print(f"  -> choosing by disagreement is {seq/max(adv, 1e-9):.2f}x cheaper")
        elif adv > seq + 1e-9:
            print("  -> WORSE than arbitrary order")
        else:
            print("  -> NO ADVANTAGE: identical cost to arbitrary order")
    print(f"\ntotal {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
