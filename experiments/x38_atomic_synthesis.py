"""X38: no menu -- primitives composed from atomic computational fabric.

X37 grew its vocabulary honestly but from a hand-written list: FILTER, DROP,
SORT, HEAD, ORACLE. The system chose among them by evidence and declined
what it could not execute, but it never invented filtration; it recognised
it. That is a selection box, not open-endedness.

Here the menu is deleted. The atoms are:

    c            the element under consideration
    a            the result accumulated from the rest
    NIL          the empty result
    CONS(e, e)   put one result in front of another
    EQ(e, lit)   is this element that literal?
    COND(b, e, e) choose

No atom filters, drops, counts or sorts. `FILTER` is not a symbol anywhere in
this file's vocabulary -- it is a SHAPE that has to be built:

    COND(EQ(c, 'A'), CONS(c, a), a)

...which says "keep it if it matches, otherwise skip it". Nobody wrote that
down. Refutation finds it, and once found it is named and added to the
dictionary as a primitive that did not previously exist.

THE ONE THING KEPT ATOMIC, stated plainly. Recursion itself is a combinator
rather than a synthesised structure: the step expression is folded over the
input. Free self-reference is the honest alternative and it is
computationally hopeless by enumeration -- a filter needs roughly fifteen
atoms, and eight-to-the-fifteenth is not searchable. Folding is atomic
computational fabric in the same sense CONS is; filtration is not. The claim
is that the CAPABILITY is synthesised, not that recursion was reinvented.

WHY IT IS TRACTABLE AT ALL: bottom-up enumeration with observational
equivalence. Expressions are built by size and any two that agree on every
probe are the same expression as far as evidence can ever tell, so only one
survives to be extended. This is the technique that makes enumerative
synthesis work in the literature, and X6 measured the same collapse here --
5,760 hypotheses onto ~135 behaviours.

THE TEST OF GENERATIVITY is not that one target is reachable. A menu with
one item can do that. It is that MANY capabilities nobody enumerated fall
out of the same atoms, and that the system distinguishes them from evidence
alone.

MEASURED: see main().
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

sys.stdout.reconfigure(line_buffering=True)

ALPHABET = "AB"

# ------------------------------------------------------------ the atoms

Expr = tuple | str


def evaluate(expr: Expr, c: str, a: str, depth: int = 0) -> str:
    """Evaluate a step expression. `c` is one element, `a` the rest's result."""
    if depth > 24:
        raise RecursionError
    if expr == "c":
        return c
    if expr == "a":
        return a
    if expr == "NIL":
        return ""
    head = expr[0]
    if head == "CONS":
        left = evaluate(expr[1], c, a, depth + 1)
        right = evaluate(expr[2], c, a, depth + 1)
        if len(left) + len(right) > 256:
            raise RecursionError
        return left + right
    if head == "COND":
        test = predicate(expr[1], c, a, depth + 1)
        return evaluate(expr[2] if test else expr[3], c, a, depth + 1)
    raise ValueError(f"unknown atom {head!r}")


def predicate(expr: Expr, c: str, a: str, depth: int = 0) -> bool:
    head = expr[0]
    if head == "EQ":
        return evaluate(expr[1], c, a, depth + 1) == expr[2]
    raise ValueError(f"unknown predicate {head!r}")


def fold(step: Expr, text: str) -> str:
    """Apply the step expression along the input, right to left.

    The recursion combinator. Everything ABOVE this -- what to keep, what to
    drop, what to duplicate -- is synthesised, not supplied.
    """
    acc = ""
    for ch in reversed(text):
        acc = evaluate(step, ch, acc)
    return acc


def render(expr: Expr) -> str:
    if isinstance(expr, str):
        return expr
    if expr[0] == "EQ":
        return f"(EQ {render(expr[1])} '{expr[2]}')"
    return "(" + " ".join(
        render(p) if not isinstance(p, str) or p in ("c", "a", "NIL")
        else p for p in expr) + ")"


def size(expr: Expr) -> int:
    """Node count. Atoms are 1; a literal inside EQ costs nothing extra.

    Kept deliberately simple: the first version tried to special-case
    literals and disagreed with the enumeration's own indexing, which left
    whole sizes empty (4 and 6 produced zero new behaviours) and made three
    capabilities look inexpressible when they were merely unreachable.
    """
    if isinstance(expr, str):
        return 1
    if expr[0] == "EQ":
        return 1 + size(expr[1])
    return 1 + sum(size(p) for p in expr[1:])


# ------------------------------------------------- bottom-up enumeration

PROBES = tuple(
    (c, a)
    for c in ("A", "B")
    for a in ("", "A", "B", "AB", "BA", "AA")
)


def signature(expr: Expr) -> tuple | None:
    """What an expression DOES on the probe set. None if it cannot run."""
    out = []
    for c, a in PROBES:
        try:
            out.append(evaluate(expr, c, a))
        except (RecursionError, ValueError):
            return None
    return tuple(out)


def enumerate_steps(max_size: int, verbose: bool = False) -> list[Expr]:
    """Build step expressions by size, keeping one per BEHAVIOUR.

    Observational equivalence is what makes this finish. Two expressions
    that agree on every probe are indistinguishable by any evidence, so
    carrying both forward doubles the work and can never change an answer.
    """
    seen: dict[tuple, Expr] = {}
    by_size: dict[int, list[Expr]] = {}

    def add(expr: Expr, n: int) -> None:
        sig = signature(expr)
        if sig is None or sig in seen:
            return
        seen[sig] = expr
        by_size.setdefault(n, []).append(expr)

    for atom in ("c", "a", "NIL"):
        add(atom, 1)

    preds: list[Expr] = []
    for n in range(2, max_size + 1):
        for lit in ALPHABET:
            for e in by_size.get(n - 2, []):
                preds.append(("EQ", e, lit))
        for i in range(1, n):
            for left in by_size.get(i, []):
                for right in by_size.get(n - 1 - i, []):
                    add(("CONS", left, right), n)
        for p in preds:
            for i in range(1, n):
                for t in by_size.get(i, []):
                    for e in by_size.get(n - 1 - size(p) - i, []):
                        if t != e:
                            add(("COND", p, t, e), n)
        if verbose:
            print(f"    size {n}: {len(by_size.get(n, []))} new behaviours "
                  f"({len(seen)} total)")
    return [e for n in sorted(by_size) for e in by_size[n]]


# ------------------------------------------------------------- the tasks

TASKS = {
    "keep only A": lambda s: "".join(c for c in s if c == "A"),
    "drop every A": lambda s: "".join(c for c in s if c != "A"),
    "identity": lambda s: s,
    "double every char": lambda s: "".join(c * 2 for c in s),
    "double only A": lambda s: "".join(c * 2 if c == "A" else c for c in s),
    "erase everything": lambda s: "",
    "keep A, doubled": lambda s: "".join(c * 2 for c in s if c == "A"),
    "replace A with BB": lambda s: "".join("BB" if c == "A" else c for c in s),
}

TRAIN = ("AB", "BAA", "ABBA")
HELD_OUT = ("A", "B", "", "BB", "ABAB", "BBAAB", "AAA")


def main() -> int:
    print("X38: synthesising primitives from atomic fabric -- no menu\n")
    print("atoms: c  a  NIL  CONS(_,_)  EQ(_,lit)  COND(_,_,_)")
    print("no atom filters, drops, counts or sorts.\n")

    t0 = time.perf_counter()
    steps = enumerate_steps(max_size=9, verbose=True)
    gen_dt = time.perf_counter() - t0
    print(f"\n{len(steps):,} behaviourally-distinct step expressions "
          f"({gen_dt:.1f}s)\n")

    print(f'{"capability":20} {"survivors":>9} {"synthesised program":38} {"exact":>6}')
    invented = {}
    exact_count = 0
    for name, fn in TASKS.items():
        survivors = []
        for step in steps:
            ok = True
            for x in TRAIN:
                try:
                    if fold(step, x) != fn(x):
                        ok = False
                        break
                except (RecursionError, ValueError):
                    ok = False
                    break
            if ok:
                survivors.append(step)

        if not survivors:
            print(f"{name:20} {0:>9} {'-- not expressible from these atoms --':38} "
                  f"{'no':>6}")
            continue

        chosen = min(survivors, key=lambda e: (size(e), render(e)))
        exact = True
        for q in HELD_OUT:
            try:
                if fold(chosen, q) != fn(q):
                    exact = False
                    break
            except (RecursionError, ValueError):
                exact = False
                break
        exact_count += int(exact)
        invented[name] = chosen
        print(f"{name:20} {len(survivors):>9,} {render(chosen)[:38]:38} "
              f"{'yes' if exact else 'NO':>6}")

    print(f"\ncapabilities synthesised exactly: {exact_count}/{len(TASKS)}")
    print("none of these were symbols in the vocabulary; each is a SHAPE")
    print("built from atoms and found by refutation against evidence.\n")

    # The generativity check: are these genuinely different functions, or
    # one function wearing several names?
    behaviours = {}
    for name, expr in invented.items():
        key = tuple(fold(expr, q) for q in HELD_OUT)
        behaviours.setdefault(key, []).append(name)
    print(f"distinct behaviours among the synthesised primitives: "
          f"{len(behaviours)}/{len(invented)}")
    for key, names in behaviours.items():
        if len(names) > 1:
            print(f"  equivalent on the held-out set: {', '.join(names)}")

    print(f"\ntotal {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
