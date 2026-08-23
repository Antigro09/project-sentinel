"""X41: does a minted primitive make the NEXT problem cheaper -- or possible?

Claim 4 of the plan -- that intelligence accumulates, that environment N+1
costs less than N -- has never been supported here. `memory/` retrieves
verified rule sets and reorders search, but nothing is ever CREATED, so
nothing compounds. X40 changed that: it mints recursive programs from raw
atoms. The question is whether a minted primitive is worth keeping.

The honest test is not "is it faster". Shaving seconds off a search is an
optimisation. The test is whether something becomes REACHABLE that was not:

    stage 1   synthesise `keepA` from atoms                 (X40's job)
    stage 2   attempt a COMPOSITE task -- keep the A's and
              double them -- with and without `keepA` in
              the vocabulary

Without the primitive the composite needs a program deep enough that the
enumeration cannot reach it inside any budget worth spending. With it, the
same task is a few atoms. If that gap holds, accumulation is real here for
the first time; if the composite is reachable either way, the library is an
optimisation and should be described as one.

THE COST THE LIBRARY CHARGES, measured rather than assumed: every minted
primitive is another atom, and atoms multiply the branching factor at every
size. A library that grows without bound makes the search it was meant to
help strictly worse. That trade is reported below alongside the win.

MEASURED: see main().
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x40_free_recursion as x40
from x40_free_recursion import (
    Refuted,
    render,
    size,
    suffix_closure,
)

LIBRARY_PATH = Path("corpus/micro_library.json")


# ------------------------------------------------------------- library


def load_library() -> dict:
    if not LIBRARY_PATH.exists():
        return {}
    return json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))


def save_library(lib: dict) -> None:
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY_PATH.write_text(json.dumps(lib, indent=2), encoding="utf-8")


def to_jsonable(expr):
    if isinstance(expr, str):
        return expr
    return [expr[0]] + [to_jsonable(p) for p in expr[1:]]


def from_jsonable(data):
    if isinstance(data, str):
        return data
    return tuple([data[0]] + [from_jsonable(p) for p in data[1:]])


# ------------------------------------------- evaluation with primitives


def make_evaluator(primitives: dict):
    """An evaluator where each library entry is callable as an atom.

    A minted primitive is applied to the WHOLE input and runs its own
    recursion internally, which is what makes it a genuine abstraction
    rather than a macro: the composite program does not have to re-derive
    the loop, only to use its result.
    """

    def evaluate(expr, x: str, memo: dict):
        if isinstance(expr, str) and expr in primitives:
            body = primitives[expr]
            return x40.run(body, suffix_closure(x)).get(x)
        if expr == "REC":
            tail = x[1:]
            if x == "" or memo.get(tail) is None:
                raise Refuted("no base")
            return memo[tail]
        if isinstance(expr, str):
            return x40.evaluate(expr, x, memo)
        head = expr[0]
        if head == "CAT":
            left = evaluate(expr[1], x, memo)
            right = evaluate(expr[2], x, memo)
            if left is None or right is None:
                raise Refuted("undefined")
            if len(left) + len(right) > x40.MAX_LEN:
                raise Refuted("overflow")
            return left + right
        if head == "COND":
            pred = expr[1]
            val = evaluate(pred[1], x, memo)
            if val is None:
                raise Refuted("undefined")
            hit = (val == "") if pred[0] == "EMPTY" else (val == pred[2])
            return evaluate(expr[2] if hit else expr[3], x, memo)
        raise Refuted("unknown")

    def run(expr, inputs):
        memo: dict[str, str | None] = {}
        for s in sorted(inputs, key=len):
            try:
                memo[s] = evaluate(expr, s, memo)
            except Refuted:
                memo[s] = None
        return memo

    return run


def enumerate_with(primitives: dict, inputs, max_size: int, run_fn):
    """Bottom-up enumeration where library entries are extra atoms."""
    seen: dict = {}
    rec_seen: set = set()
    pure: dict[int, list] = {}
    recs: dict[int, list] = {}

    def add(expr, n):
        if x40.contains_rec(expr):
            sig = x40.rec_signature(expr, inputs)
            if sig in rec_seen:
                return
            rec_seen.add(sig)
            recs.setdefault(n, []).append(expr)
            return
        table = run_fn(expr, inputs)
        if all(v is None for v in table.values()):
            return
        sig = tuple(table[s] for s in inputs)
        if sig in seen:
            return
        seen[sig] = expr
        pure.setdefault(n, []).append(expr)

    for atom in x40.ATOMS:
        add(atom, 1)
    for name in primitives:
        add(name, 1)

    def blocks(n):
        return pure.get(n, []) + recs.get(n, [])

    preds = []
    for n in range(2, max_size + 1):
        for smaller in range(1, n):
            for e in pure.get(smaller, []):
                preds.append(("EMPTY", e))
                for lit in x40.ALPHABET:
                    preds.append(("EQ", e, lit))
        for i in range(1, n):
            for left in blocks(i):
                for right in blocks(n - 1 - i):
                    add(("CAT", left, right), n)
        for p in preds:
            ps = size(p)
            for i in range(1, n):
                rest = n - 1 - ps - i
                if rest < 1:
                    continue
                for t in blocks(i):
                    for e in blocks(rest):
                        if t is not e:
                            add(("COND", p, t, e), n)
    return [e for n in sorted(set(pure) | set(recs)) for e in blocks(n)]


# --------------------------------------------------------------- tasks

KEEP_A = lambda s: "".join(c for c in s if c == "A")
COMPOSITE = lambda s: "".join(c * 2 for c in s if c == "A")

TRAIN = ("ABB", "BAAB", "AB")
HELD_OUT = ("", "A", "B", "AA", "BB", "ABAB", "AAAB", "BBB", "BABA")
SELF_TESTS = ("A", "B", "", "AA", "BB", "AAB", "BBA", "ABAB", "BAB", "AAA")


def attempt(fn, primitives: dict, max_size: int, label: str):
    """Synthesise `fn`, self-test the survivors, verify on held-out inputs."""
    run_fn = make_evaluator(primitives)
    inputs = suffix_closure(*TRAIN)
    t0 = time.perf_counter()
    candidates = enumerate_with(primitives, inputs, max_size, run_fn)
    survivors = [
        e for e in candidates
        if all(run_fn(e, inputs).get(s) == fn(s) for s in TRAIN)
    ]
    if not survivors:
        return {"label": label, "found": False, "candidates": len(candidates),
                "secs": time.perf_counter() - t0, "program": None,
                "exact": False, "tests": 0}

    live = list(survivors)
    tests = 0
    for _ in range(6):
        if len(live) <= 1:
            break
        best, split = None, 1
        for q in SELF_TESTS:
            answers = {run_fn(e, suffix_closure(q)).get(q) for e in live}
            if len(answers) > split:
                best, split = q, len(answers)
        if best is None:
            break
        want = fn(best)
        live = [e for e in live if run_fn(e, suffix_closure(best)).get(best) == want]
        tests += 1
    if not live:
        live = survivors

    chosen = min(live, key=lambda e: (size(e), render(e)))
    exact = all(run_fn(chosen, suffix_closure(q)).get(q) == fn(q) for q in HELD_OUT)
    return {"label": label, "found": True, "candidates": len(candidates),
            "secs": time.perf_counter() - t0, "program": chosen,
            "exact": exact, "tests": tests}


def main() -> int:
    print("X41: do minted primitives make the next problem cheaper, or possible?\n")
    if LIBRARY_PATH.exists():
        LIBRARY_PATH.unlink()

    # -------------------------------------------------- stage 1: mint
    print("STAGE 1 -- mint `keepA` from raw atoms")
    first = attempt(KEEP_A, {}, max_size=11, label="keep only A")
    print(f"  {first['candidates']:,} candidates, {first['secs']:.1f}s, "
          f"{first['tests']} self-tests")
    print(f"  {render(first['program']) if first['program'] else '--'}")
    print(f"  exact on held-out: {'YES' if first['exact'] else 'NO'}")
    if not first["exact"]:
        print("  minting failed; nothing to accumulate")
        return 1

    lib = {"keepA": to_jsonable(first["program"])}
    save_library(lib)
    print(f"  saved to {LIBRARY_PATH}\n")

    # ------------------------------ stage 2: composite, with and without
    print("STAGE 2 -- composite task: keep the A's AND double them")
    primitives = {name: from_jsonable(body) for name, body in load_library().items()}

    cold = attempt(COMPOSITE, {}, max_size=11, label="cold")
    warm = attempt(COMPOSITE, primitives, max_size=5, label="warm")

    print(f"  without library (size<=11): "
          f"{'found' if cold['found'] else 'NOT FOUND'}, "
          f"{cold['candidates']:,} candidates, {cold['secs']:.1f}s, "
          f"exact={'yes' if cold['exact'] else 'no'}")
    print(f"  with `keepA`   (size<=5) : "
          f"{'found' if warm['found'] else 'NOT FOUND'}, "
          f"{warm['candidates']:,} candidates, {warm['secs']:.1f}s, "
          f"exact={'yes' if warm['exact'] else 'no'}")
    if warm["program"] is not None:
        print(f"  warm program: {render(warm['program'])}")

    print()
    if warm["exact"] and not cold["exact"]:
        print("  ACCUMULATION IS REAL: the composite is out of reach from atoms")
        print("  at this budget and trivial once the primitive exists.")
    elif warm["exact"] and cold["exact"]:
        ratio = cold["candidates"] / max(1, warm["candidates"])
        print(f"  both reachable; the library is an OPTIMISATION "
              f"({ratio:.0f}x fewer candidates, "
              f"{cold['secs']/max(warm['secs'],1e-9):.0f}x faster)")
    else:
        print("  no benefit measured")

    # ------------------------------------------- the cost of the library
    print("\nTHE COST THE LIBRARY CHARGES")
    inputs = suffix_closure(*TRAIN)
    bare = len(enumerate_with({}, inputs, 5, make_evaluator({})))
    withlib = len(enumerate_with(primitives, inputs, 5, make_evaluator(primitives)))
    print(f"  candidates at size<=5: {bare:,} bare, {withlib:,} with 1 primitive "
          f"({withlib/max(bare,1):.2f}x)")
    print("  every minted atom multiplies branching at every size, so a library")
    print("  that grows without pruning makes the search it was meant to help worse.")
    print(f"\ntotal {time.perf_counter():.0f}s process time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
