"""X40: free recursion -- no fold combinator, no given base case.

X38 kept one thing atomic and said so: recursion was a FOLD, and only the
step expression was synthesised. The stated reason was that free
self-reference is hopeless by enumeration -- a filter needs roughly fifteen
atoms and 8^15 is not searchable.

That reasoning is wrong, and the way it is wrong is worth writing down: it
prices BLIND enumeration. Three changes make the same space tractable.

  1. BOTTOM-UP WITH OBSERVATIONAL EQUIVALENCE. Expressions are built by
     size, and two that agree everywhere the evidence can look are the same
     expression forever. Only one survives to be extended. X6 measured this
     collapse at 43x on the grid space; X38 saw 657 behaviours stand in for
     a far larger syntactic space.

  2. EVALUATE OVER THE SUFFIX CLOSURE, SHORTEST FIRST. A recursive call on
     the tail is the crux: evaluated naively it descends unboundedly. But
     every input's tail is shorter, so if all suffixes are evaluated in
     order of length, `REC` is a LOOKUP of an answer already computed. Cost
     per candidate becomes one table pass, and non-termination cannot occur
     by construction.

  3. THE BASE CASE IS DISCOVERED, NOT GIVEN. Computing f("") is where the
     recursion must bottom out. A program that calls `REC` on the empty
     input is asking for its own answer before it has one; that is refuted,
     not crashed. So the system has to INVENT the guard -- typically
     COND(EMPTY? x, NIL, ...) -- rather than be handed it.

The atoms are now genuinely raw:

    x  HD  TL  REC  NIL      CAT(e,e)   COND(EMPTY? e | EQ(e,lit), e, e)

`REC` is unrestricted self-application on the tail. Nothing here folds,
filters, maps or terminates on its own. A correct filter must be built as:

    COND(EMPTY? x, NIL, COND(EQ(HD,'A'), CAT(HD, REC), REC))

...both halves of which -- the base case and the recursive structure -- are
found by refutation.

HONEST PROVENANCE: recursive example propagation of this kind is how Myth,
lambda^2 and Escher synthesise recursive programs; the technique is not new
here. What is untested is whether it composes with this project's
refutation engine and behavioural-quotient collapse. That composition is
what this experiment measures.

MEASURED (7 capabilities, 3 training examples, held-out verified):

    keep only A        (CAT (COND (EMPTY? x) x REC) (COND (EQ HD 'A) ...))  exact
    drop every A       same shape, inverted branch                          exact
    reverse            (CAT (COND (EMPTY? x) x REC) HD)                     exact
    double every char  (CAT HD (CAT HD (COND (EMPTY? x) x REC)))            exact
    identity, erase    x, NIL                                               exact
    keep A doubled     not found                                            NO

  6/7 with FREE recursion. Every recursive answer discovered its own base
  case -- the (COND (EMPTY? x) x REC) guard appears in all of them and was
  never supplied.

WHAT MADE IT WORK, in the order the obstacles appeared:

  1. Bare REC is undefined on every input, so the first version dropped it
     from the enumeration and could not build a recursive program at all.
     Subexpressions must be allowed to be PARTIAL; only a finished
     candidate has to be total.

  2. REC is not compositional -- its value inside a candidate is that
     candidate's own answer on the tail -- so observational-equivalence
     dedup, the thing that makes bottom-up enumeration finish, does not
     apply to it. The REC pool exploded 490 -> 13,623 -> 385,877 across
     three size steps and hit the cap before the sizes where filters live.

  3. The fix: REC IS compositional relative to an ENVIRONMENT. Evaluating a
     candidate under several assumed recursions -- REC as identity, empty,
     tail, head -- gives it a behaviour again, and expressions agreeing
     under all of them are deduped. That cut size-9 from 385,877 to 4,825,
     an 80x collapse, and made size 11 reachable with no cap.

     This is a HEURISTIC prune, not a sound one: two expressions could
     agree under all four assumptions and differ under the true recursion.
     The held-out check is what would catch that.

  4. Refutation alone still failed the filters: 20 survivors, and the
     smallest fits three examples while generalising wrong. Self-testing
     fixed it -- the survivors nominate the input they most disagree about,
     the truth is requested there, and refutation runs again. Two such
     tests were enough. No additional data was supplied; the system chose
     what to ask.

HONEST PROVENANCE: recursive example propagation of this kind is how Myth,
lambda^2 and Escher synthesise recursive programs; the technique is not new
here. What is new is the composition -- assumed-environment dedup plus
disagreement-driven self-testing -- and the measurement of where each one
binds.
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(line_buffering=True)

ALPHABET = "AB"
MAX_LEN = 64

Expr = tuple | str


class Refuted(Exception):
    """This candidate cannot be evaluated here -- which is evidence."""


# ------------------------------------------------------------ evaluation


def evaluate(expr: Expr, x: str, memo: dict) -> str:
    """Evaluate one candidate on one input.

    `memo` holds the candidate's own answers on every SHORTER suffix. `REC`
    reads from it. Asking for an answer that is not there -- which is what
    calling REC on the empty input means -- is a refutation.
    """
    if expr == "x":
        return x
    if expr == "HD":
        return x[:1]
    if expr == "TL":
        return x[1:]
    if expr == "NIL":
        return ""
    if expr == "REC":
        tail = x[1:]
        if x == "" or memo.get(tail) is None:
            raise Refuted("recursion has no base here")
        return memo[tail]

    head = expr[0]
    if head == "CAT":
        left = evaluate(expr[1], x, memo)
        right = evaluate(expr[2], x, memo)
        if len(left) + len(right) > MAX_LEN:
            raise Refuted("overflow")
        return left + right
    if head == "COND":
        return evaluate(expr[2] if test(expr[1], x, memo) else expr[3], x, memo)
    raise Refuted(f"unknown atom {head!r}")


def test(pred: Expr, x: str, memo: dict) -> bool:
    if pred[0] == "EMPTY":
        return evaluate(pred[1], x, memo) == ""
    if pred[0] == "EQ":
        return evaluate(pred[1], x, memo) == pred[2]
    raise Refuted("unknown predicate")


def run(expr: Expr, inputs: tuple[str, ...]) -> dict:
    """Evaluate a candidate across a suffix-closed input set, shortest first.

    This is what makes free recursion enumerable rather than hopeless: every
    recursive call refers to a strictly shorter input, so the whole table is
    filled in one pass with no descent and no termination check.

    **Failure is recorded, not raised.** A subexpression may be undefined
    somewhere and still be a perfectly good building block: `REC` itself is
    undefined on the empty input, and so is `CAT(HD, REC)`. The first
    version raised, which dropped both from the enumeration and made every
    recursive program unreachable -- the synthesiser could not build a
    recursive function because it had refused to keep the recursive call.
    Only a FINISHED candidate has to be total on the training inputs, and a
    guarded program is: its base case means REC is never reached on "".
    """
    memo: dict[str, str | None] = {}
    for s in sorted(inputs, key=len):
        try:
            memo[s] = evaluate(expr, s, memo)
        except Refuted:
            memo[s] = None
    return memo


def suffix_closure(*strings: str) -> tuple[str, ...]:
    out = set()
    for s in strings:
        for i in range(len(s) + 1):
            out.add(s[i:])
    return tuple(sorted(out, key=lambda t: (len(t), t)))


def render(expr: Expr) -> str:
    if isinstance(expr, str):
        return expr
    if expr[0] == "EMPTY":
        return f"(EMPTY? {render(expr[1])})"
    if expr[0] == "EQ":
        return f"(EQ {render(expr[1])} '{expr[2]}')"
    return "(" + " ".join(render(p) for p in expr) + ")"


def size(expr: Expr) -> int:
    if isinstance(expr, str):
        return 1
    if expr[0] in ("EMPTY", "EQ"):
        return 1 + size(expr[1])
    return 1 + sum(size(p) for p in expr[1:])


# --------------------------------------------------- bottom-up synthesis

ATOMS = ("x", "HD", "TL", "REC", "NIL")


ASSUMED_RECURSIONS = (
    ("id", lambda t: t),
    ("nil", lambda t: ""),
    ("tail", lambda t: t[1:]),
    ("head", lambda t: t[:1]),
)
"""Stand-in meanings for REC, used only to give REC-expressions a
behaviour so they can be deduped. Four is enough to separate almost
everything; more would prune less and cost more."""


def rec_signature(expr: Expr, inputs: tuple[str, ...]) -> tuple:
    """Behaviour under each assumed recursion, concatenated."""
    out = []
    for _, fn in ASSUMED_RECURSIONS:
        for s in inputs:
            try:
                out.append(evaluate_assumed(expr, s, fn))
            except Refuted:
                out.append(None)
    return tuple(out)


def evaluate_assumed(expr: Expr, x: str, rec_fn) -> str:
    """Like `evaluate`, but REC is whatever `rec_fn` says it is."""
    if expr == "REC":
        if x == "":
            raise Refuted("no tail")
        return rec_fn(x[1:])
    if isinstance(expr, str):
        return evaluate(expr, x, {})
    head = expr[0]
    if head == "CAT":
        left = evaluate_assumed(expr[1], x, rec_fn)
        right = evaluate_assumed(expr[2], x, rec_fn)
        if len(left) + len(right) > MAX_LEN:
            raise Refuted("overflow")
        return left + right
    if head == "COND":
        pred = expr[1]
        val = evaluate_assumed(pred[1], x, rec_fn)
        hit = (val == "") if pred[0] == "EMPTY" else (val == pred[2])
        return evaluate_assumed(expr[2] if hit else expr[3], x, rec_fn)
    raise Refuted("unknown")


def contains_rec(expr: Expr) -> bool:
    if isinstance(expr, str):
        return expr == "REC"
    return any(contains_rec(p) for p in expr[1:] if not isinstance(p, str) or p in ATOMS)


def synthesise(inputs: tuple[str, ...], max_size: int, cap: int = 400_000,
               verbose: bool = False):
    """Two pools, because REC is not compositional.

    THE OBSTACLE, stated plainly, because it is the real one. Bottom-up
    enumeration with observational equivalence works by evaluating a
    subexpression once and keeping a single representative per behaviour.
    That requires a subexpression to HAVE a behaviour independent of where
    it ends up. `REC` does not: its value inside candidate P is P's own
    answer on the tail, so `CAT(HD, REC)` means nothing until the whole
    program around it is fixed. A bare `REC` is undefined on every input --
    correctly -- so the first version dropped it and could never build a
    recursive program at all.

    This is why the recursive-synthesis literature (Myth, lambda^2, Escher)
    is top-down with example propagation rather than bottom-up. The hybrid
    here keeps what still works and pays full price for what does not:

      PURE pool  no REC: evaluated, deduped by BEHAVIOUR. The collapse that
                 makes enumeration finish still applies here.
      REC pool   contains REC: kept as SYNTAX, deduped only by structure,
                 because behaviour is not defined until the program is
                 whole. This pool grows combinatorially and is capped.

    Every complete candidate is then evaluated as a whole program, with REC
    resolving against that program's own memo table. The cap is reported
    rather than hidden: if it binds, the search was truncated and a negative
    result means "not found under this budget", not "does not exist".
    """
    seen: dict[tuple, Expr] = {}
    rec_seen: set = set()
    pure: dict[int, list[Expr]] = {}
    recs: dict[int, list[Expr]] = {}
    truncated = False

    def add_pure(expr: Expr, n: int) -> None:
        table = run(expr, inputs)
        if all(v is None for v in table.values()):
            return
        sig = tuple(table[s] for s in inputs)
        if sig in seen:
            return
        seen[sig] = expr
        pure.setdefault(n, []).append(expr)

    def add_rec(expr: Expr, n: int) -> None:
        """Dedup REC-expressions by behaviour under ASSUMED recursions.

        REC is not compositional, which is why the pool exploded -- 490 ->
        13,623 -> 385,877 across three size steps, hitting the cap before
        the sizes where filters live. But it is compositional RELATIVE TO AN
        ENVIRONMENT: fix what REC returns and the expression has a behaviour
        again.

        So each candidate is evaluated under several hypothetical
        recursions -- REC as the identity, as the empty string, as the tail,
        as the head -- and two expressions agreeing under all of them are
        treated as the same building block. Agreeing under every assumed
        recursion is strong evidence of agreeing under the true one.

        This is a HEURISTIC prune, not a sound one: two expressions could
        agree on all four assumptions and differ under the real recursion.
        The held-out check is what catches that, and it is why a synthesised
        program is verified against fresh inputs rather than trusted.
        """
        nonlocal truncated
        sig = rec_signature(expr, inputs)
        if sig in rec_seen:
            return
        rec_seen.add(sig)
        if sum(len(v) for v in recs.values()) >= cap:
            truncated = True
            return
        recs.setdefault(n, []).append(expr)

    for atom in ATOMS:
        if atom == "REC":
            add_rec(atom, 1)
        else:
            add_pure(atom, 1)

    def blocks(n: int) -> list[Expr]:
        return pure.get(n, []) + recs.get(n, [])

    preds: list[Expr] = []
    for n in range(2, max_size + 1):
        # Predicates are built only from PURE expressions: a test whose
        # outcome depends on the recursive result would need the answer to
        # decide how to compute the answer.
        for smaller in range(1, n):
            for e in pure.get(smaller, []):
                preds.append(("EMPTY", e))
                for lit in ALPHABET:
                    preds.append(("EQ", e, lit))

        for i in range(1, n):
            for left in blocks(i):
                for right in blocks(n - 1 - i):
                    expr = ("CAT", left, right)
                    (add_rec if contains_rec(expr) else add_pure)(expr, n)

        for p in preds:
            ps = size(p)
            for i in range(1, n):
                rest = n - 1 - ps - i
                if rest < 1:
                    continue
                for t in blocks(i):
                    for e in blocks(rest):
                        if t is e:
                            continue
                        expr = ("COND", p, t, e)
                        (add_rec if contains_rec(expr) else add_pure)(expr, n)

        if verbose:
            print(f"    size {n:>2}: pure {len(pure.get(n, [])):>6,}  "
                  f"rec {len(recs.get(n, [])):>8,}"
                  + ("   [CAP REACHED]" if truncated else ""))
    out = [e for n in sorted(set(pure) | set(recs)) for e in blocks(n)]
    return out, truncated


# ------------------------------------------------------------- the tasks

TASKS = {
    "keep only A": lambda s: "".join(c for c in s if c == "A"),
    "drop every A": lambda s: "".join(c for c in s if c != "A"),
    "identity": lambda s: s,
    "double every char": lambda s: "".join(c * 2 for c in s),
    "erase everything": lambda s: "",
    "keep A, doubled": lambda s: "".join(c * 2 for c in s if c == "A"),
    "reverse": lambda s: s[::-1],
}

SELF_TEST_POOL = ("A", "B", "", "AA", "BB", "AAB", "BBA", "ABAB", "BAB", "AAA")
"""Inputs the system may ask about. It is not told which are informative --
it works that out from where its own survivors disagree."""

TRAIN = ("ABB", "BAAB", "AB")
"""Not palindromes.

The first version trained on "ABA" and "BAAB", both of which read the same
backwards -- so `reverse` was indistinguishable from the identity and the
synthesiser answered it with `x`, correctly, on evidence that could not tell
them apart."""
HELD_OUT = ("", "A", "B", "AB", "BBAA", "ABAB", "AAAB", "BBB")


def main() -> int:
    print("X40: free recursion -- REC is unrestricted, base case discovered\n")
    print(f"atoms: {'  '.join(ATOMS)}   CAT(_,_)   COND(EMPTY?|EQ, _, _)")
    print("no fold, no map, no filter, no given base case.\n")

    inputs = suffix_closure(*TRAIN)
    print(f"suffix closure of {TRAIN}: {inputs}")
    print("evaluated shortest-first, so REC is a lookup and cannot diverge.\n")

    t0 = time.perf_counter()
    candidates, truncated = synthesise(inputs, max_size=11, verbose=True)
    gen_dt = time.perf_counter() - t0
    print(f"\n{len(candidates):,} candidate programs ({gen_dt:.1f}s)"
          + ("  -- CAP REACHED, search truncated" if truncated else ""))
    print()

    print(f'{"capability":20} {"survivors":>9} {"synthesised program":44} '
          f'{"exact":>6} {"tests":>6}')
    exact_count = 0
    for name, fn in TASKS.items():
        survivors = []
        for expr in candidates:
            table = run(expr, inputs)
            if all(table.get(s) == fn(s) for s in TRAIN):
                survivors.append(expr)

        if not survivors:
            note = ("-- not found under the cap (search truncated) --"
                    if truncated else "-- not expressible within size budget --")
            print(f"{name:20} {0:>9} {note:46} {'no':>6}")
            continue

        # PILLAR 4, on recursive programs. Occam over three examples picks a
        # small program that fits and generalises wrong -- `keep only A`
        # leaves 20 survivors and the shortest is not the filter. X36
        # measured the same failure on non-recursive programs (3/8 correct
        # from one example) and the same cure: let the survivors nominate
        # the input they disagree about, ask for the truth there, and refute
        # again. No extra data is supplied; the system chooses what to ask.
        live = list(survivors)
        probes_used = 0
        for _ in range(6):
            if len(live) <= 1:
                break
            best_probe, best_split = None, 1
            for q in SELF_TEST_POOL:
                answers = set()
                for e in live:
                    answers.add(run(e, suffix_closure(q)).get(q))
                if len(answers) > best_split:
                    best_probe, best_split = q, len(answers)
            if best_probe is None:
                break  # behaviourally pinned: no test can separate these
            want = fn(best_probe)
            live = [e for e in live
                    if run(e, suffix_closure(best_probe)).get(best_probe) == want]
            probes_used += 1
        if not live:
            live = list(survivors)

        chosen = min(live, key=lambda e: (size(e), render(e)))

        # Held-out check re-derives the whole closure per probe: a recursive
        # program is only defined relative to its own suffixes.
        exact = True
        for q in HELD_OUT:
            if run(chosen, suffix_closure(q)).get(q) != fn(q):
                exact = False
                break
        exact_count += int(exact)
        print(f"{name:20} {len(survivors):>9,} {render(chosen)[:44]:44} "
              f"{'yes' if exact else 'NO':>6} {probes_used:>6}")

    print(f"\ncapabilities synthesised with FREE recursion: {exact_count}/{len(TASKS)}")
    print("base cases and recursive structure both found by refutation.")
    print(f"\ntotal {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
