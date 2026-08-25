"""X58: the one-rule repair bound never binds, and two of my bugs did.

X57 closed the alphabet gap with a joint rewrite bounded at one inserted rule
and one changed default, and ended by admitting nothing measured how often one
rule is not enough. Three tasks is not a sample. This runs fourteen real
text-parsing tasks -- parsing patterns crossed with delimiters, not random
decision lists, which X55 showed collapse and measure nothing -- through the
whole pipeline.

    task                   nodes   found    raw  +repair  +2 rules
    strip after '#'            7     yes     ok       ok         -
    strip after '"'            7     yes     ok       ok         -
    strip after ';'            7     yes     ok       ok         -
    strip after '|'            7     yes     ok       ok         -
    halt at '#'                6     yes     ok       ok         -
    halt at ';'                6     yes     ok       ok         -
    keep only '#'              7     yes     ok       ok         -
    keep only '|'              7     yes     ok       ok         -
    capture toggle '"'        17     yes     ok       ok         -
    capture toggle '|'        17     yes     ok       ok         -
    capture nested ()         17     yes     ok       ok         -
    skip nested ()            17     yes     ok       ok         -
    capture nested []         17     yes     ok       ok         -
    skip nested []            17     yes     ok       ok         -

    recovered and generalise without repair : 14/14
    needed two rules                        : 0/14
    still failing                           : 0/14

"generalise" means the held-out snippets AND 300 synthesised tapes drawn from
bytes the evidence never contained. The answer to X57's open question is that
the one-rule bound never binds on this family, so raising it is not the next
thing to do.

AND A RESULT THAT QUALIFIES X57. All fourteen generalise WITHOUT any repair,
including `capture toggle '"'` -- the task that needed the three-edit rewrite
in X56 and X57. The difference is the evidence: five richer snippets here
against five thinner ones there. So the repair is not what makes these
programs alphabet-agnostic; better evidence is, and the repair is a fallback
for when the evidence is thin. X57's mechanism is real and its necessity was
overstated by the tapes it was measured on.

TWO BUGS THE SWEEP FOUND, both mine and both invisible on three tasks.

  marker derivation was emission-only. A byte qualified as a marker if it sat
  on a boundary between copied and skipped text -- which says nothing about a
  target that never copies anything. `halt at '#'` derived NO markers and the
  pipeline could not even build a space. Halting is an observable event too:
  a byte is a marker if arriving at it stops the machine. Added as a fallback
  for targets with no emissions.

  the predicate set was offset-0 only. `halt at m` is expressible only as
  IF(m@+1, SEQ(ADV, HALT), ADV), because a decision list cannot test a byte
  the head has not reached. With offset-0 tests the task is not in the
  language at all -- which reads as a search failure and is a vocabulary gap.
  X50 had offsets -1, 0 and +1; X56 dropped to 0 for cost and this is what
  that cost.

A third, smaller one, disclosed because the table above was produced before
it was fixed: the halt fallback clamped pos-1 to 0, which made the FIRST byte
of every tape look halt-associated and derived a spurious 'p' from 'p#q"r"s'.
The effect was an extra stack symbol, so the run above was slightly larger
than it needed to be and every result still held; the two halt tasks were
re-verified after the fix and recover (IF '#'@+1 (SEQ ADV HALT) ADV) with a
single derived marker. The other twelve tasks were unaffected -- their markers
come from the emission rule, which the fallback never reaches.

These were found only by widening the task family, which is the argument for
sweeps over demonstrations.

WHAT THIS DOES NOT SHOW. Fourteen tasks, five snippets of at most ten bytes,
one delimiter alphabet. Every task here is a single pass over one stream with
one stack; nothing tests two streams, a scratchpad, or a program that must
read what it wrote. 2065s.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as S50
import x54_frontier_ranker as R
import x56_byte_vm as B
import x57_repair as P
from x56_byte_vm import OTHER, Space, output, render, size_of

PUSH_ON, POP_ON, EMIT_ON = B.PUSH_ON, B.POP_ON, B.EMIT_ON
TAPES = ['a(b[c])d#e', '"x"|y|z;w', 'p#q"r"s', '[m(n)]|o;', 'u"v";(w)']
EVAL = ['g(h[i])j#k', '"t"|u|v;s', 'q#r"m"n', '[a(z)]|b;', 'c"d";(e)',
        '#only', '"solo"', '(a[b]c)', 'x|y|z', 'plain']
WIDE = "GHJKLMNQRSTUVW0789<>/\\-_*+=~"


def at(c):
    return ("AT", 0, c)


def strip_after(m):
    return ("LOOP", ("IF", at(m), "NOP", EMIT_ON))


def halt_at(m):
    return ("SEQ", "ADV", ("IF", at(m), "HALT", "NOP"))


def keep_markers(m):
    return ("LOOP", ("IF", at(m), EMIT_ON, "ADV"))


def capture_toggle(m):
    return ("LOOP", ("IF", at(m), ("IF", ("EMPTY",), PUSH_ON, POP_ON),
                     ("IF", ("EMPTY",), "ADV", EMIT_ON)))


def capture_nested(o, c):
    return ("LOOP", ("IF", at(o), PUSH_ON,
                     ("IF", at(c), POP_ON,
                      ("IF", ("EMPTY",), "ADV", EMIT_ON))))


def skip_nested(o, c):
    return ("LOOP", ("IF", at(o), PUSH_ON,
                     ("IF", at(c), POP_ON,
                      ("IF", ("EMPTY",), EMIT_ON, "ADV"))))


TASKS = {}
for m in "#\";|":
    TASKS[f"strip after {m!r}"] = strip_after(m)
for m in "#;":
    TASKS[f"halt at {m!r}"] = halt_at(m)
for m in "#|":
    TASKS[f"keep only {m!r}"] = keep_markers(m)
for m in '"|':
    TASKS[f"capture toggle {m!r}"] = capture_toggle(m)
for o, c in (("(", ")"), ("[", "]")):
    TASKS[f"capture nested {o}{c}"] = capture_nested(o, c)
    TASKS[f"skip nested {o}{c}"] = skip_nested(o, c)


def build(truth):
    markers, _ = B.derive_markers(TAPES, truth)
    if not markers:
        return None
    space = Space(TAPES, markers, bound=2)
    alpha = sorted({c for t in TAPES for c in t})
    target = space.interpret(truth)
    stackst = [("EMPTY",)] + [("TOP", m) for m in sorted(markers)] + \
        [("TOP", OTHER)]
    # Offset +1 as well as 0. `halt at m` is only expressible as
    # IF(m@+1, SEQ(ADV, HALT), ADV) -- a decision list cannot look at a byte
    # the head has not reached yet, so with offset-0 tests alone the task is
    # not in the language at all. That read as a search failure and was a
    # vocabulary gap.
    preds = [at(c) for c in alpha + ["$"]]
    preds += [("AT", 1, c) for c in alpha + ["$"]]
    preds += [("EMPTY",)] + stackst
    preds += [("BOTH", at(m), st) for m in sorted(markers) for st in stackst]
    preds += [d[1] for d in B.derive(space, target, alpha, markers)]
    keep, masks, seen = [], [], set()
    for p in preds:
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        keep.append(p)
        masks.append(m)
    acts = [(a, space.atoms[a]) for a in B.ACTIONS]
    acts += [(("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b]))
             for a in B.ACTIONS for b in B.ACTIONS if a != b]
    chars = [at(c) for c in alpha + ["$"]]
    fams = {"char": [(p, space.pred(p)) for p in chars],
            "stack": [(p, space.pred(p)) for p in stackst],
            "char x stack": [(("BOTH", a, b), space.pred(("BOTH", a, b)))
                             for a in chars for b in stackst]}
    return space, markers, target, keep, masks, acts, fams


def repair2(space, expr, target, wrap, preds, actions, fams, cap=90.0):
    """Two inserted rules instead of one. Bounded by wall clock, because the
    point is to find out whether it CAN, not to run it in production."""
    t0 = time.perf_counter()
    chain, default = S50._split(expr)
    if not chain:
        return None
    big = max(range(len(chain)), key=lambda i: size_of(chain[i][0]))
    general = [p for p in preds if size_of(p) <= 3]
    small = [a for a in actions if isinstance(a, str)]
    for g in general:
        for d in small:
            if time.perf_counter() - t0 > cap:
                return None
            trial = [list(r) for r in chain]
            trial[big][0] = g
            for p1 in general:
                for a1 in actions:
                    c1 = trial[:big] + [[p1, a1]] + trial[big:]
                    for p2 in general:
                        for a2 in actions:
                            c2 = c1[:big] + [[p2, a2]] + c1[big:]
                            if np.array_equal(
                                    wrap(space.table(S50._join(c2, d))), target):
                                return S50._join(c2, d)
    return None


def survives(prog, truth, markers, rng):
    if any(output(prog, tp, markers) != output(truth, tp, markers)
           for tp in EVAL):
        return False
    alpha = sorted(set("".join(TAPES)) | set(WIDE))
    for _ in range(300):
        tp = "".join(alpha[int(i)] for i in rng.integers(0, len(alpha), 12))
        if output(prog, tp, markers) != output(truth, tp, markers):
            return False
    return True


def main() -> int:
    t0 = time.perf_counter()
    print("X58: how often does the one-rule repair bound bind?\n")
    print(f"{len(TASKS)} parsing tasks over {len(TAPES)} real snippets, "
          f"{len(EVAL)} held out\n")
    print(f'{"task":22} {"nodes":>5} {"found":>7} {"raw":>6} {"+repair":>8} '
          f'{"+2 rules":>9}')
    print("-" * 62)
    rng = np.random.default_rng(4)
    tally = {"missing": 0, "raw ok": 0, "repair ok": 0, "two ok": 0,
             "still fails": 0}
    for name, truth in TASKS.items():
        got = build(truth)
        if got is None:
            print(f"{name:22} {size_of(truth):>5} {'no markers':>7}")
            tally["missing"] += 1
            continue
        space, markers, target, keep, masks, acts, fams = got
        # Both wraps. `halt at` is a straight-line program, not a loop, and
        # searching only for LOOP(body) made it unreachable -- which read as a
        # derivation failure and was a harness failure.
        expr, wrap, looped = None, None, False
        # Identity first: it costs no loop composition, so a straight-line
        # target is settled in milliseconds instead of after a 400-state
        # LOOP search has been exhausted.
        for w, lo in ((lambda t: t, False), (lambda t: space.loop(t, 14), True)):
            e, _, _, _, _ = R.frontier(space, target, acts, keep, masks, w, 400)
            if e is not None:
                expr, wrap, looped = e, w, lo
                break
        if expr is None:
            print(f"{name:22} {size_of(truth):>5} {'--':>7}")
            tally["missing"] += 1
            continue
        lean = S50.polish(space, expr, keep, target, wrap)
        shape = (lambda e: ("LOOP", e)) if looped else (lambda e: e)
        raw = survives(shape(lean), truth, markers, np.random.default_rng(4))
        fixed, _ = P.repair(space, lean, target, wrap, keep,
                            [e for e, _ in acts], fams)
        one = survives(shape(fixed), truth, markers, np.random.default_rng(4))
        two = "-"
        if not one:
            r2 = repair2(space, lean, target, wrap, keep,
                         [e for e, _ in acts], fams)
            two = "ok" if (r2 is not None and survives(
                shape(r2), truth, markers, np.random.default_rng(4))) else "no"
        tally["raw ok" if raw else "x"] = tally.get("raw ok", 0) + int(raw)
        if one:
            tally["repair ok"] += 1
        elif two == "ok":
            tally["two ok"] += 1
        else:
            tally["still fails"] += 1
        print(f"{name:22} {size_of(truth):>5} {'yes':>7} "
              f"{('ok' if raw else 'FAILS'):>6} {('ok' if one else 'FAILS'):>8} "
              f"{two:>9}")

    n = len(TASKS)
    print(f"\n  recovered and generalise without repair : {tally['raw ok']}/{n}")
    print(f"  fixed by the one-rule repair            : {tally['repair ok']}/{n}")
    print(f"  needed two rules                        : {tally['two ok']}/{n}")
    print(f"  still failing                           : {tally['still fails']}/{n}")
    print(f"  not recovered at all                    : {tally['missing']}/{n}")
    print("\nREADING")
    if tally["two ok"] == 0 and tally["still fails"] == 0:
        print("  the one-rule bound never binds on this family. Raising it is")
        print("  not the next thing to do.")
    elif tally["two ok"] > 0:
        print(f"  the one-rule bound binds on {tally['two ok']}/{n} tasks and a")
        print("  second rule clears them, so the bound is the limit rather")
        print("  than the mechanism.")
    else:
        print(f"  {tally['still fails']}/{n} fail even with two rules, so the")
        print("  limit is not the rule count.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
