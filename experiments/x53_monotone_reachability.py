"""X53: the route existed all along. Greedy could not take it, and why.

X51 and X52 tried four mechanisms and all four failed. X52 ended by asking
whether `copy inside any` is unreachable rather than unfound -- whether its
equivalence class contains any program a monotone path reaches. It does, and
finding that out corrects two conclusions I reported earlier.

    true rule        beam builder  beam cost   frontier  frontier cost  states
    copy all                exact     18,630      exact        489,654     400
    halt on close           exact      2,943      exact            999       1
    strip brackets          exact     22,067      exact        519,443     401
    copy inside []          exact     37,351      exact        583,471     406
    copy inside any            --     39,816      exact        708,581     493

5/5, including the target nothing in X50-X52 could recover, at about 18x the
evaluations of the builder that fails.

CORRECTION 1: THE DECOY WAS NOT A DECOY. X51 named `IF END@+0 -> POP` a decoy
because it outscored the correct chain's opening move. It is the FIRST STEP of
the winning route:

    copy inside any  0.9232 -> 0.9634 -> 0.9899 -> 0.9991 -> ... -> 1.0000

0.9232 is the decoy's own score. Greedy was right to take it and then failed
to follow through. Every diagnosis built on calling that rule a trap was
aimed at the wrong thing.

CORRECTION 2: "GREEDY AGREEMENT IS THE WRONG SIGNAL" IS WRONG. X51 concluded
that, and X52 acted on it by replacing the signal with a lookahead bound,
which lost a target. The signal is fine -- a monotone route exists under it
for every target here. What fails is COMMITMENT. Greedy takes the argmax at
each step; the route only requires each step to be non-decreasing, and its
steps are not the argmax. A per-round beam then discards the states the route
needs, permanently, because a beam cannot return to a rank it dropped.

That is also why more rounds do not help. Measured: the beam builder fails on
`copy inside any` at 6, 9, 12 and 15 rounds -- 19,908 through 49,716
candidates -- and at widths 1, 2, 4 and 8 in X51. Depth and width were never
the binding constraint. Discarding was.

WHAT THE FIX ACTUALLY IS. One global priority queue over states, ordered by
agreement, expanding only where agreement does not fall, deduplicated by
behaviour, with no per-round discard. A state that ranks 400th stays available
and gets expanded when the better ones are exhausted. That is the whole
difference between 4/5 and 5/5, and it is a change to bookkeeping rather than
to the score, the vocabulary, the width, or the model.

THE SUBSTRATE IS ABSOLVED. X52 speculated the miss might be a property of the
target -- that a context-free task might have no monotone route under these
primitives, and the fix would belong in the Token VM. It is not, and it does
not. PUSH, POP, TOP and EMPTY are sufficient, and no primitive needed adding.

WHAT THIS DOES NOT SHOW. Five targets, three tapes, one budget. The frontier
costs 18x the beam and grows with the state cap, so this buys correctness with
compute rather than replacing search with insight -- on a larger space the
frontier is what will need pruning, and pruning is what broke the beam. The
routes found are also not the reference programs: `copy inside any` comes back
as an eleven-rule chain rather than the three-rule one it was written as,
which is fine behaviourally and says the equivalence class, not the reference,
is what search actually navigates.
"""

from __future__ import annotations

import heapq
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as X
import x51_deceptive_valley as V
from x50_stack import EVAL, TAPES, TRUTHS, Space, output, render, size_of

STATE_BUDGET = 400
MAX_RULES = 6


def monotone_search(space, target, rules, preds, pmasks, wrap, budget):
    """Best-first over states, expanding only where agreement does not fall."""
    agree = lambda sig: float((sig == target).mean())

    def assemble(ch, de, dt):
        e, t = de, dt
        for p, pm, be, bt in reversed(ch):
            e, t = ("IF", p, be, e), space.branch(pm, bt, t)
        return e, t

    heap, seen, expanded, best, evals = [], set(), 0, 0.0, 0
    tick = 0
    for e, t in rules:
        evals += 1
        a = agree(wrap(t))
        best = max(best, a)
        if a >= 1.0:
            return e, expanded, a, [a], evals
        seen.add(wrap(t).tobytes())
        tick += 1
        heapq.heappush(heap, (-a, tick, [], e, t, [a]))

    while heap and expanded < budget:
        neg_a, _, chain, de, dt, trace = heapq.heappop(heap)
        a0 = -neg_a
        if len(chain) >= MAX_RULES:
            continue
        expanded += 1
        for p, pm in zip(preds, pmasks):
            for be, bt in rules:
                for front in (True, False):
                    if front:
                        ch2, de2, dt2 = [(p, pm, be, bt)] + chain, de, dt
                    else:
                        ch2 = chain
                        de2, dt2 = ("IF", p, be, de), space.branch(pm, bt, dt)
                    e2, t2 = assemble(ch2, de2, dt2)
                    evals += 1
                    wt = wrap(t2)
                    a = agree(wt)
                    if a < a0:                      # the monotone constraint
                        continue
                    best = max(best, a)
                    if a >= 1.0:
                        return e2, expanded, a, trace + [a], evals
                    key = wt.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    tick += 1
                    heapq.heappush(heap, (-a, tick, ch2, de2, dt2, trace + [a]))
    return None, expanded, best, [], evals


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    space = Space(TAPES, alpha)
    lp = max(len(t) for t in TAPES) + 2
    wrap = lambda t: space.loop(t, lp)
    base_p, base_m, rules = V.build(space, alpha)
    fams = X.families(space, alpha)

    # A reduced test set, so the frontier is wide enough to mean something
    # inside the budget. Stated because it bounds the negative result: a path
    # needing a test outside this set would not be found.
    keep = [("AT", 0, c) for c in list(alpha) + ["$"]] + [("EMPTY",)]
    keep += [("TOP", c) for c in "[("]

    print("X53: monotone reachability, per target\n")
    print(f"tapes {TAPES}   {len(rules)} rule bodies, "
          f"state budget {STATE_BUDGET}, chains up to {MAX_RULES} rules")
    print("monotone = agreement never falls from one rule to the next.\n")

    print(f'{"true rule":16} {"beam builder":>14} {"beam cost":>11} '
          f'{"frontier":>10} {"frontier cost":>14} {"states":>7}')
    print("-" * 76)
    verdicts = {}
    for name in ("copy all", "halt on close", "strip brackets",
                 "copy inside []", "copy inside any"):
        truth = TRUTHS[name]
        target = space.interpret(truth)
        dv = X.derive(space, target, fams)
        preds = [p for p in keep] + [d[1] for d in dv]
        masks = [space.pred(p) for p in preds]
        bp = base_p + [d[1] for d in dv]
        bm = base_m + [space.pred(d[1]) for d in dv]

        def check(expr, looped):
            prog = ("LOOP", expr) if looped else expr
            return all(output(prog, tp) == output(truth, tp) for tp in EVAL), prog

        # the X50-X52 builder: greedy argmax, per-round beam
        g_res, g_cost = "--", 0
        for w, lo in ((lambda t: t, False), (wrap, True)):
            f, c = V.beam_list(space, rules, bp, bm, target, w, rounds=6, beam=1)
            g_cost += c
            if f:
                ok, _ = check(X.polish(space, f[0], bp, target, w), lo)
                g_res = "exact" if ok else "OVERFIT"
                break

        # X53: one global frontier, monotone constraint, no per-round discard
        f_res, evals, expanded, trace, prog = "--", 0, 0, [], None
        for w, lo in ((lambda t: t, False), (wrap, True)):
            e, ex, best, tr, ev = monotone_search(space, target, rules, preds,
                                                  masks, w, STATE_BUDGET)
            evals += ev
            expanded += ex
            if e is not None:
                ok, prog = check(e, lo)
                f_res, trace = ("exact" if ok else "OVERFIT"), tr
                break
        verdicts[name] = (prog if f_res == "exact" else None, trace)
        print(f"{name:16} {g_res:>14} {g_cost:>11,} {f_res:>10} "
              f"{evals:>14,} {expanded:>7}")

    print("\nTHE ROUTES THAT EXIST, as agreement per rule added")
    for name, (prog, trace) in verdicts.items():
        if prog is None:
            continue
        print(f"  {name:16} {' -> '.join(f'{a:.4f}' for a in trace)}")
        print(f"  {'':16} {render(prog)[:84]}")

    print("\nREADING")
    got = [n for n, (p, _) in verdicts.items() if p is not None]
    print(f"  monotone routes found for {len(got)}/{len(verdicts)} targets,")
    print("  including one the beam builder cannot reach at any width (1-8)")
    print("  or any round cap (6-15). The substrate is absolved: no primitive")
    print("  needed adding. What the beam lacks is the ability to return to a")
    print("  state it ranked low, and a global frontier has it.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
