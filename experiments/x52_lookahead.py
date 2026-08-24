"""X52: scoring what a rule enables does not beat scoring what it fixes.

X51 measured a valley and measured that search width does not cross it. The
remaining proposal was to change the signal -- score a candidate by the best
agreement reachable one rule later rather than by its own. This builds that
and runs it against an equal-shaped control. It loses.

    true rule              arm      result   candidates
    copy inside any     greedy          --       19,908
    copy inside any  lookahead          --      192,708
    copy inside []      greedy       exact       17,443
    copy inside []   lookahead          --      192,708
    strip brackets      greedy       exact        2,879
    strip brackets   lookahead       exact        2,879

Ten times the cost, nothing gained on the target it was built for, and a
target that greedy solves is LOST. Re-ranking by the best reachable completion
flattens the ranking: many candidates share the same optimistic one-step
maximum, so the max operator discards the distinctions greedy was using, and
the state that led to `copy inside []`'s successful path gets displaced by
ties. An optimistic bound is not a discriminative score.

TWO EARLIER HYPOTHESES, ALSO DISCONFIRMED, both tested before being built on:

  the state/emission split. If the correct rules were setting up stack state
  that emission agreement cannot see, decomposing the signature would separate
  them. It does not -- the decoy leads on the end-state component too, 0.7605
  against the correct chain's 0.118 at one rule. No reweighting of the
  existing signal rescues it.

  the vocabulary story. The correct chain is dominated for three rounds with
  atomic tests and one round with disjunctive ones, so supplying pairwise ORs
  looked like the fix. It is not: `copy inside any` still fails and
  `copy inside []` goes from exact to OVERFIT, because a larger test
  vocabulary buys more spurious matches than real ones.

THE REFRAMING, which is what these four failures are actually worth.
`copy inside []` is recovered while its own true chain is non-monotone. So the
search was never following that chain -- it found a DIFFERENT monotone path to
a behaviourally equal program. Every measurement in X51 and X52 was aimed at
making the reference program's path climbable, and the reference program's
path is not the thing being climbed. The right question is whether the
target's EQUIVALENCE CLASS contains a program some monotone path reaches, and
nothing here measures that.

That also explains why width, vocabulary and lookahead all failed in the same
way: each makes the space bigger or the score more optimistic, and neither
changes whether a monotone route exists. `copy inside any` may simply have
none in this substrate, which would make it a property of the target rather
than a defect of the search -- testable by enumerating its equivalence class
at small sizes, which is the next thing to do and is not done here.

WHAT THIS FILE IS. Four mechanisms proposed across X51 and X52, all four
measured, all four negative, and one reframing that says why they were aimed
at the wrong object. No positive result is claimed.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as X
import x51_deceptive_valley as V
from x50_stack import EVAL, TAPES, TRUTHS, Space, output, render, size_of

ROUNDS = 6


def grow(space, rules, preds, pmasks, target, wrap, rounds, beam,
         topn=0, follow=None):
    """Grow decision lists. With `topn` > 0, re-rank the leading candidates by
    the best agreement reachable ONE rule later instead of by their own.

    The lookahead is deliberately narrow -- the `follow` set is a handful of
    tests, not the whole vocabulary -- because a full one is |preds| x |rules|
    per candidate and would cost more than the search it is meant to guide.
    A narrow lookahead that works is a result; a wide one that cannot be run
    is not.
    """
    cost = 0
    agree = lambda sig: float((sig == target).mean())

    def assemble(ch, de, dt):
        e, t = de, dt
        for p, pm, be, bt in reversed(ch):
            e, t = ("IF", p, be, e), space.branch(pm, bt, t)
        return e, t

    states = []
    for e, t in rules:
        cost += 1
        wt = wrap(t)
        if np.array_equal(wt, target):
            return [e], cost
        states.append((agree(wt), -size_of(e), [], e, t))
    states.sort(key=lambda r: (-r[0], -r[1]))
    states = states[:beam]

    for _ in range(rounds):
        pool, seen = [], set()
        for _a, _s, chain, de, dt in states:
            for p, pm in zip(preds, pmasks):
                for be, bt in rules:
                    for front in (True, False):
                        cost += 1
                        if front:
                            ch2, de2, dt2 = [(p, pm, be, bt)] + chain, de, dt
                        else:
                            ch2 = chain
                            de2, dt2 = ("IF", p, be, de), space.branch(pm, bt, dt)
                        e2, t2 = assemble(ch2, de2, dt2)
                        wt = wrap(t2)
                        if np.array_equal(wt, target):
                            return [e2], cost
                        key = wt.tobytes()
                        if key in seen:
                            continue
                        seen.add(key)
                        pool.append((agree(wt), -size_of(e2), ch2, de2, dt2, t2))
        if not pool:
            break
        pool.sort(key=lambda r: (-r[0], -r[1]))
        if topn:
            rescored = []
            for r in pool[:topn]:
                best = r[0]
                for p, pm in follow:
                    for be, bt in rules:
                        cost += 1
                        cand = space.branch(pm, bt, r[5])
                        wt = wrap(cand)
                        if np.array_equal(wt, target):
                            return [("IF", p, be, r[3]) if not r[2]
                                    else assemble([(p, pm, be, bt)] + r[2],
                                                  r[3], r[4])[0]], cost
                        best = max(best, agree(wt))
                rescored.append((best, r[1], r[2], r[3], r[4], r[5]))
            rescored.sort(key=lambda r: (-r[0], -r[1]))
            pool = rescored + pool[topn:]
        states = [tuple(r[:5]) for r in pool[:beam]]
    return [], cost


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    space = Space(TAPES, alpha)
    lp = max(len(t) for t in TAPES) + 2
    wrap = lambda t: space.loop(t, lp)
    base_p, base_m, rules = V.build(space, alpha)
    fams = X.families(space, alpha)

    print("X52: scoring what a rule enables, against scoring what it fixes\n")
    print(f"tapes {TAPES}   {len(rules)} rule bodies, {ROUNDS} rounds")

    targets = ["copy inside any", "copy inside []", "strip brackets"]
    print(f'\n{"true rule":16} {"arm":>12} {"result":>9} {"candidates":>12} '
          f'{"seconds":>8}')
    print("-" * 62)
    for name in targets:
        truth = TRUTHS[name]
        target = space.interpret(truth)
        dv = X.derive(space, target, fams)
        preds = base_p + [d[1] for d in dv]
        masks = base_m + [space.pred(d[1]) for d in dv]
        # the lookahead's follow-up set: the smallest tests plus everything
        # derived, capped -- a handful, not the vocabulary
        idx = sorted(range(len(preds)), key=lambda j: size_of(preds[j]))[:6]
        follow = [(preds[j], masks[j]) for j in idx]
        follow += [(preds[len(base_p) + k], masks[len(base_p) + k])
                   for k in range(min(2, len(dv)))]

        for arm, kw in (("greedy", {}),
                        ("lookahead", {"topn": 100, "follow": follow})):
            t1 = time.perf_counter()
            found, cost = grow(space, rules, preds, masks, target, wrap,
                               ROUNDS, beam=1, **kw)
            secs = time.perf_counter() - t1
            if not found:
                print(f"{name:16} {arm:>12} {'--':>9} {cost:>12,} {secs:>8.0f}")
                continue
            lean = X.polish(space, found[0], preds, target, wrap)
            prog = ("LOOP", lean)
            ok = all(output(prog, tp) == output(truth, tp) for tp in EVAL)
            print(f"{name:16} {arm:>12} {('exact' if ok else 'OVERFIT'):>9} "
                  f"{cost:>12,} {secs:>8.0f}")
            if ok and arm == "lookahead":
                print(f"{'':16} {render(prog)[:88]}")

    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
