"""X55: the difficulty band exists, and it is too thin to farm.

X54 could not train a frontier ranker because there was no data, and named
the prerequisite: a curriculum of tasks where agreement ordering expands
hundreds of states and still finishes. This builds the generator and measures
whether that band can be populated.

THE SURVEY. Random decision lists of 1-7 rules, agreement ordering, 400-state
cap, 700s of generation:

    22 generated, 0 duplicates of a real target
      0 states (found in the seed layer)      2
      1-19 states                            15
      20+ states                              1
      unsolved at the cap                     4

Seventeen of twenty-two are trivial, four are out of reach, and ONE lands in
the band. The distribution is bimodal, which is what the falsifier written
before the run said would sink this: difficulty is a property of a target's
EQUIVALENCE CLASS, not of how many rules were used to write it, so a
seven-rule random program is usually behaviourally equal to a one-rule one and
resolves instantly.

THE COST THIS IMPLIES, which is the number worth carrying forward. 22 tasks
took 700s, so ~32s each, and 1 in 22 is usable. A corpus of 100 in-band tasks
therefore costs roughly 2,200 generated tasks and about 20 hours. Uniform
random generation is not a viable way to build this curriculum.

PHASE 2, AND WHAT IT CANNOT SHOW. Training on the single in-band task gives
121 states, 6 of them on-route, and the resulting ranker solves 0/3 -- the
same as random, against agreement's 3/3 at 1,007 / 7,435 / 132,545
evaluations. That is NOT evidence about the model. With a one-task corpus this
run cannot separate "the curriculum is too small" from "the ranker is too
weak", and reading the arms as a model result would be exactly the kind of
overclaim the rest of this project exists to avoid. The measurable result here
is the band's density; the arms are reported for completeness.

WHAT THIS CHANGES ABOUT THE PLAN. The curriculum idea is not refuted -- the
band is real, one task landed in it -- but uniform sampling cannot reach it at
any sensible cost. What would: generating from the in-band task rather than
from scratch, mutating a known-hard target and keeping mutations that stay
hard, so difficulty is inherited instead of rediscovered. That is a different
generator from the one proposed and built here, and it is the honest next
step rather than a bigger sampling budget.

ONE MORE THING THE TESTS TURNED UP. `copy inside any` sits in the band only
because of the derived tests: without them the frontier does not solve it at
all inside 200 expanded states. The difficulty band and the derivation engine
are not independent parts of this result, so a curriculum built by mutating
targets has to carry the derivation with it.

Three experiments now agree that the learned component has no measured role in
this substrate: X54 could not feed it, X55 cannot cheaply build the food, and
X53 showed the symbolic ordering it was meant to improve is already doing
essentially all the work.
"""

from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as X
import x51_deceptive_valley as V
import x54_frontier_ranker as R
from x47_priced_vocabulary import Logistic
from x50_stack import EVAL, TAPES, TRUTHS, Space, output, size_of

GEN_BUDGET_S = 700          # wall-clock cap on curriculum generation
STATE_CAP = 400             # per-task state cap during the survey
BAND = (20, STATE_CAP - 1)  # "expanded this many and still finished"
TEST = ("strip brackets", "copy inside []", "copy inside any")


def random_target(space, rules, preds, masks, wrap, rng):
    n = int(rng.integers(1, 8))
    chain = [[preds[int(rng.integers(0, len(preds)))],
              rules[int(rng.integers(0, len(rules)))][0]] for _ in range(n)]
    body = X._join(chain, rules[int(rng.integers(0, len(rules)))][0])
    return body, wrap(space.table(body)), n


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    space = Space(TAPES, alpha)
    lp = max(len(t) for t in TAPES) + 2
    wrap = lambda t: space.loop(t, lp)
    _, _, rules = V.build(space, alpha)
    fams = X.families(space, alpha)
    keep = [("AT", 0, c) for c in list(alpha) + ["$"]] + [("EMPTY",)]
    keep += [("TOP", c) for c in "[("]
    masks0 = [space.pred(p) for p in keep]

    print("X55: is there a difficulty band to learn from?\n")
    real = {n: space.interpret(t) for n, t in TRUTHS.items()}

    print(f"PHASE 1 -- survey. Random 1-7 rule targets, agreement ordering, "
          f"cap {STATE_CAP} states,\n  wall-clock budget {GEN_BUDGET_S}s.")
    rng = np.random.default_rng(5)
    hist, corpus, tried, dup = Counter(), [], 0, 0
    while time.perf_counter() - t0 < GEN_BUDGET_S:
        tried += 1
        body, tgt, nrules = random_target(space, rules, keep, masks0, wrap, rng)
        if any(np.array_equal(tgt, rt) for rt in real.values()):
            dup += 1
            continue
        col = []
        expr, expanded, evals, moves, _ = R.frontier(
            space, tgt, rules, keep, masks0, wrap, STATE_CAP, collect=col)
        if expr is None:
            hist["unsolved"] += 1
            continue
        if expanded == 0:
            hist["0 (found in the seed layer)"] += 1
        elif expanded < BAND[0]:
            hist[f"1-{BAND[0]-1}"] += 1
        else:
            hist[f"{BAND[0]}+"] += 1
            corpus.append((tgt, moves, col))
    print(f"  {tried} generated, {dup} discarded as duplicates of a real target")
    for k in ("0 (found in the seed layer)", f"1-{BAND[0]-1}",
              f"{BAND[0]}+", "unsolved"):
        print(f"    {k:28} {hist[k]:>4}")
    print(f"  usable curriculum tasks (in band, solved): {len(corpus)}")

    if not corpus:
        print("\n  THE BAND IS EMPTY. Every generated task is either trivial or")
        print("  unsolvable at this cap, so no curriculum can supply the data")
        print("  X54 needed. The ranker is untrainable by this route, and the")
        print("  next move belongs somewhere other than more task generation.")
        print(f"\ntotal {time.perf_counter()-t0:.0f}s")
        return 0

    print("\nPHASE 2 -- train on the band, test on the three real targets")
    xs, ys = [], []
    for tgt, moves, col in corpus:
        for tab, nr in R.replay(space, moves, rules, wrap):
            w = wrap(tab)
            xs.append(R.feats(space, tab, w, tgt, nr, float((w == tgt).mean())))
            ys.append(1.0)
        for tab, nr, a in col:
            xs.append(R.feats(space, tab, wrap(tab), tgt, nr, a))
            ys.append(0.0)
    model = Logistic().fit(np.array(xs), np.array(ys))
    print(f"  {len(xs):,} states ({int(sum(ys))} on-route) from "
          f"{len(corpus)} in-band tasks")

    print(f'\n{"true rule":16} {"agreement":>12} {"learned":>12} {"random":>12}')
    print("-" * 56)
    tally = {}
    for name in TEST:
        truth = TRUTHS[name]
        target = real[name]
        dv = X.derive(space, target, fams)
        preds = list(keep) + [d[1] for d in dv]
        masks = [space.pred(p) for p in preds]
        cells = []
        for arm in ("agreement", "learned", "random"):
            if arm == "agreement":
                pri = None
            elif arm == "learned":
                pri = lambda t, wt, nr, a: model.score(
                    R.feats(space, t, wt, target, nr, a))
            else:
                pri = lambda t, wt, nr, a: float(rng.random())
            expr, expanded, evals, _, _ = R.frontier(
                space, target, rules, preds, masks, wrap, STATE_CAP,
                priority=pri)
            ok = expr is not None and all(
                output(("LOOP", expr), tp) == output(truth, tp) for tp in EVAL)
            cells.append(f"{(f'{evals:,}' if ok else '--'):>12}")
            tally.setdefault(arm, []).append(evals if ok else None)
        print(f"{name:16} " + " ".join(cells))

    print("\nREADING")
    solved = {a: sum(1 for v in r if v is not None) for a, r in tally.items()}
    print("  solved: " + ", ".join(f"{a} {solved[a]}/{len(TEST)}" for a in tally))
    pairs = [(a, b) for a, b in zip(tally["agreement"], tally["learned"])
             if a and b]
    if len(corpus) < 10:
        print(f"  Only {len(corpus)} in-band task(s) were found, so this run")
        print("  CANNOT separate 'the curriculum is too small' from 'the model")
        print("  is too weak'. The measurable result is the band's density,")
        print("  not the ranker's score. Do not read the arms as a model result.")
    elif solved["learned"] <= solved["random"]:
        print("  the learned ordering still does not beat random. The band")
        print("  produced data and the data did not help; the limit is the")
        print("  features or the model, not the curriculum.")
    elif pairs and np.mean([a / b for a, b in pairs]) > 1.05:
        print(f"  learned beats agreement by "
              f"{np.mean([a / b for a, b in pairs]):.2f}x on targets both solve.")
        print("  The curriculum was the missing piece X54 said it was.")
    else:
        print("  learned matches agreement without beating it. The curriculum")
        print("  supplied trainable data; it did not buy a speedup.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
