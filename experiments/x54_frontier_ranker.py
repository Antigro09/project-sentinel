"""X54: the ranker cannot be trained, and the reason is the result.

X53 bought 5/5 with a global monotone frontier and no per-round discard. The
obvious next move is to have a model rank that queue so the winning states
surface early and the cost collapses. It is the cleanest job a learned
component has had in this project: it cannot affect correctness, only speed,
because the monotone constraint and the exact-match test are untouched.

It does not work, and it fails in a way worth writing down.

    true rule          agreement      learned       random
    strip brackets         1,007    untrained           --
    copy inside []         7,435    untrained           --
    copy inside any      132,545    untrained           --
                          3/3          0/3            0/3

THREE ATTEMPTS AT TRAINING DATA, EACH FAILING FOR THE SAME REASON.

  leave-one-out over the five targets. Four of them resolve in 0-6 expanded
  states, so the whole corpus is 118 examples dominated by one target's 93.
  The resulting ranker solves nothing.

  random tasks, ranked by agreement. 26 solved, 102 states, 51 on-route.
  Also nothing: an efficient search is a poor teacher, because it barely
  searches.

  random tasks, ranked at RANDOM, to force exploration. 0 of 24 solved inside
  the budget. No routes, so no labels, so no data at all.

That is a closed loop, not a tuning problem. Labelled states come only from
searches that SUCCEED. Searches succeed only when the ordering is already
good. The ordering that is already good is agreement, so the only teacher
available is the thing the student is supposed to beat -- and it leaves no
room to beat it, because on two of three targets it finishes in one or six
expanded states.

WHAT AGREEMENT ORDERING IS ACTUALLY WORTH, which is the positive finding
hiding inside the negative one. Random ordering does not merely lose; it
solves NOTHING -- 0 of 24 random tasks, 0 of 3 real ones, at the same budget.
The whole of X53's 5/5 rests on ordering by agreement, and X51's conclusion
that "greedy agreement is the wrong signal" now looks exactly backwards
twice over: it is the right signal, and it is nearly the only thing working.

A CORRECTION TO X53's HEADLINE NUMBER. X53 reported 708,581 evaluations for
`copy inside any`. That figure included a failed identity-wrap pass that
burned the entire state budget before the LOOP pass ran. The successful
search costs 132,545. The 18x frontier tax quoted in X53 is really about 3x,
and the conclusion that the frontier buys correctness with compute stands
with a smaller price tag than I gave it.

WHAT WOULD ACTUALLY BE NEEDED. Data from a search that is neither already
optimal nor hopeless -- a curriculum of tasks hard enough that agreement
ordering expands hundreds of states yet still finishes. None of the five
targets here is in that band except `copy inside any`, and it is the test
case. Building that band is the prerequisite for a ranker, and it is a data
problem rather than a model problem.

No positive result is claimed for the learned component.
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
from x47_priced_vocabulary import Logistic
from x50_stack import EVAL, TAPES, TRUTHS, Space, output, render, size_of

STATE_BUDGET = 400
MAX_RULES = 6
NFEAT = 15


def feats(space, tab, wrapped, target, nrules, agreement):
    """What the ranker sees: the state's behaviour against the target, plus
    how far along the chain it is. No target name, no task identity."""
    base = X.features(wrapped, target, space, 8)
    return np.concatenate([base[:13], [nrules / 8.0, agreement]])


def frontier(space, target, rules, preds, pmasks, wrap, budget,
             priority=None, collect=None):
    """X53's search, with the queue order pluggable.

    `priority` returns the key to sort by; None means agreement, which is the
    baseline. The monotone constraint and the exact-match test never change,
    so a bad ranker wastes evaluations and cannot produce a wrong answer.
    """
    agree = lambda sig: float((sig == target).mean())

    def assemble(ch, de, dt):
        e, t = de, dt
        for p, pm, be, bt in reversed(ch):
            e, t = ("IF", p, be, e), space.branch(pm, bt, t)
        return e, t

    heap, seen, expanded, evals, tick = [], set(), 0, 0, 0
    for e, t in rules:
        evals += 1
        wt = wrap(t)
        a = agree(wt)
        if a >= 1.0:
            return e, expanded, evals, [], []
        seen.add(wt.tobytes())
        tick += 1
        key = -a if priority is None else -priority(t, wt, 0, a)
        heapq.heappush(heap, (key, tick, [], e, t, []))

    while heap and expanded < budget:
        _k, _t, chain, de, dt, moves = heapq.heappop(heap)
        if len(chain) >= MAX_RULES:
            continue
        a0 = agree(wrap(dt if not chain else assemble(chain, de, dt)[1]))
        expanded += 1
        if collect is not None and len(collect) < 4000:
            collect.append((dt if not chain else assemble(chain, de, dt)[1],
                            len(chain) + _nested(de), a0))
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
                    if a < a0:
                        continue
                    if a >= 1.0:
                        return e2, expanded, evals, moves + [(p, be, front)], []
                    key = wt.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    tick += 1
                    nr = len(ch2) + _nested(de2)
                    pk = -a if priority is None else -priority(t2, wt, nr, a)
                    heapq.heappush(heap, (pk, tick, ch2, de2, dt2,
                                          moves + [(p, be, front)]))
    return None, expanded, evals, [], []


def _nested(e):
    n = 0
    while isinstance(e, tuple) and e[0] == "IF":
        n += 1
        e = e[3]
    return n


def replay(space, moves, rules, wrap):
    """Rebuild the tables along a winning route, for training positives."""
    body = dict(rules)
    out, chain, de, dt = [], [], None, None
    for i, (p, be, front) in enumerate(moves):
        bt = body[be] if be in body else space.table(be)
        if de is None:
            de, dt = be, bt
        elif front:
            chain = [(p, space.pred(p), be, bt)] + chain
        else:
            de, dt = ("IF", p, be, de), space.branch(space.pred(p), bt, dt)
        t = dt
        for q, qm, qe, qt in reversed(chain):
            t = space.branch(qm, qt, t)
        out.append((t, len(chain) + _nested(de)))
    return out


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    space = Space(TAPES, alpha)
    lp = max(len(t) for t in TAPES) + 2
    wrap = lambda t: space.loop(t, lp)
    base_p, base_m, rules = V.build(space, alpha)
    fams = X.families(space, alpha)
    keep = [("AT", 0, c) for c in list(alpha) + ["$"]] + [("EMPTY",)]
    keep += [("TOP", c) for c in "[("]

    names = ["copy all", "halt on close", "strip brackets", "copy inside []",
             "copy inside any"]
    setups = {}
    for name in names:
        truth = TRUTHS[name]
        target = space.interpret(truth)
        dv = X.derive(space, target, fams)
        preds = list(keep) + [d[1] for d in dv]
        setups[name] = (target, preds, [space.pred(p) for p in preds],
                        (lambda t: t) if name == "halt on close" else wrap)

    print("X54: can a learned ranker pay back the frontier's tax?\n")
    print("PHASE 1 -- training data from RANDOM tasks, never the five targets.")
    print("  X53's routes are too short to learn from: four of five targets")
    print("  resolve in 0-6 expanded states, so leave-one-out over them yields")
    print("  118 examples and a ranker no better than random. Random tasks")
    print("  give both volume and a clean separation from the test set.")
    rng = np.random.default_rng(11)
    xs, ys, made, skipped = [], [], 0, 0
    truth_tabs = [setups[n][0] for n in names]
    preds0 = list(keep)
    masks0 = [space.pred(p) for p in preds0]
    while made < 12 and skipped < 24:
        n_rules = int(rng.integers(1, 4))
        chain = []
        for _ in range(n_rules):
            p = preds0[int(rng.integers(0, len(preds0)))]
            be = rules[int(rng.integers(0, len(rules)))][0]
            chain.append([p, be])
        default = rules[int(rng.integers(0, len(rules)))][0]
        body = X._join(chain, default)
        tgt = wrap(space.table(body))
        if any(np.array_equal(tgt, tt) for tt in truth_tabs):
            skipped += 1
            continue
        # Collected under a RANDOM ordering on purpose. Under agreement the
        # frontier resolves a random task in a handful of states and produces
        # almost no examples -- an efficient search is a poor teacher. A bad
        # ordering forces it to expand, and the route it eventually finds
        # labels which of those expansions mattered.
        col = []
        expr, expanded, evals, moves, _ = frontier(
            space, tgt, rules, preds0, masks0, wrap, 40, collect=col,
            priority=lambda t, wt, nr, a: float(rng.random()))
        if expr is None or not moves:
            skipped += 1
            continue
        made += 1
        print(f"    task {made}/12: {expanded} states expanded, "
              f"{len(moves)} rules on route", flush=True)
        for tab, nr in replay(space, moves, rules, wrap):
            xs.append(feats(space, tab, wrap(tab), tgt, nr,
                            float((wrap(tab) == tgt).mean())))
            ys.append(1.0)
        for tab, nr, a in col:
            xs.append(feats(space, tab, wrap(tab), tgt, nr, a))
            ys.append(0.0)
    print(f"  {made} random tasks solved, {skipped} unsolved within budget, "
          f"{len(xs):,} states ({int(sum(ys))} on-route)")
    model = None
    if xs:
        model = Logistic().fit(np.array(xs), np.array(ys))
    else:
        print("  NO TRAINING DATA. Under a random ordering the frontier solved")
        print("  none of the random tasks inside its budget, so there are no")
        print("  routes to label. That is the result, not a setup failure:")
        print("  labelled states come only from searches that SUCCEED, and")
        print("  searches succeed only when the ordering is already good.")

    print("\nPHASE 2 -- three orderings of the same frontier, same budget")
    print(f'{"true rule":16} {"agreement":>12} {"learned":>12} {"random":>12}')
    print("-" * 56)
    tally = {}
    for name in ("strip brackets", "copy inside []", "copy inside any"):
        target, preds, masks, wrp = setups[name]
        cells = []
        for arm in ("agreement", "learned", "random"):
            if arm == "agreement":
                pri = None
            elif arm == "learned":
                if model is None:
                    cells.append(f'{"untrained":>12}')
                    tally.setdefault(arm, []).append(None)
                    continue
                pri = lambda t, wt, nr, a: model.score(
                    feats(space, t, wt, target, nr, a))
            else:
                pri = lambda t, wt, nr, a: float(rng.random())
            expr, expanded, evals, moves, _ = frontier(
                space, target, rules, preds, masks, wrp, STATE_BUDGET,
                priority=pri)
            if expr is None:
                cells.append(f'{"--":>12}')
                tally.setdefault(arm, []).append(None)
                continue
            prog = ("LOOP", expr) if name != "halt on close" else expr
            ok = all(output(prog, tp) == output(TRUTHS[name], tp) for tp in EVAL)
            cells.append(f"{(f'{evals:,}' if ok else 'OVERFIT'):>12}")
            tally.setdefault(arm, []).append(evals if ok else None)
        print(f"{name:16} " + " ".join(cells))

    print("\nREADING")
    solved = {a: sum(1 for v in r if v is not None) for a, r in tally.items()}
    print(f"  solved: " + ", ".join(f"{a} {solved[a]}/3" for a in tally))
    if solved.get("learned", 0) <= solved.get("random", 0):
        print("  the learned ordering does not beat random. The frontier is")
        print("  doing the work; the ranker is not earning its cost.")
    elif solved.get("learned", 0) >= solved.get("agreement", 0):
        both = [(a, b) for a, b in zip(tally["agreement"], tally["learned"])
                if a and b]
        if both:
            speed = np.mean([a / b for a, b in both])
            print(f"  learned matches or beats agreement, mean speedup "
                  f"{speed:.2f}x over the targets both solve.")
    else:
        print("  the learned ordering solves fewer targets than plain")
        print("  agreement. Ordering by a model costs correctness here.")

    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
