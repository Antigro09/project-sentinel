"""X51: the valley is real, lookahead is not the answer, and here is why.

X50 recovered the pushdown rule, missed `copy inside any` -- the EASIER of its
two stack tasks -- on every arm, and left that unexplained. This chases it,
and almost every result is negative. They are worth more than the positive
one.

THE VALLEY IS REAL, and it is not what X50 guessed. Tracing the agreement
along each target's true chain against the best single rule available at that
moment:

    true rule         correct 1st   best decoy
    copy all               1.0000       0.8875   ok
    halt on close          1.0000       0.9658   ok
    strip brackets         0.9316       0.9558   DECOY WINS
    copy inside any        0.8985       0.9232   DECOY WINS
    copy inside []         0.9553       0.9800   DECOY WINS

The SAME decoy -- `IF END@+0 -> POP`, which does nothing but tidy the end of
the tape -- outscores the correct opening move on three of five targets. X50
blamed prepend-versus-append ordering. That was wrong: a greedy climb carrying
ONE chain is lost at step one whichever end it grows from, and X50's fix
worked by finding a different path to an equivalent program, not by escaping
the valley.

SO WIDEN THE BEAM. It does not help, and the flatness is the result:

    true rule              B=1     B=2     B=4     B=8
    copy all               18k     37k     74k    148k
    halt on close           2k      2k      2k      2k
    strip brackets         22k     41k     79k    155k
    copy inside any         --      --      --      --
    copy inside []         37k     73k    146k    292k

Cost scales linearly with width and recovery does not move at all. Adding
DIVERSITY to the beam -- best state per distinct default, the fix that worked
in X47 when a top-k was all near-identical -- changes nothing either.

WHY, QUANTIFIED. The correct chain for `copy inside any`, built from atomic
tests one rule at a time, against the 0.9232 a single rule reaches alone:

    0 rules + default emit    0.8801   BELOW
    1 rule                    0.8956   BELOW
    2 rules                   0.9027   BELOW
    3 rules                   0.9577   above
    4 rules                   0.9993   above
    5 rules                   1.0000   above

It is dominated for THREE CONSECUTIVE ROUNDS. A beam ranked by agreement would
have to hold a state below the single-rule ceiling while thousands of better
looking states compete for the same slots, three rounds running. That is not a
width-8 problem and not a width-64 problem; it is the wrong objective. By
contrast `copy inside []` is dominated for one round only, which is why it
survived at all.

The honest conclusion is that greedy agreement is the wrong signal for
compositional structure, and no amount of search width repairs a signal that
points away from the answer for three steps. What would repair it is a
different objective -- credit for rules that make FUTURE rules possible rather
than for immediate agreement -- and that is not built here.

ALPHABET GENERALISATION: STILL FAILING. X50's programs generalise over depth
and not over alphabet, and the fix attempted here did not work:

    true rule         as found   polished   rule added
    strip brackets       FAILS      FAILS        False
    copy inside []       FAILS      FAILS        False

The idea was that generality sometimes needs a rule ADDED rather than shrunk --
`copy inside []` keeps a 15-node enumeration of observed characters because the
rule handling `]` sits in the default, behind the emit rule, so the general
TOP'[' really would emit a closing bracket. Letting polish insert a rule never
found one that helped inside its budget. Reported as a failed attempt rather
than dropped.

Depth generalisation still holds for all four recovered programs on an
unbounded stack.

EQTOP, "the character here equals whatever is on top of the stack", was added
because it binds a variable instead of naming a letter -- the vocabulary for
"pop when this matches" with no alphabet in it. It appears in 0 of 4 recovered
programs. It did not earn its place on these tasks, and saying otherwise
because it is a good idea would be exactly the sort of claim this project
exists to avoid.

WHAT THIS FILE ADDS. One explanation, quantified, for a miss X50 recorded and
could not account for; and three mechanisms tried and measured not to work --
wider beams, beam diversity, and rule insertion. The next move is an objective
that scores a rule by what it enables, not by what it immediately fixes.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as X
from x50_stack import (ACTIONS, EVAL, TAPES, TRUTHS, TYPED, Space, _join,
                       _split, or_chain, output, render, run, scan_depth,
                       size_of, St)

FRESH = ["[[qmn]]", "[(qmn)]", "[[[wz]]]", "q[w(e)r]t", "[[[[mn]]]]"]
DEEP = ["[[[[abc]]]]", "x[([[ab]])]x", "[[[[[[c]]]]]]", "[(((a)))]",
        "[[[a](b)c]]", "[([[[c]]])]"]


def beam_list(space, rules, preds, pmasks, target, wrap, rounds, beam):
    """Grow decision lists keeping the best `beam` partial chains.

    Width 1 is exactly X50's builder. Anything wider can hold a chain that
    currently looks worse than a decoy, which is the whole question.
    """
    cost = 0
    agree = lambda sig: float((sig == target).mean())
    order = sorted(range(len(preds)), key=lambda j: size_of(preds[j]))
    preds = [preds[j] for j in order]
    pmasks = [pmasks[j] for j in order]

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
                            de2 = ("IF", p, be, de)
                            dt2 = space.branch(pm, bt, dt)
                        e2, t2 = assemble(ch2, de2, dt2)
                        wt = wrap(t2)
                        if np.array_equal(wt, target):
                            return [e2], cost
                        key = wt.tobytes()
                        if key in seen:
                            continue
                        seen.add(key)
                        pool.append((agree(wt), -size_of(e2), ch2, de2, dt2))
        if not pool:
            break
        pool.sort(key=lambda r: (-r[0], -r[1]))
        # DIVERSITY, not just rank. Ranking alone fills the beam with variants
        # of one chain, and the chain that has to survive a valley is by
        # definition the one ranking is against -- the same failure X47 hit
        # when a top-k of blocks was all near-identical motion. Keep the best
        # state per distinct DEFAULT first, then fill the rest by rank.
        states, taken = [], set()
        for r in pool:
            key = str(r[3])
            if key not in taken:
                taken.add(key)
                states.append(r)
            if len(states) >= beam:
                break
        for r in pool:
            if len(states) >= beam:
                break
            if r not in states:
                states.append(r)
    return [], cost


def polish_plus(space, expr, preds, pmasks, rules, target, wrap):
    """X50's polish, and if a big test survives it, try ADDING a rule.

    X50's `copy inside []` kept a 15-node enumeration of observed characters
    because the rule handling `]` was in the default, behind the emit rule --
    so the general TOP'[' really would have emitted a closing bracket. No
    amount of shrinking or reordering removes that; the chain is missing a
    rule. Generality sometimes costs a rule rather than saving one.
    """
    best = X.polish(space, expr, preds, target, wrap)
    chain, default = _split(best)
    if not chain or max(size_of(p) for p, _ in chain) <= 3:
        return best, False
    base, tried, polished = size_of(best), 0, 0
    for pos in range(len(chain) + 1):
        for p, pm in zip(preds, pmasks):
            for be, _bt in rules:
                if tried >= 2000 or polished >= 20:
                    return best, False
                tried += 1
                trial = chain[:pos] + [[p, be]] + chain[pos:]
                cand = _join(trial, default)
                if not np.array_equal(wrap(space.table(cand)), target):
                    continue
                polished += 1
                cand = X.polish(space, cand, preds, target, wrap)
                if size_of(cand) < base:
                    return cand, True
    return best, False


def build(space, alpha, use_eqtop=True):
    preds, masks, seen = [], [], set()
    cands = [("AT", o, c) for o in (-1, 0, 1) for c in list(alpha) + ["$"]]
    cands += [("TOP", c) for c in alpha] + [("EMPTY",)]
    if use_eqtop:
        cands += [("EQTOP", o) for o in (-1, 0, 1)]
    for p in cands:
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        preds.append(p)
        masks.append(m)
    rules = [(a, space.atoms[a]) for a in ACTIONS]
    rules += [(("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b]))
              for a in ACTIONS for b in ACTIONS if a != b]
    return preds, masks, rules


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    space = Space(TAPES, alpha)
    lp = max(len(t) for t in TAPES) + 2
    wrap = lambda t: space.loop(t, lp)
    base_preds, base_masks, rules = build(space, alpha)
    preds, masks = base_preds, base_masks

    print("X51: how much lookahead does structure need?\n")
    print(f"tapes {TAPES}   alphabet {''.join(alpha)!r}")
    print(f"{len(preds)} tests (including EQTOP, which names no letter), "
          f"{len(rules)} rule bodies")

    print("\nTHE VALLEY, per target: the correct opening move against the best")
    print("decoy available at that moment.")
    print(f'{"true rule":16} {"correct 1st":>12} {"best decoy":>11}   verdict')
    for name, truth in TRUTHS.items():
        target = space.interpret(truth)
        ag = lambda s: float((s == target).mean())
        chain, default = _split(truth[1])
        first = ag(wrap(space.table(_join(chain[:1], "ADV")))) if chain else 1.0
        decoy = max(ag(wrap(space.branch(pm, bt, space.atoms["ADV"])))
                    for pm in masks for _, bt in rules)
        print(f"{name:16} {first:>12.4f} {decoy:>11.4f}   "
              f"{'DECOY WINS' if decoy > first else 'ok'}")

    print("\nRECOVERY AGAINST BEAM WIDTH (width 1 is X50's builder)")
    head = f'{"true rule":16} ' + " ".join(f"{'B='+str(b):>11}" for b in (1, 2, 4, 8))
    print(head + "\n" + "-" * len(head))
    winners = {}
    fams = X.families(space, alpha)
    for name, truth in TRUTHS.items():
        target = space.interpret(truth)
        # Per-target derived tests, exactly as X50 built them. Leaving these
        # out made every structured target unreachable at every beam width --
        # which reads as a beam result and is nothing of the kind.
        dv = X.derive(space, target, fams)
        preds = base_preds + [d[1] for d in dv]
        masks = base_masks + [space.pred(d[1]) for d in dv]
        cells = []
        for b in (1, 2, 4, 8):
            # Both wraps, identity first because it costs no loop composition.
            # X50 tried both; dropping one here made every straight-line target
            # unreachable, which looked like a beam result and was not.
            found, cost, looped = [], 0, False
            for w, lp_flag in ((lambda t: t, False), (wrap, True)):
                f, c = beam_list(space, rules, preds, masks, target, w,
                                 rounds=6, beam=b)
                cost += c
                if f:
                    found, looped = f, lp_flag
                    break
            if not found:
                cells.append(f'{"--":>11}')
                continue
            w = wrap if looped else (lambda t: t)
            lean = X.polish(space, found[0], preds, target, w)
            prog = ("LOOP", lean) if looped else lean
            ok = all(output(prog, tp) == output(truth, tp) for tp in EVAL)
            cells.append(f"{(str(int(cost/1000))+'k' if ok else 'OVERFIT'):>11}")
            if ok and name not in winners:
                winners[name] = (found[0], b, looped)
        print(f"{name:16} " + " ".join(cells))

    print("\nALPHABET GENERALISATION, before and after polish-with-insertion")
    print("  tested on the same shapes with characters never seen:")
    print(f'{"true rule":16} {"as found":>10} {"polished":>10}  {"rule added":>10}')
    finals = {}
    for name, truth in TRUTHS.items():
        if name not in winners:
            print(f"{name:16} {'--':>10} {'--':>10}")
            continue
        body, _, looped = winners[name]
        target = space.interpret(truth)
        dv = X.derive(space, target, fams)
        preds = base_preds + [d[1] for d in dv]
        masks = base_masks + [space.pred(d[1]) for d in dv]
        w = wrap if looped else (lambda t: t)
        raw = ("LOOP", body) if looped else body
        fresh_ok = lambda e: all(output(e, tp, bound=None) ==
                                 output(truth, tp, bound=None) for tp in FRESH)
        before = fresh_ok(raw)
        lean, added = polish_plus(space, body, preds, masks, rules, target, w)
        after_expr = ("LOOP", lean) if looped else lean
        after = fresh_ok(after_expr)
        finals[name] = after_expr
        print(f"{name:16} {('OK' if before else 'FAILS'):>10} "
              f"{('OK' if after else 'FAILS'):>10}  {str(added):>10}")

    print("\nDEPTH GENERALISATION of the polished programs (unbounded stack)")
    for name, truth in TRUTHS.items():
        e = finals.get(name)
        if e is None:
            continue
        wrong = [tp for tp in DEEP
                 if output(e, tp, bound=None) != output(truth, tp, bound=None)]
        print(f"  {name:16} {'OK' if not wrong else f'FAILS on {wrong[0]!r}'}")

    print("\nWHY `copy inside any` IS MISSED: the correct chain's own trace.")
    print("  Built from ATOMIC tests only, the form the builder can actually")
    print("  reach, one rule at a time. Beside it, the best agreement any")
    print("  single rule can reach from a bare default -- what it competes with.")
    truth = TRUTHS["copy inside any"]
    target = space.interpret(truth)
    ag = lambda t: float((space.loop(t, lp) == target).mean())
    atomic = [(("AT", 0, "["), X.PUSH_ON), (("AT", 0, "("), X.PUSH_ON),
              (("AT", 0, "]"), X.POP_ON), (("AT", 0, ")"), X.POP_ON),
              (("EMPTY",), "ADV")]
    ceiling = max(ag(space.branch(pm, bt, space.atoms["ADV"]))
                  for pm in base_masks for _, bt in rules)
    acc = []
    print(f'  {"chain so far":34} {"agreement":>10} {"vs best 1-rule":>15}')
    for i in range(len(atomic) + 1):
        e = _join(atomic[:i], X.EMIT_ON)
        a = ag(space.table(e))
        mark = "BELOW" if a < ceiling else "above"
        label = f"{i} rule(s) + default emit"
        print(f"  {label:34} {a:>10.4f} {mark:>15}")
    print(f"  best single rule from a bare default: {ceiling:.4f}")

    used = [name for name, e in finals.items() if "EQTOP" in render(e)]
    print(f"\nEQTOP appears in {len(used)}/{len(finals)} recovered programs"
          + (f": {used}" if used else " -- it did not earn its place here"))
    for name in (TYPED, "copy inside any"):
        if name in finals:
            print(f"\n  {name}:\n    {render(finals[name])[:150]}")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
