"""X57: three edits at once, because no single one preserves behaviour.

X56 left `capture quoted` exact on its evidence and wrong on held-out text.
The explanation I gave for that was wrong, so here is the measured one. After
polish the recovered chain is:

    rule 0   IF ('"' & EMPTY)          -> (SEQ PUSH ADV)
    rule 1   IF END                    -> NOP
    rule 2   IF <199-node enumeration> -> (SEQ EMIT ADV)
    default                               (SEQ ADV POP)

Nothing is "trapped in the default". The splitter flattens the entire
right-nested chain, so there is no conditional left to promote, and the
default is an ACTION. What makes the enumeration load-bearing is that the
default POPS: every byte the enumeration does not name falls through and
closes the string early. That is the held-out failure exactly -- 'axy' coming
back as 'a}'.

Generalising therefore needs THREE simultaneous edits: a general emit test, a
different default, and a new rule to do the popping the default was doing.
Polish makes one behaviour-preserving move at a time, and measured directly,
no single (position, general test) substitution preserves behaviour. It never
fixed this and could not have.

THE MECHANISM THAT WORKS is a bounded joint rewrite: for each general test and
each atomic default, insert one derived-or-small rule at the enumerated
position and check the whole chain against the evidence. Found in 208s:

    capture quoted   test -> (TOP'"'|TOP'\x00')     "the stack is not empty"
                     default -> ADV
                     +rule IF '"' -> (SEQ ADV POP)

MEASURED, on five real snippets of JSON, code and comments:

    task               before   after   unseen-byte attack
    strip comment          ok      ok   survived 300
    capture quoted      FAILS      ok   survived 300
    capture brackets       ok      ok   survived 300

3/3 on held-out text AND against 300 synthesised tapes drawn from bytes the
evidence never contained. The alphabet-generalisation failure that bit in X50
and again in X56 is closed for these tasks.

A MECHANISM THAT DID NOT WORK, kept in the file because the reason is the
useful part. The first design derived the missing rule from the DISAGREEMENT
between the candidate and the truth -- a residual is just another event set,
and the same partition machinery that derives predicates from events should
derive one from it. It finds nothing. A program runs to completion, so one
wrong step contaminates every step after it: with the emit test generalised
and no pop rule, the string never closes and the residual covers most of the
tape rather than the quote positions. A residual is a specification only when
it is LOCAL, and a whole-run residual is not. The derived candidates are still
tried first, ahead of the general ones, since they cost nothing when they do
happen to be clean.

WHAT THIS DOES NOT SHOW. Three tasks, five tapes of at most eleven bytes. The
joint rewrite is bounded at one inserted rule and one changed default, so a
program needing two new rules is still out of reach -- and nothing here says
how often that happens.
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
from x56_byte_vm import EVAL, TAPES, TASKS, OTHER, Space, output, render, size_of


def disagreement(space, sig, target):
    """Per-situation mismatch: where does this program differ from the truth?"""
    e1, h1, c1 = space.unpack(sig)
    e2, h2, c2 = space.unpack(target)
    return (e1 != e2) | (h1 != h2) | (c1 != c2).any(axis=1)


def cover_predicate(fams, bad):
    """The tightest test over one partition that covers a situation set.

    Identical in form to deriving a predicate from an event -- because a
    residual IS an event. Returns the exact ones first.
    """
    out = []
    for fname, fam in fams.items():
        parts = [a for a, m in fam if (m & bad).any()]
        if not parts or len(parts) == len(fam):
            continue
        union = np.zeros(bad.shape, dtype=bool)
        for a, m in fam:
            if a in parts:
                union |= m
        out.append((bool((union == bad).all()), size_of(B.or_chain(parts)),
                    B.or_chain(parts)))
    out.sort(key=lambda r: (not r[0], r[1]))
    return [t for _, _, t in out]


def repair(space, expr, target, wrap, preds, actions, fams, max_test=3):
    """Replace an enumerated test with a general one, then derive the rule
    that has to exist for the replacement to be correct."""
    chain, default = S50._split(expr)
    if not chain:
        return expr, "no chain"
    big = max(range(len(chain)), key=lambda i: size_of(chain[i][0]))
    if size_of(chain[big][0]) <= max_test:
        return expr, "no enumerated test"
    general = [p for p in preds if size_of(p) <= max_test]

    for g in general:
        for d in [a for a in actions if isinstance(a, str)]:
            trial = [list(r) for r in chain]
            trial[big][0] = g
            tab = wrap(space.table(S50._join(trial, d)))
            if np.array_equal(tab, target):
                return S50._join(trial, d), f"test -> {render(g)}, default -> {render(d)}"
            bad = disagreement(space, tab, target)
            if not bad.any():
                continue
            # Deriving the missing rule from the disagreement set does not
            # work, and it is worth saying why rather than deleting it: the
            # program runs to completion, so one wrong step contaminates
            # everything after it. With the emit test generalised and no pop
            # rule, the string never closes and the residual covers most of
            # the tape instead of the quote positions. A residual is only a
            # specification when it is LOCAL, and a whole-run residual is not.
            cands = cover_predicate(fams, bad)[:3]
            cands += [q for q in general if q not in cands]
            for p in cands:
                for act in actions:
                    cand = trial[:big] + [[p, act]] + trial[big:]
                    if np.array_equal(
                            wrap(space.table(S50._join(cand, d))), target):
                        return (S50._join(cand, d),
                                f"test -> {render(g)}, default -> "
                                f"{render(d)}, +rule IF {render(p)} -> "
                                f"{render(act)}")
    return expr, "no repair found"


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    print("X57: repair by disagreement\n")
    print(f'{"task":18} {"before":>9} {"after":>9}  {"repair":>0}')
    print("-" * 72)
    for name, truth in TASKS.items():
        markers, _ = B.derive_markers(TAPES, truth)
        space = Space(TAPES, markers, bound=2)
        target = space.interpret(truth)
        wrap = lambda t: space.loop(t, 13)

        stackst = [("EMPTY",)] + [("TOP", m) for m in sorted(markers)] + \
            [("TOP", OTHER)]
        preds = [("AT", 0, c) for c in alpha + ["$"]] + [("EMPTY",)] + stackst
        preds += [("BOTH", ("AT", 0, m), st)
                  for m in sorted(markers) for st in stackst]
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
        chars = [("AT", 0, c) for c in alpha + ["$"]]
        fams = {
            "char": [(p, space.pred(p)) for p in chars],
            "stack": [(p, space.pred(p)) for p in stackst],
            "char x stack": [(("BOTH", a, b), space.pred(("BOTH", a, b)))
                             for a in chars for b in stackst],
        }

        expr, _, _, _, _ = R.frontier(space, target, acts, keep, masks,
                                      wrap, 400)
        if expr is None:
            print(f"{name:18} {'--':>9} {'--':>9}")
            continue
        lean = S50.polish(space, expr, keep, target, wrap)
        ok0 = all(output(("LOOP", lean), tp, markers) ==
                  output(truth, tp, markers) for tp in EVAL)
        fixed, note = repair(space, lean, target, wrap, keep,
                             [e for e, _ in acts], fams)
        ok1 = all(output(("LOOP", fixed), tp, markers) ==
                  output(truth, tp, markers) for tp in EVAL)
        assert np.array_equal(wrap(space.table(fixed)), target), \
            "repair broke agreement on the evidence"
        # Adversarial check over bytes the evidence never contained. The
        # held-out snippets already include several, but 300 synthesised
        # tapes drawn from a wider byte range is the stronger claim.
        rng = np.random.default_rng(7)
        wide = sorted(set(alpha) | set("GHJKLMNqrstuvw0789<>/\\-_*"))
        split = None
        for _ in range(300):
            tp = "".join(wide[int(i)] for i in rng.integers(0, len(wide), 12))
            if output(("LOOP", fixed), tp, markers) != output(truth, tp, markers):
                split = tp
                break
        adv = "survived 300" if split is None else f"SPLIT by {split!r}"
        print(f"{name:18} {('ok' if ok0 else 'FAILS'):>9} "
              f"{('ok' if ok1 else 'FAILS'):>9}  {note}")
        print(f"{'':18} {render(('LOOP', fixed))[:92]}")
        print(f"{'':18} unseen-byte attack: {adv}")

    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
