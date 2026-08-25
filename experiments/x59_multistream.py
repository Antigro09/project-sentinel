"""X59: two read heads, one write stream, and the scratchpad that cannot be.

The proposed layout was three streams -- read-only input, write-only output,
and a read/WRITE scratchpad. Two of those are affordable and one is not, and
pricing them before building is the only reason this file exists rather than
an abandoned run:

    configuration                          situations   per behaviour
    1 read-only stream (a pointer)                 27        0.005 MB
    2 read-only streams (two pointers)            243        0.049 MB
    2 read-only + a stack                       5,103          ~1 MB
    2 read-only + read/WRITE scratchpad  36,691,771,392       7,338 GB

A read-only stream contributes a POINTER. A write-only stream contributes
emission counts, which compose additively -- X48's trick. A read/write
scratchpad contributes its CONTENTS, so the state carries |alphabet|^length
and the table representation stops existing. Seven terabytes per behaviour
with a frontier holding thousands is not a budget problem, and an eight-byte
scratchpad is already past it. The scratchpad needs state ABSTRACTION rather
than state enumeration, which is a different mechanism and is not attempted
here.

MEASURED, on three stream pairs, verified against six held-out pairs and 200
synthesised ones drawn from bytes the evidence never contained:

    task                   nodes   with SAME  held-out   no SAME
    copy stream 1              4          23        ok        ok
    copy stream 2              4          29        ok        ok
    zip both                   8       1,881     FAILS        --
    scan 2 to match            5          37        ok        --
    copy 2 until match         7          37        ok        --
    halt at match              6         149        ok        --

    recovered and generalise: 5/6      tasks that need SAME: 3/6

    scan 2 to match     (LOOP (IF SAME(+0,+0) NOP ADV2))
    copy 2 until match  (LOOP (IF SAME(+0,+0) NOP (SEQ EMIT2 ADV2)))
    halt at match       (IF SAME(+0,+1) (SEQ ADV2 HALT) ADV2)

THE NEW PRIMITIVE EARNS ITS PLACE, which is worth saying because the last one
did not. SAME asks whether the bytes under the two heads agree and names no
byte at all. Three of six tasks are unrecoverable without it -- the `no SAME`
column is the ablation, not a formality. X51 added EQTOP on the same
reasoning and it appeared in 0 of 4 recovered programs; this is what the
difference looks like when a primitive is actually load-bearing.

AND IT NEEDED OFFSETS, for the reason X58 already established. `halt at match`
is advance-then-compare, and a decision list cannot test a byte a head has not
reached, so with SAME(0,0) alone the task is not in the language. SAME(o1, o2)
recovers it as the lookahead form above. That is the second time this exact
gap has appeared; it is a property of right-nested rule chains rather than of
any one substrate.

THE REMAINING FAILURE, priced rather than waved at. `zip both` is
LOOP(SEQ(SEQ(EMIT1,ADV1), SEQ(EMIT2,ADV2))) -- a depth-2 rule BODY. The body
pool is depth-1: six atoms and the thirty SEQ pairs over them. Admitting
depth-2 bodies takes the pool from 36 to 1,332 and multiplies the frontier's
branching by ~37x, so it is a real cost rather than an oversight, and nothing
here measures whether that cost pays for itself.

A BUG THIS FILE FOUND IN AN OLDER ONE. X50's size_of listed atom names
explicitly, so SAME(o1, o2) -- an atom with integer arguments -- was recursed
into and crashed polish on its offsets. Now anything that is not a compound
counts as one node whatever its arity, which is what the function meant all
along.

WHAT THIS DOES NOT SHOW. Six tasks, three pairs of at most six bytes, no
stack. Every program here reads two streams and writes one; none reads what it
wrote, which is the whole point of the scratchpad and the whole thing that is
still out of reach.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as S50
import x54_frontier_ranker as R

ACTIONS = ("NOP", "ADV1", "ADV2", "EMIT1", "EMIT2", "HALT")


@dataclass(frozen=True, slots=True)
class St:
    p1: int
    p2: int
    out: tuple = ()
    live: bool = True


def test_pred(p, pair, st) -> bool:
    k = p[0]
    if k == "OR":
        return test_pred(p[1], pair, st) or test_pred(p[2], pair, st)
    if k == "SAME":
        # Offsets, for the same reason X58 needed AT at +1: a decision list
        # cannot test a byte a head has not reached, so `halt at match` --
        # advance, then compare -- is not in the language without them.
        a, b = pair
        i, j = st.p1 + p[1], st.p2 + p[2]
        return (0 <= i < len(a) and 0 <= j < len(b) and a[i] == b[j])
    _, s, off, ch = p                      # ("AT", stream, offset, char)
    tape, pos = (pair[0], st.p1) if s == 1 else (pair[1], st.p2)
    i = pos + off
    if ch == "$":
        return not (0 <= i < len(tape))
    return 0 <= i < len(tape) and tape[i] == ch


def run(expr, pair, st, fuel=8192):
    if fuel <= 0 or not st.live:
        return st, fuel
    if expr == "NOP":
        return st, fuel - 1
    if expr == "ADV1":
        return replace(st, p1=min(st.p1 + 1, len(pair[0]))), fuel - 1
    if expr == "ADV2":
        return replace(st, p2=min(st.p2 + 1, len(pair[1]))), fuel - 1
    if expr == "EMIT1":
        if st.p1 < len(pair[0]):
            return replace(st, out=st.out + ((1, st.p1),)), fuel - 1
        return st, fuel - 1
    if expr == "EMIT2":
        if st.p2 < len(pair[1]):
            return replace(st, out=st.out + ((2, st.p2),)), fuel - 1
        return st, fuel - 1
    if expr == "HALT":
        return replace(st, live=False), fuel - 1
    h = expr[0]
    if h == "SEQ":
        st, fuel = run(expr[1], pair, st, fuel)
        return run(expr[2], pair, st, fuel)
    if h == "IF":
        return run(expr[2] if test_pred(expr[1], pair, st) else expr[3],
                   pair, st, fuel - 1)
    if h == "LOOP":
        for _ in range(len(pair[0]) + len(pair[1]) + 2):
            nxt, fuel = run(expr[1], pair, st, fuel)
            if nxt == st or fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


def size_of(e) -> int:
    if isinstance(e, str):
        return 1
    if e[0] in ("AT", "SAME"):
        return 1
    return 1 + sum(size_of(p) for p in e[1:])


def render(e) -> str:
    if isinstance(e, str):
        return e
    k = e[0]
    if k == "SAME":
        return f"SAME({e[1]:+d},{e[2]:+d})"
    if k == "AT":
        _, s, off, ch = e
        return ("END" if ch == "$" else repr(ch)) + f"@s{s}{off:+d}"
    if k == "OR":
        return f"({render(e[1])}|{render(e[2])})"
    if k == "IF":
        return f"(IF {render(e[1])} {render(e[2])} {render(e[3])})"
    if k == "LOOP":
        return f"(LOOP {render(e[1])})"
    return f"(SEQ {render(e[1])} {render(e[2])})"


def or_chain(parts):
    t = parts[-1]
    for p in reversed(parts[:-1]):
        t = ("OR", p, t)
    return t


def output(expr, pair):
    res, _ = run(expr, pair, St(0, 0))
    return ("".join(pair[s - 1][i] for s, i in res.out), res.p1, res.p2,
            res.live)


class Space:
    """A situation is (which pair, head 1, head 2). Two pointers, no contents,
    which is exactly why this is affordable and a scratchpad is not."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.sits, self.index = [], {}
        for pi, (a, b) in enumerate(pairs):
            for p1 in range(len(a) + 1):
                for p2 in range(len(b) + 1):
                    self.index[(pi, p1, p2)] = len(self.sits)
                    self.sits.append((pi, p1, p2))
        self.n = len(self.sits)
        self.slot, off = {}, 0
        for pi, (a, b) in enumerate(pairs):
            for s, tape in ((1, a), (2, b)):
                for i in range(len(tape)):
                    self.slot[(pi, s, i)] = off
                    off += 1
        self.w = off
        self.width = 2 * self.n + self.n * self.w
        self.ident = np.arange(self.n, dtype=np.int32)
        self.atoms = {a: self._atom(a) for a in ACTIONS}

    def pack(self, end, halt, cnt):
        return np.concatenate([end, halt.astype(np.int32),
                               cnt.ravel()]).astype(np.int32)

    def unpack(self, sig):
        n = self.n
        return sig[:n], sig[n:2 * n].astype(bool), sig[2 * n:].reshape(n, self.w)

    def _atom(self, name):
        end = np.empty(self.n, dtype=np.int32)
        halt = np.zeros(self.n, dtype=bool)
        cnt = np.zeros((self.n, self.w), dtype=np.int32)
        for i, (pi, p1, p2) in enumerate(self.sits):
            a, b = self.pairs[pi]
            q1, q2 = p1, p2
            if name == "HALT":
                halt[i] = True
            elif name == "ADV1":
                q1 = min(p1 + 1, len(a))
            elif name == "ADV2":
                q2 = min(p2 + 1, len(b))
            elif name == "EMIT1" and p1 < len(a):
                cnt[i, self.slot[(pi, 1, p1)]] = 1
            elif name == "EMIT2" and p2 < len(b):
                cnt[i, self.slot[(pi, 2, p2)]] = 1
            end[i] = self.index[(pi, q1, q2)]
        return self.pack(end, halt, cnt)

    def pred(self, p):
        return np.array([test_pred(p, self.pairs[pi], St(p1, p2))
                         for pi, p1, p2 in self.sits], dtype=bool)

    def seq(self, a, b):
        ea, ha, ca = self.unpack(a)
        eb, hb, cb = self.unpack(b)
        live = ~ha
        return self.pack(np.where(live, eb[ea], ea).astype(np.int32),
                         np.where(live, hb[ea], True),
                         ca + np.where(live[:, None], cb[ea], 0))

    def branch(self, p, a, b):
        ea, ha, ca = self.unpack(a)
        eb, hb, cb = self.unpack(b)
        return self.pack(np.where(p, ea, eb).astype(np.int32),
                         np.where(p, ha, hb), np.where(p[:, None], ca, cb))

    def loop(self, a, passes):
        out = a.copy()
        for _ in range(passes - 1):
            nxt = self.seq(out, a)
            if np.array_equal(nxt, out):
                break
            out = nxt
        return out

    def table(self, e):
        if isinstance(e, str):
            return self.atoms[e]
        if e[0] == "SEQ":
            return self.seq(self.table(e[1]), self.table(e[2]))
        if e[0] == "LOOP":
            k = max(len(a) + len(b) for a, b in self.pairs) + 2
            return self.loop(self.table(e[1]), k)
        return self.branch(self.pred(e[1]), self.table(e[2]), self.table(e[3]))

    def interpret(self, e):
        end = np.empty(self.n, dtype=np.int32)
        halt = np.zeros(self.n, dtype=bool)
        cnt = np.zeros((self.n, self.w), dtype=np.int32)
        for i, (pi, p1, p2) in enumerate(self.sits):
            res, _ = run(e, self.pairs[pi], St(p1, p2))
            end[i] = self.index[(pi, res.p1, res.p2)]
            halt[i] = not res.live
            for s, q in res.out:
                cnt[i, self.slot[(pi, s, q)]] += 1
        return self.pack(end, halt, cnt)


PAIRS = [("cab", "xyzcab"), ("bqp", "aabqp"), ("zmn", "qzmn")]
EVAL = [("ab", "ppab"), ("qz", "zzqz"), ("mc", "xmc"), ("p", "aap"),
        ("nb", "bnb"), ("ca", "zca")]

SAME = ("SAME", 0, 0)
SAMES = [("SAME", o1, o2) for o1 in (0, 1) for o2 in (0, 1)]
E1A1 = ("SEQ", "EMIT1", "ADV1")
E2A2 = ("SEQ", "EMIT2", "ADV2")

TASKS = {
    "copy stream 1":   ("LOOP", E1A1),
    "copy stream 2":   ("LOOP", E2A2),
    "zip both":        ("LOOP", ("SEQ", E1A1, E2A2)),
    "scan 2 to match": ("LOOP", ("IF", SAME, "NOP", "ADV2")),
    "copy 2 until match": ("LOOP", ("IF", SAME, "NOP", E2A2)),
    "halt at match":   ("SEQ", "ADV2", ("IF", SAME, "HALT", "NOP")),
}


def derive(space, target, alpha):
    end, halted, cnt = space.unpack(target)
    own1 = np.zeros(space.n, dtype=bool)
    own2 = np.zeros(space.n, dtype=bool)
    for i, (pi, p1, p2) in enumerate(space.sits):
        a, b = space.pairs[pi]
        if p1 < len(a):
            own1[i] = cnt[i, space.slot[(pi, 1, p1)]] > 0
        if p2 < len(b):
            own2[i] = cnt[i, space.slot[(pi, 2, p2)]] > 0
    fams = {}
    for s in (1, 2):
        fams[f"char@s{s}"] = [(("AT", s, 0, c), space.pred(("AT", s, 0, c)))
                              for c in alpha + ["$"]]
    fams["same"] = [(SAME, space.pred(SAME)),
                    (("AT", 1, 0, "$"), space.pred(("AT", 1, 0, "$")))]
    events = {"emitted 1 here": own1, "emitted 2 here": own2,
              "halted": halted, "head-stayed": (end == space.ident) & ~halted}
    out, seen = [], set()
    for ev, b in events.items():
        if not b.any() or b.all():
            continue
        for fname, fam in fams.items():
            parts = [a for a, m in fam if (m & b).any()]
            if not parts or len(parts) == len(fam):
                continue
            union = np.zeros(space.n, dtype=bool)
            for a, m in fam:
                if a in parts:
                    union |= m
            key = union.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((f"{ev}/{fname}", or_chain(parts),
                        bool((union == b).all())))
    return sorted(out, key=lambda r: (not r[2], size_of(r[1])))


WIDE = "defghijklorstuvw0123"
BUDGET = 400


def survives(prog, truth, rng, alpha):
    if any(output(prog, pr) != output(truth, pr) for pr in EVAL):
        return False
    wide = sorted(set(alpha) | set(WIDE))
    for _ in range(200):
        a = "".join(wide[int(i)] for i in rng.integers(0, len(wide), 3))
        b = "".join(wide[int(i)] for i in rng.integers(0, len(wide), 5))
        if output(prog, (a, b)) != output(truth, (a, b)):
            return False
    return True


def attempt(space, truth, alpha, use_same):
    target = space.interpret(truth)
    preds = [("AT", s, o, c) for s in (1, 2) for o in (0, 1)
             for c in alpha + ["$"]]
    if use_same:
        preds = list(SAMES) + preds
    preds += [d[1] for d in derive(space, target, alpha)] if use_same else []
    keep, masks, seen = [], [], set()
    for p in preds:
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        keep.append(p)
        masks.append(m)
    acts = [(a, space.atoms[a]) for a in ACTIONS]
    acts += [(("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b]))
             for a in ACTIONS for b in ACTIONS if a != b]
    k = max(len(a) + len(b) for a, b in space.pairs) + 2
    for w, looped in ((lambda t: t, False), (lambda t: space.loop(t, k), True)):
        expr, _, evals, _, _ = R.frontier(space, target, acts, keep, masks,
                                          w, BUDGET)
        if expr is not None:
            lean = S50.polish(space, expr, keep, target, w)
            return (("LOOP", lean) if looped else lean), evals
    return None, 0


def main() -> int:
    t0 = time.perf_counter()
    space = Space(PAIRS)
    alpha = sorted({c for a, b in PAIRS for c in a + b})
    print("X59: two read heads, one write stream\n")
    print(f"pairs {PAIRS}")
    print(f"{space.n} situations (pair x head1 x head2), "
          f"{space.width*4/1e6:.3f} MB per behaviour")
    print(f"a read/write scratchpad of the same length would be "
          f"36,691,771,392 situations and ~7,338 GB per behaviour.\n")
    print("primitives: NOP ADV1 ADV2 EMIT1 EMIT2 HALT  SEQ IF LOOP  "
          "AT(stream,offset,byte) SAME")
    print("SAME names no byte: it asks whether the two heads agree.\n")

    bad = [n for n, t in TASKS.items()
           if not np.array_equal(space.table(t), space.interpret(t))]
    print(f"table/interpreter agreement: {len(TASKS)-len(bad)}/{len(TASKS)}")
    if bad:
        return 1

    print(f'\n{"task":22} {"nodes":>5} {"with SAME":>11} {"held-out":>9} '
          f'{"no SAME":>9}')
    print("-" * 62)
    won = drop = 0
    for name, truth in TASKS.items():
        prog, evals = attempt(space, truth, alpha, True)
        if prog is None:
            print(f"{name:22} {size_of(truth):>5} {'--':>11}")
            continue
        ok = survives(prog, truth, np.random.default_rng(3), alpha)
        won += int(ok)
        prog0, _ = attempt(space, truth, alpha, False)
        ok0 = prog0 is not None and survives(prog0, truth,
                                             np.random.default_rng(3), alpha)
        drop += int(ok and not ok0)
        print(f"{name:22} {size_of(truth):>5} {evals:>11,} "
              f"{('ok' if ok else 'FAILS'):>9} {('ok' if ok0 else '--'):>9}")
        if ok:
            print(f"{'':22} {render(prog)[:86]}")

    n = len(TASKS)
    print(f"\n  recovered and generalise: {won}/{n}")
    print(f"  tasks that need SAME    : {drop}/{n}")
    print("\nREADING")
    if drop == 0:
        print("  SAME is never load-bearing: everything solvable with it is")
        print("  solvable without it. Like EQTOP in X51, it did not earn its")
        print("  place and should be reported as such.")
    else:
        print(f"  {drop} task(s) are recoverable only with SAME, so the")
        print("  cross-stream test is load-bearing rather than decorative --")
        print("  which is more than EQTOP managed in X51.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
