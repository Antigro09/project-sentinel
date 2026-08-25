"""X60: reading what you wrote is affordable; writing a TAPE is not.

X59 priced a read/write scratchpad at 36,691,771,392 situations and ~7,338 GB
per behaviour. The proposed fix was to collapse states agreeing on every
predicate the vocabulary can currently test -- X56's stack quotient
generalised to contents.

THAT COLLAPSE IS UNSOUND, and the counterexample is one operation long. With a
vocabulary that can only test SAME(head, register), on tape 'zq':

    state A   head 0 ('z'), register 'q'  -> SAME False
    state B   head 0 ('z'), register 'w'  -> SAME False   indistinguishable
    after ADV, head 1 ('q'):
    state A                               -> SAME True
    state B                               -> SAME False   diverged

Agreeing on every current predicate does not imply agreeing after an
operation, so the collapse is not a congruence. X56's quotient was exact
because a stack symbol with no TOP test can never be distinguished by any
program at any future moment; a register byte with no current match is
distinguished the instant the head moves. The two arguments look identical
and only one of them holds.

WHERE THE 7 TB ACTUALLY COMES FROM. It is |alphabet|^LENGTH. The exponent is
the length of writable memory, not the fact of writability, so a bounded
register costs |alphabet|^COUNT:

    writable tape of 5 bytes    7^5 = 16,807 memory states
    one register                              8

Storing the byte is EXACT rather than abstract -- the byte decides every test
on it -- and the space is finite because the alphabet is. 128 situations,
0.0077 MB per behaviour, against 7,338 GB. The boundary between affordable and
impossible is bounded working set, and it is sharp.

MEASURED, on three tapes with repeated bytes, against eight held-out tapes and
200 synthesised ones over bytes the evidence never contained:

    task               nodes     evals  held-out  no register
    copy all               4        15        ok           ok
    emit matches           9        54        ok           --
    skip matches           9        78        ok           --
    scan to repeat         9        26        ok           --
    copy to repeat        11        26        ok           --

    recovered and generalise: 5/5      need the register: 4/5

    emit matches    (SEQ LOAD (LOOP (IF MATCH+0 (SEQ EMIT ADV) ADV)))
    scan to repeat  (SEQ (SEQ LOAD ADV) (LOOP (IF MATCH+0 NOP ADV)))

The `no register` column is the ablation: LOAD and every test that can read
the register removed. Four of five tasks become unrecoverable, so writable
memory is load-bearing here rather than decorative -- the same standard SAME
passed in X59 and EQTOP failed in X51.

THE THIRD SHAPE GAP IN THREE EXPERIMENTS, and by now it is a pattern rather
than an incident. The frontier searches ONE decision list, optionally looped.
Every memory task here is SEQ(LOAD, LOOP(...)) -- a loop with a prologue --
which is not in that space at all, so all four scored `--` and read as search
failures. X59's `zip both` needed a depth-2 body; X58's `halt at m` needed a
lookahead test; this needs a prologue. None of the three was a search problem.
The fix is to let any body be a prologue, and probing shapes is nearly free
because a shape that fits is found immediately -- 26 to 78 evaluations here,
against a 400-state budget -- so each candidate prologue gets 40 states and
only the two plain shapes get the full one.

WHAT THIS DOES NOT SHOW. Five tasks, three tapes of at most five bytes, ONE
register. Two registers cost |alphabet|^2 and are still cheap; a working set
that grows with the input is exactly what remains impossible, and nothing here
says how much of real work needs one.
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

NONE = ""
ACTIONS = ("NOP", "ADV", "EMIT", "HALT", "LOAD")


@dataclass(frozen=True, slots=True)
class St:
    pos: int
    reg: str = NONE
    out: tuple = ()
    live: bool = True


def test_pred(p, tape, st) -> bool:
    k = p[0]
    if k == "OR":
        return test_pred(p[1], tape, st) or test_pred(p[2], tape, st)
    if k == "FULL":
        return st.reg != NONE
    if k == "MATCH":
        i = st.pos + p[1]
        return st.reg != NONE and 0 <= i < len(tape) and tape[i] == st.reg
    _, off, ch = p
    i = st.pos + off
    if ch == "$":
        return not (0 <= i < len(tape))
    return 0 <= i < len(tape) and tape[i] == ch


def run(expr, tape, st, fuel=8192):
    if fuel <= 0 or not st.live:
        return st, fuel
    if expr == "NOP":
        return st, fuel - 1
    if expr == "ADV":
        return replace(st, pos=min(st.pos + 1, len(tape))), fuel - 1
    if expr == "EMIT":
        if st.pos < len(tape):
            return replace(st, out=st.out + (st.pos,)), fuel - 1
        return st, fuel - 1
    if expr == "HALT":
        return replace(st, live=False), fuel - 1
    if expr == "LOAD":
        # Write the byte under the head into the register. This is the only
        # write, and the register is the only thing the machine can read back.
        if st.pos < len(tape):
            return replace(st, reg=tape[st.pos]), fuel - 1
        return st, fuel - 1
    h = expr[0]
    if h == "SEQ":
        st, fuel = run(expr[1], tape, st, fuel)
        return run(expr[2], tape, st, fuel)
    if h == "IF":
        return run(expr[2] if test_pred(expr[1], tape, st) else expr[3],
                   tape, st, fuel - 1)
    if h == "LOOP":
        for _ in range(len(tape) + 2):
            nxt, fuel = run(expr[1], tape, st, fuel)
            if nxt == st or fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


def size_of(e) -> int:
    if isinstance(e, str):
        return 1
    if e[0] not in ("OR", "IF", "LOOP", "SEQ"):
        return 1
    return 1 + sum(size_of(p) for p in e[1:])


def render(e) -> str:
    if isinstance(e, str):
        return e
    k = e[0]
    if k == "FULL":
        return "FULL"
    if k == "MATCH":
        return f"MATCH{e[1]:+d}"
    if k == "AT":
        return ("END" if e[2] == "$" else repr(e[2])) + f"@{e[1]:+d}"
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


def output(expr, tape):
    res, _ = run(expr, tape, St(0))
    return "".join(tape[i] for i in res.out), res.pos, res.reg, res.live


class Space:
    """Situation = (tape, head, register byte). The register holds a BYTE, so
    the state space is finite in the alphabet rather than in tape length."""

    def __init__(self, tapes, alpha):
        self.tapes, self.alpha = tapes, [NONE] + list(alpha)
        self.sits, self.index = [], {}
        for ti, tp in enumerate(tapes):
            for pos in range(len(tp) + 1):
                for reg in self.alpha:
                    self.index[(ti, pos, reg)] = len(self.sits)
                    self.sits.append((ti, pos, reg))
        self.n = len(self.sits)
        self.base, off = [], 0
        for tp in tapes:
            self.base.append(off)
            off += len(tp)
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
        for i, (ti, pos, reg) in enumerate(self.sits):
            tp = self.tapes[ti]
            p, r = pos, reg
            if name == "HALT":
                halt[i] = True
            elif name == "ADV":
                p = min(pos + 1, len(tp))
            elif name == "LOAD" and pos < len(tp):
                r = tp[pos]
            elif name == "EMIT" and pos < len(tp):
                cnt[i, self.base[ti] + pos] = 1
            end[i] = self.index[(ti, p, r)]
        return self.pack(end, halt, cnt)

    def pred(self, p):
        return np.array([test_pred(p, self.tapes[ti], St(pos, reg))
                         for ti, pos, reg in self.sits], dtype=bool)

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
            return self.loop(self.table(e[1]),
                             max(len(t) for t in self.tapes) + 2)
        return self.branch(self.pred(e[1]), self.table(e[2]), self.table(e[3]))

    def interpret(self, e):
        end = np.empty(self.n, dtype=np.int32)
        halt = np.zeros(self.n, dtype=bool)
        cnt = np.zeros((self.n, self.w), dtype=np.int32)
        for i, (ti, pos, reg) in enumerate(self.sits):
            res, _ = run(e, self.tapes[ti], St(pos, reg))
            end[i] = self.index[(ti, res.pos, res.reg)]
            halt[i] = not res.live
            for q in res.out:
                cnt[i, self.base[ti] + q] += 1
        return self.pack(end, halt, cnt)


EA = ("SEQ", "EMIT", "ADV")
MATCH = ("MATCH", 0)
TAPES = ["abcab", "bqbz", "zmzn"]
EVAL = ["qaqa", "mnm", "zbz", "cac", "bb", "ab", "abcabc", "qq"]
WIDE = "defghijklorstuvw0123"

TASKS = {
    "copy all":       ("LOOP", EA),
    "emit matches":   ("SEQ", "LOAD", ("LOOP", ("IF", MATCH, EA, "ADV"))),
    "skip matches":   ("SEQ", "LOAD", ("LOOP", ("IF", MATCH, "ADV", EA))),
    "scan to repeat": ("SEQ", "LOAD", ("SEQ", "ADV",
                                       ("LOOP", ("IF", MATCH, "NOP", "ADV")))),
    "copy to repeat": ("SEQ", "LOAD", ("SEQ", "ADV",
                                       ("LOOP", ("IF", MATCH, "NOP", EA)))),
}
BUDGET = 400


def derive(space, target, alpha):
    end, halted, cnt = space.unpack(target)
    own = np.zeros(space.n, dtype=bool)
    for i, (ti, pos, _) in enumerate(space.sits):
        if pos < len(space.tapes[ti]):
            own[i] = cnt[i, space.base[ti] + pos] > 0
    chars = [("AT", 0, c) for c in list(alpha) + ["$"]]
    fams = {"char": [(p, space.pred(p)) for p in chars],
            "match": [(MATCH, space.pred(MATCH)),
                      (("FULL",), space.pred(("FULL",)))]}
    events = {"emitted-here": own, "halted": halted,
              "head-stayed": (end == space.ident) & ~halted}
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


def survives(prog, truth, rng, alpha):
    if any(output(prog, tp) != output(truth, tp) for tp in EVAL):
        return False
    wide = sorted(set(alpha) | set(WIDE))
    for _ in range(200):
        tp = "".join(wide[int(i)] for i in rng.integers(0, len(wide), 6))
        if output(prog, tp) != output(truth, tp):
            return False
    return True


def attempt(space, truth, alpha, memory):
    """`memory` off removes LOAD and every test that can read the register --
    the ablation that says whether the register is load-bearing."""
    target = space.interpret(truth)
    acts = [a for a in ACTIONS if memory or a != "LOAD"]
    preds = [("AT", o, c) for o in (0, 1) for c in list(alpha) + ["$"]]
    if memory:
        preds = [MATCH, ("MATCH", 1), ("FULL",)] + preds
        preds += [d[1] for d in derive(space, target, alpha)]
    keep, masks, seen = [], [], set()
    for p in preds:
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        keep.append(p)
        masks.append(m)
    bodies = [(a, space.atoms[a]) for a in acts]
    bodies += [(("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b]))
               for a in acts for b in acts if a != b]
    k = max(len(t) for t in space.tapes) + 2
    # Program SHAPES, not just decision lists. The frontier searches one chain,
    # optionally looped -- and every memory task here is SEQ(LOAD, LOOP(...)),
    # a loop with a one-action PROLOGUE, which is simply not in that space. It
    # read as a search failure and was a shape gap, the same class as X59's
    # depth-2 body.
    shapes = [(lambda t: t, lambda e: e, BUDGET),
              (lambda t: space.loop(t, k), lambda e: ("LOOP", e), BUDGET)]
    # Any BODY may be the prologue, not just an atom -- `scan to repeat` is
    # SEQ(LOAD, SEQ(ADV, LOOP(...))), a two-action one. Probing shapes is
    # cheap because a shape that fits is found almost immediately: the two
    # recovered here took 54 and 78 evaluations. So each prologue gets a small
    # budget and only the two plain shapes get the full one.
    for be, bt in bodies:
        shapes.append((
            (lambda t0: (lambda t: space.seq(t0, space.loop(t, k))))(bt),
            (lambda e0: (lambda e: ("SEQ", e0, ("LOOP", e))))(be), 40))
    for w, rebuild, budget in shapes:
        expr, _, evals, _, _ = R.frontier(space, target, bodies, keep, masks,
                                          w, budget)
        if expr is not None:
            lean = S50.polish(space, expr, keep, target, w)
            return rebuild(lean), evals
    return None, 0


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    space = Space(TAPES, alpha)
    print("X60: a machine that reads what it wrote\n")
    print(f"tapes {TAPES}   alphabet {''.join(alpha)!r}")
    print(f"{space.n} situations (tape x head x register byte), "
          f"{space.width*4/1e6:.4f} MB per behaviour")
    print(f"a WRITABLE TAPE of length {max(len(t) for t in TAPES)} over this "
          f"alphabet would carry\n  {len(alpha)}^{max(len(t) for t in TAPES)} "
          f"= {len(alpha)**max(len(t) for t in TAPES):,} memory states instead "
          f"of {len(alpha)+1}.\n")

    bad = [n for n, t in TASKS.items()
           if not np.array_equal(space.table(t), space.interpret(t))]
    print(f"table/interpreter agreement: {len(TASKS)-len(bad)}/{len(TASKS)}")
    if bad:
        print(f"  MISMATCH on {bad}")
        return 1

    print(f'\n{"task":18} {"nodes":>5} {"evals":>9} {"held-out":>9} '
          f'{"no register":>12}')
    print("-" * 58)
    won = needs = 0
    for name, truth in TASKS.items():
        prog, evals = attempt(space, truth, alpha, True)
        if prog is None:
            print(f"{name:18} {size_of(truth):>5} {'--':>9}")
            continue
        ok = survives(prog, truth, np.random.default_rng(2), alpha)
        won += int(ok)
        p0, _ = attempt(space, truth, alpha, False)
        ok0 = p0 is not None and survives(p0, truth, np.random.default_rng(2),
                                          alpha)
        needs += int(ok and not ok0)
        print(f"{name:18} {size_of(truth):>5} {evals:>9,} "
              f"{('ok' if ok else 'FAILS'):>9} {('ok' if ok0 else '--'):>12}")
        if ok:
            print(f"{'':18} {render(prog)[:84]}")

    n = len(TASKS)
    print(f"\n  recovered and generalise : {won}/{n}")
    print(f"  need the register        : {needs}/{n}")
    print("\nREADING")
    if needs == 0:
        print("  the register is never load-bearing -- everything solvable with")
        print("  it is solvable without. It has not earned its place.")
    else:
        print(f"  {needs}/{n} tasks are recoverable only with a register the")
        print("  machine writes and reads back. Bounded working memory is")
        print("  affordable and load-bearing; an unbounded tape is neither.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
