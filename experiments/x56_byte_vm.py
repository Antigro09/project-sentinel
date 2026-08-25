"""X56: the byte VM on real text, and the quotient that makes it affordable.

The plan for this step said the substrate would scale to a 256-byte alphabet
"with zero architectural performance penalties". That is true of the
DERIVATION engine, which reads predicates off events and never touches the
lattice. It is false of the stack machine, and measuring before building was
the difference between an experiment and an out-of-memory error:

    tapes  depth  stack alphabet   situations   MB per behaviour
    toy        2  full (8)              1,971               0.20
    real       2  full (32)            62,363              14.47
    real       2  quotient                            0.07-0.21

The situation space is tapes x positions x |stack alphabet|^depth. The cost
is primarily COMPUTE: every composition touches 14.47 MB instead of 0.20, so
each candidate is ~70x more memory traffic, and the successful bracket run's
18,281 evaluations would move hundreds of terabytes instead of finishing in
15 seconds. Memory is a secondary risk -- the frontier heap holds a table per
state, so ~10^4 states would approach this machine's 128 GB -- but calling it
a certain out-of-memory failure, as an earlier draft of this file did, was an
overstatement. It is a slowdown first and a ceiling second.

THE QUOTIENT, AND IT IS EXACT. A program inspects the stack only through the
TOP(c) tests in its vocabulary, so two stacks differing only in symbols that
have no TOP test are indistinguishable to every program over that vocabulary.
Collapsing all non-marker bytes to one symbol is therefore lossless, not an
approximation -- and the table/interpreter check confirms it on all three
tasks rather than taking the argument's word for it. 35x to 100x smaller.

WHICH BYTES ARE MARKERS IS DERIVED. A byte the target never emits AND that
sits on a boundary between copied and skipped text is a structural candidate:
delimiters get consumed, payload gets copied. "Never emitted" alone is far too
generous on real text -- most bytes are outside the payload on most tapes --
and the boundary test is what isolates a delimiter. It returns '#' for the
comment task, '"' for the quoting task, and '([]' for the bracket task. No
rule in this file names any of them.

MEASURED, on five real snippets of JSON, code and comments, verified against
seven held-out ones:

    task               nodes    result      evals  states  held-out
    strip comment          7     exact        181       1        ok
    capture quoted        17     exact     10,027       4     FAILS
    capture brackets      21     exact     18,281       6        ok

    strip comment     (LOOP (IF '#'@0 NOP (SEQ EMIT ADV)))
    capture brackets  (LOOP (IF '('@0 (SEQ ADV PUSH) (IF ']'@0 (SEQ ADV POP) ...

Two of three real-text tasks recovered and generalise, including nested
bracket capture on actual source-code fragments.

WHAT CONJUNCTIONS COST. A decision list is right-nested, so it cannot put a
branch in a then-arm -- and `capture quoted` is written as
IF(quote, IF(empty, push, pop), ...). The rules it needs are conjunctions:
"a quote and the stack is empty" opens, "a quote and it is not" closes. Those
come from a PRODUCT of two partitions, byte-here with stack-state, which X49
established for characters and depth and which carries here unchanged. The
product is restricted to marker bytes, keeping it a handful of tests rather
than the full grid.

THE FAILURE, WITH THE SAME NAMED CAUSE AS X50. `capture quoted` recovers a
program exact on the evidence whose emit rule is an ENUMERATION of the
(byte, stack state) pairs actually observed. Held-out, x, y, w, h, i and s had
never appeared inside quotes, so they are not emitted: got 'a}' for 'axy'.
X49's polish pass is wired in and does not fix it, for the reason X50 already
recorded -- the general test TOP'"' would also emit a closing quote unless the
closing rule sits ahead of it, and that rule ended up in the DEFAULT, which
polish can reorder around but cannot promote into a chain rule. Depth and
structure generalise; alphabet still does not, and the cause is a missing
capability in the simplifier rather than anything new.

WHAT THIS DOES NOT SHOW. Five tapes of at most eleven bytes, twenty-six
observed of 256 possible, stacks bounded at depth 2 for the search. This says
the machine and the derivation survive contact with real text at small scale.
It does not say they survive a file.
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as S50
import x54_frontier_ranker as R

OTHER = "\\x00"
ACTIONS = ("NOP", "ADV", "EMIT", "HALT", "PUSH", "POP")


@dataclass(frozen=True, slots=True)
class St:
    pos: int
    stack: tuple = ()
    out: tuple = ()
    live: bool = True


def test_pred(p, tape, st) -> bool:
    k = p[0]
    if k == "OR":
        return test_pred(p[1], tape, st) or test_pred(p[2], tape, st)
    if k == "BOTH":
        return test_pred(p[1], tape, st) and test_pred(p[2], tape, st)
    if k == "EMPTY":
        return not st.stack
    if k == "TOP":
        return bool(st.stack) and st.stack[-1] == p[1]
    _, off, ch = p
    i = st.pos + off
    if ch == "$":
        return not (0 <= i < len(tape))
    return 0 <= i < len(tape) and tape[i] == ch


def run(expr, tape, st, markers, fuel=8192, bound=None):
    """`markers` is the quotient: any byte outside it pushes as OTHER."""
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
    if expr == "PUSH":
        if st.pos < len(tape) and (bound is None or len(st.stack) < bound):
            c = tape[st.pos]
            return replace(st, stack=st.stack + (c if c in markers else OTHER,)), fuel - 1
        return st, fuel - 1
    if expr == "POP":
        return replace(st, stack=st.stack[:-1]), fuel - 1
    h = expr[0]
    if h == "SEQ":
        st, fuel = run(expr[1], tape, st, markers, fuel, bound)
        return run(expr[2], tape, st, markers, fuel, bound)
    if h == "IF":
        return run(expr[2] if test_pred(expr[1], tape, st) else expr[3],
                   tape, st, markers, fuel - 1, bound)
    if h == "LOOP":
        for _ in range(len(tape) + 2):
            nxt, fuel = run(expr[1], tape, st, markers, fuel, bound)
            if nxt == st or fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


def size_of(e) -> int:
    if isinstance(e, str):
        return 1
    if e[0] in ("AT", "TOP", "EMPTY"):
        return 1
    return 1 + sum(size_of(p) for p in e[1:])


def render(e) -> str:
    if isinstance(e, str):
        return e
    k = e[0]
    if k == "EMPTY":
        return "EMPTY"
    if k == "TOP":
        return f"TOP{e[1]!r}"
    if k == "AT":
        return ("END" if e[2] == "$" else repr(e[2])) + f"@{e[1]:+d}"
    if k == "BOTH":
        return f"({render(e[1])}&{render(e[2])})"
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


def output(expr, tape, markers, bound=None):
    res, _ = run(expr, tape, St(pos=0), markers, bound=bound)
    return "".join(tape[p] for p in res.out), res.pos, res.live


class Space:
    def __init__(self, tapes, markers, bound=2):
        self.tapes, self.markers, self.bound = tapes, markers, bound
        alpha = list(markers) + [OTHER]
        stacks = [()]
        for k in range(1, bound + 1):
            stacks += list(itertools.product(alpha, repeat=k))
        self.stacks = stacks
        self.sits, self.index = [], {}
        for ti, tp in enumerate(tapes):
            for pos in range(len(tp) + 1):
                for stk in stacks:
                    self.index[(ti, pos, stk)] = len(self.sits)
                    self.sits.append((ti, pos, stk))
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
        for i, (ti, pos, stk) in enumerate(self.sits):
            tp = self.tapes[ti]
            p, s2 = pos, stk
            if name == "HALT":
                halt[i] = True
            elif name == "ADV":
                p = min(pos + 1, len(tp))
            elif name == "PUSH":
                if pos < len(tp) and len(stk) < self.bound:
                    c = tp[pos]
                    s2 = stk + (c if c in self.markers else OTHER,)
            elif name == "POP":
                s2 = stk[:-1]
            elif name == "EMIT" and pos < len(tp):
                cnt[i, self.base[ti] + pos] = 1
            end[i] = self.index[(ti, p, s2)]
        return self.pack(end, halt, cnt)

    def pred(self, p):
        return np.array([test_pred(p, self.tapes[ti], St(pos=pos, stack=stk))
                         for ti, pos, stk in self.sits], dtype=bool)

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
            return self.loop(self.table(e[1]), max(len(t) for t in self.tapes) + 2)
        return self.branch(self.pred(e[1]), self.table(e[2]), self.table(e[3]))

    def interpret(self, e):
        end = np.empty(self.n, dtype=np.int32)
        halt = np.zeros(self.n, dtype=bool)
        cnt = np.zeros((self.n, self.w), dtype=np.int32)
        for i, (ti, pos, stk) in enumerate(self.sits):
            res, _ = run(e, self.tapes[ti], St(pos=pos, stack=stk),
                         self.markers, bound=self.bound)
            end[i] = self.index[(ti, res.pos, res.stack)]
            halt[i] = not res.live
            for q in res.out:
                cnt[i, self.base[ti] + q] += 1
        return self.pack(end, halt, cnt)


def derive_markers(tapes, truth, k=4):
    """A marker is a byte the target never emits AND that sits on a boundary
    between emitted and non-emitted text.

    "Never emitted" alone is far too generous on real text -- most bytes are
    outside the payload on most tapes. The boundary test is what isolates a
    DELIMITER: a byte with copied text on one side and skipped text on the
    other. Quotes, brackets and comment characters satisfy it; letters do not.
    Nothing here names any of them.
    """
    every, emitted = set(), set()
    per_byte = {}
    for tape in tapes:
        res, _ = run(truth, tape, St(pos=0), set(tape))
        em = set(res.out)
        for p, c in enumerate(tape):
            every.add(c)
            if p in em:
                emitted.add(c)
            before = (p - 1) in em
            after = (p + 1) in em
            hit, tot = per_byte.get(c, (0, 0))
            per_byte[c] = (hit + int(before != after), tot + 1)
    cands = []
    for c in sorted(every - emitted):
        hit, tot = per_byte[c]
        if hit:
            cands.append((hit / tot, hit, c))
    if not cands:
        # A target that never EMITS leaves no emission boundary to read, so
        # the whole derivation comes back empty and the task is unreachable --
        # which is exactly how `halt at '#'` failed. Halting is an observable
        # event too: a byte is a marker if arriving at it stops the machine.
        for c in sorted(every - emitted):
            hit = tot = 0
            for tape in tapes:
                for pos, ch in enumerate(tape):
                    # pos 0 has no predecessor, and clamping to 0 made the
                    # first byte of every tape look halt-associated -- it
                    # derived a spurious 'p' from 'p#q"r"s', where the halt is
                    # caused by the '#' one step later.
                    if ch != c or pos == 0:
                        continue
                    tot += 1
                    res, _ = run(truth, tape, St(pos=pos - 1), set(tape))
                    hit += int(not res.live)
            if tot == 0:
                continue
            if hit:
                cands.append((hit / tot, hit, c))
        cands.sort(reverse=True)
    cands.sort(reverse=True)
    return set(c for _, _, c in cands[:k]), cands


def derive(space, target, alpha, markers):
    """Derived tests, including the PRODUCT of "what byte is here" with "what
    is on top of the stack".

    A decision list is right-nested, so it cannot put a branch in a then-arm.
    `capture quoted` is written as IF(quote, IF(empty, push, pop), ...) and has
    no flat form over plain tests -- the rules it needs are CONJUNCTIONS,
    "a quote and the stack is empty". A product of two partitions is a
    partition, so those conjunctions derive exactly and in one pass, and the
    list goes flat. X49 found this for characters and depth; the same argument
    carries to characters and stack state.
    """
    end, halted, cnt = space.unpack(target)
    own = np.zeros(space.n, dtype=bool)
    for i, (ti, pos, _) in enumerate(space.sits):
        if pos < len(space.tapes[ti]):
            own[i] = cnt[i, space.base[ti] + pos] > 0
    chars = [("AT", 0, c) for c in list(alpha) + ["$"]]
    stackst = [("EMPTY",)] + [("TOP", m) for m in sorted(markers)] + \
        [("TOP", OTHER)]
    fams = {
        "char": [(p, space.pred(p)) for p in chars],
        "stack": [(p, space.pred(p)) for p in stackst],
        "char x stack": [(("BOTH", a, b), space.pred(("BOTH", a, b)))
                         for a in chars for b in stackst],
    }
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


OPEN = ("OR", ("AT", 0, "("), ("AT", 0, "["))
CLOSE = ("OR", ("AT", 0, ")"), ("AT", 0, "]"))
QUOTE = ("AT", 0, '"')
HASH = ("AT", 0, "#")
PUSH_ON, POP_ON, EMIT_ON = ("SEQ", "PUSH", "ADV"), ("SEQ", "POP", "ADV"), \
    ("SEQ", "EMIT", "ADV")

# Five tapes, not three. Three left `capture quoted` matching the evidence
# and failing held-out; the quotient makes the space small enough that more
# evidence is nearly free, which is the point of the quotient.
TAPES = ['{"k":"ab"}', 'f(a[i])+"z"', 'y=1 # "no"', 'a["b"]#c', 'p="q"+r']
EVAL = ['{"a":"xy"}', 'g(b[j])-"w"', 'z=2 # "hi"', '["p","q"]', 'm(n)#c',
        '"s"+t', 'q=[r]#s']

TASKS = {
    # flat: copy until a marker
    "strip comment":    ("LOOP", ("IF", HASH, "NOP", EMIT_ON)),
    # a toggle: the SAME byte opens and closes, so the stack is the only
    # thing that can tell which one this is
    "capture quoted":   ("LOOP", ("IF", QUOTE,
                                  ("IF", ("EMPTY",), PUSH_ON, POP_ON),
                                  ("IF", ("EMPTY",), "ADV", EMIT_ON))),
    # a counter: nesting
    "capture brackets": ("LOOP", ("IF", OPEN, PUSH_ON,
                                  ("IF", CLOSE, POP_ON,
                                   ("IF", ("EMPTY",), "ADV", EMIT_ON)))),
}
BUDGET = 400


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    print("X56: the byte VM on real text\n")
    print(f"tapes {TAPES}")
    print(f"{len(alpha)} distinct bytes observed: {''.join(alpha)!r}")

    rules = [(a, None) for a in ACTIONS]
    print(f'\n{"task":18} {"markers derived":>22} {"stacks":>7} '
          f'{"situations":>11} {"MB/beh":>8} {"full MB":>8}')
    spaces = {}
    for name, truth in TASKS.items():
        markers, cands = derive_markers(TAPES, truth)
        space = Space(TAPES, markers, bound=2)
        full = (len(alpha) + 1)
        full_n = sum(len(t) + 1 for t in TAPES) * (1 + full + full ** 2)
        full_mb = (2 * full_n + full_n * space.w) * 4 / 1e6
        spaces[name] = (space, markers)
        print(f"{name:18} {repr(''.join(sorted(markers))):>22} "
              f"{len(space.stacks):>7} {space.n:>11,} "
              f"{space.width*4/1e6:>8.2f} {full_mb:>8.1f}")

    print("\ntable/interpreter agreement (the quotient must be exact, not "
          "approximate):")
    for name, truth in TASKS.items():
        space, _ = spaces[name]
        ok = np.array_equal(space.table(truth), space.interpret(truth))
        print(f"  {name:18} {'ok' if ok else 'MISMATCH -- quotient is lossy'}")
        if not ok:
            return 1

    print(f'\n{"task":18} {"nodes":>5} {"result":>9} {"evals":>10} '
          f'{"states":>7} {"held-out":>9}')
    print("-" * 64)
    for name, truth in TASKS.items():
        space, markers = spaces[name]
        target = space.interpret(truth)
        preds = [("AT", 0, c) for c in alpha + ["$"]] + [("EMPTY",)]
        stackst = [("EMPTY",)] + [("TOP", m) for m in sorted(markers)] + \
            [("TOP", OTHER)]
        preds += stackst
        # Conjunctions of a MARKER byte with a stack state. The push and pop
        # rules are exactly these -- "a quote and the stack is empty" opens,
        # "a quote and it is not" closes -- and no derived event names them,
        # because pushing is not visible in emission, halting or head motion.
        # Restricting the product to markers keeps it at a handful of tests
        # rather than the full |alphabet| x |stack| grid, and markers are
        # themselves derived rather than declared.
        preds += [("BOTH", ("AT", 0, m), st)
                  for m in sorted(markers) for st in stackst]
        dv = derive(space, target, alpha, markers)
        preds += [d[1] for d in dv]
        masks, keep = [], []
        seen = set()
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
        ex = [d[0] for d in dv if d[2]]
        print(f"{'':18} derived exactly: {', '.join(ex[:4]) if ex else 'none'}")
        expr, expanded, evals, _, _ = R.frontier(
            space, target, acts, keep, masks, lambda t: space.loop(t, 13),
            BUDGET)
        if expr is None:
            print(f"{name:18} {size_of(truth):>5} {'--':>9} {evals:>10,} "
                  f"{expanded:>7} {'-':>9}")
            continue
        # X49's polish, which X56 was missing. Without it the emit rule comes
        # back as an ENUMERATION of the (byte, stack state) pairs actually
        # observed -- exact on the evidence and wrong on the next byte, which
        # is precisely how `capture quoted` failed held-out: x, y, w, h, i, s
        # had never appeared inside quotes. Shrinking each test after the
        # chain is complete is what turns the list into a rule.
        wrap = lambda t: space.loop(t, 13)
        lean = S50.polish(space, expr, keep, target, wrap)
        prog = ("LOOP", lean)
        ok = all(output(prog, tp, markers) == output(truth, tp, markers)
                 for tp in EVAL)
        print(f"{name:18} {size_of(truth):>5} {'exact':>9} {evals:>10,} "
              f"{expanded:>7} {('ok' if ok else 'FAILS'):>9}")
        print(f"{'':18} {render(prog)[:92]}")

    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
