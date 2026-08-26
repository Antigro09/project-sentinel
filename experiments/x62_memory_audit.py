"""X62: the memory audit -- four quantities, and three of them disagree.

X61 measured one number and found the bounded/growing split even. That
refuted "almost all parsing is bounded" and could not decide what to build,
because the decision needs four quantities and conflating them is how X58-X60
each lost an experiment to a misdiagnosis.

PRE-REGISTERED BEFORE ANY NUMBER: if a stack plus a couple of registers
covers most families compactly, the next work is tool- and code-oriented on
that machine. If associative or set-shaped memory is repeatedly out of reach,
the next mechanism is a sparse mutable store -- one executing only the keys a
candidate touches -- not a bigger substrate and not a writable tape, which
X59 priced at 7,338 GB per behaviour.

1. RESIDUAL STATE COMPLEXITY, on X61's diagonal, and what the shape implies:

    family       task                    1     2     3     4     5  growth
    streaming    strip comment           2     2     2     2     2  constant
    streaming    capture quoted          2     2     2     2     2  constant
    register     dedupe adjacent         6     6     6     6     6  constant
    register     emit matching first     6     6     6     6     6  constant
    stack        capture brackets        2     3     4     5     6  linear
    stack        balanced prefix         3     4     5     6     7  linear
    set          first occurrence only   6    16    26    31    32  converging
    set          emit if seen before     6    16    26    31    32  converging
    sequence     delayed copy            1    31    31    31    31  constant
    sequence     reverse                 6    31   156   781 3,906  exponential
    associative  substitute              1     2     8    38    58  converging

THE CLASSIFIER HAD TO READ THE TREND, NOT THE LAST PAIR. Comparing idx[-1] to
idx[-2] called the set tasks "linear": their last step is +1, which looks like
growth and is a plateau arriving. The increments are 10, 10, 5, 1 --
decelerating -- and 32 is exactly 2^|alphabet|, the number of subsets.

WHICH CHANGES THE HEADLINE. Only `reverse` grows with input LENGTH. Set and
associative memory are BOUNDED for a fixed alphabet -- 32 and 58 -- so the
machine's gap there is not capacity. It is SHAPE: a register holds a byte, and
neither a subset nor a key-value map is a byte. The bound is exponential in
alphabet size, so it is scaling the alphabet, not scaling the input, that
makes a sparse store necessary. X61's "growing working set" framing was too
coarse to see that.

2. EXPRESSIBILITY, three-way, because "not found" and "not expressible" are
different answers and X58-X60 spent three experiments confusing them. A
witness is a hand-written program; None is a claim with a counting argument:

    7/11 expressible. Not expressible:
      first occurrence only  needs 2^5 = 32 set states; one register holds 6
      emit if seen before    same
      reverse                EMIT reads the head; nothing emits FROM the stack
      substitute             needs a key->value map; a register holds one byte

3. SYNTHESIS, over 2,232 situations at 0.107 MB per behaviour:

    task                     found      evals  generalises
    strip comment              yes    282,339           ok
    capture quoted              no    976,521            -
    dedupe adjacent             no    976,521            -
    emit matching first        yes    919,514           ok
    capture brackets           yes    290,868           ok
    balanced prefix             no    976,521            -
    delayed copy               yes    290,375        FAILS

FOUR OF SEVEN EXPRESSIBLE TASKS WERE NOT FOUND, and one that was found does
not generalise -- two training tapes is thin evidence, exactly X58's finding
that the repair pass was a fallback for thin evidence rather than a
mechanism. Expressibility and findability come apart on more than half this
suite, which is the strongest argument yet for keeping them as separate
columns.

4. THE DECISION. Set 0/2 and associative 0/1 are not expressible at all, so
the pre-registered rule fires: the next mechanism is a sparse mutable store,
keyed and executed lazily. Stack and register families are 2/2 each, so the
existing machine covers them -- but 4/7 unfound says the bottleneck there is
search and shape, not memory, and that is a different repair.

A BUG THIS FILE FOUND IN ITSELF. `balanced prefix` failed held-out on
'(((z)))' and the witness was correct: PUSH was bounded in the INTERPRETER as
well as in the search abstraction, so a right program was judged wrong by a
machine that had been quietly crippled. That is X46's `STEP refused to enter
walls` a sixth time. The interpreter's stack is now unbounded and only the
table abstraction is bounded.

WHAT THIS DOES NOT SHOW. Eleven tasks, a five-byte alphabet, two training
tapes, one register, stack depth 2 in search. The alphabet size is doing a
lot of work in section 1 -- every "bounded" verdict for set and table memory
is bounded BY IT -- and a realistic alphabet moves those numbers by orders of
magnitude while leaving `reverse` exactly where it is.
"""

from __future__ import annotations

import itertools
import math
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x50_stack as S50
import x54_frontier_ranker as R

ALPHA = "ab()#"


# ---------------------------------------------------------- task families
# Written as plain functions so the measurement is about the TASK and not
# about any machine that might implement it.

def strip_comment(s):
    return s.split("#")[0]


def capture_quoted(s):
    out, inside = [], False
    for c in s:
        if c == "#":
            inside = not inside
        elif inside:
            out.append(c)
    return "".join(out)


def dedupe_adjacent(s):
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def emit_matching_first(s):
    return "".join(c for c in s[1:] if s and c == s[0])


def capture_brackets(s):
    out, depth = [], 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif depth > 0:
            out.append(c)
    return "".join(out)


def balanced_prefix(s):
    out, depth = [], 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                break
        out.append(c)
    return "".join(out)


def first_occurrence_only(s):
    seen, out = set(), []
    for c in s:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "".join(out)


def emit_if_seen_before(s):
    seen, out = set(), []
    for c in s:
        if c in seen:
            out.append(c)
        seen.add(c)
    return "".join(out)


def reverse(s):
    return s[::-1]


def delayed_copy(s):
    """Emit each byte two positions later -- sequence memory of fixed depth."""
    return s[:-2] if len(s) > 2 else ""


def substitute(s):
    """Associative: '(' k v ')' declares k->v, then every later k emits v."""
    table, out, i = {}, [], 0
    while i < len(s):
        if s[i] == "(" and i + 3 < len(s) and s[i + 3] == ")":
            table[s[i + 1]] = s[i + 2]
            i += 4
            continue
        out.append(table.get(s[i], s[i]))
        i += 1
    return "".join(out)


FAMILIES = {
    "streaming":   [("strip comment", strip_comment),
                    ("capture quoted", capture_quoted)],
    "register":    [("dedupe adjacent", dedupe_adjacent),
                    ("emit matching first", emit_matching_first)],
    "stack":       [("capture brackets", capture_brackets),
                    ("balanced prefix", balanced_prefix)],
    "set":         [("first occurrence only", first_occurrence_only),
                    ("emit if seen before", emit_if_seen_before)],
    "sequence":    [("delayed copy", delayed_copy), ("reverse", reverse)],
    "associative": [("substitute", substitute)],
}


# ------------------------------------------------- residual state complexity

def residual(f, p, suffixes):
    base = f(p)
    out = []
    for s in suffixes:
        whole = f(p + s)
        out.append(whole[len(base):] if whole.startswith(base) else None)
    return tuple(out)


def nerode_index(f, n, alpha):
    """X61's diagonal: prefix and suffix length grow together, so neither
    horizon manufactures a plateau."""
    words = ["".join(t) for k in range(n + 1)
             for t in itertools.product(alpha, repeat=k)]
    return len({residual(f, p, words) for p in words})


def classify(idx, alpha):
    """Growth shape -> memory class, read from the TREND of increments.

    Comparing only the last two values called the set tasks "linear": their
    indices run 6, 16, 26, 31, 32 and the last step is +1, which looks like
    growth and is a plateau arriving. The increments are 10, 10, 5, 1 --
    decelerating -- and 32 is exactly 2^|alphabet|, the number of subsets. A
    converging index is BOUNDED, however large the bound.
    """
    d = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]
    if d[-1] == 0:
        return ("constant", "registers")
    if d[-1] < d[-2]:
        cap = 2 ** len(alpha)
        # a set over the alphabet tops out at 2^|alphabet|; anything
        # converging ABOVE that is carrying more than membership.
        near = "set-like" if idx[-1] <= cap * 1.1 else "table-like"
        return ("converging", f"bounded, {near}")
    if d[-1] <= d[-2]:
        return ("linear", "counter or stack")
    return ("exponential", "stores the input")


# ------------------------------------------------------- the machine audited
#
# The union of everything built so far: a read head, a bounded stack (X50),
# one register (X60), and emission. The configuration is FIXED here rather
# than derived per task -- this is an audit of a machine class, so the machine
# has to be the same one for every task.

NONE = ""
ACTS = ("NOP", "ADV", "EMIT", "HALT", "PUSH", "POP", "LOAD")
DEPTH = 2


class St:
    __slots__ = ("pos", "stack", "reg", "out", "live")

    def __init__(self, pos, stack=(), reg=NONE, out=(), live=True):
        self.pos, self.stack, self.reg, self.out, self.live = \
            pos, stack, reg, out, live

    def key(self):
        return (self.pos, self.stack, self.reg, self.out, self.live)

    def copy(self, **kw):
        d = dict(pos=self.pos, stack=self.stack, reg=self.reg, out=self.out,
                 live=self.live)
        d.update(kw)
        return St(**d)


def test_pred(p, tape, st) -> bool:
    k = p[0]
    if k == "OR":
        return test_pred(p[1], tape, st) or test_pred(p[2], tape, st)
    if k == "EMPTY":
        return not st.stack
    if k == "FULL":
        return st.reg != NONE
    if k == "TOP":
        return bool(st.stack) and st.stack[-1] == p[1]
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
        return st.copy(pos=min(st.pos + 1, len(tape))), fuel - 1
    if expr == "EMIT":
        if st.pos < len(tape):
            return st.copy(out=st.out + (st.pos,)), fuel - 1
        return st, fuel - 1
    if expr == "HALT":
        return st.copy(live=False), fuel - 1
    if expr == "PUSH":
        # UNBOUNDED here. DEPTH bounds the search abstraction only. Bounding
        # the interpreter too made `balanced prefix` fail held-out on
        # '(((z)))' -- a correct program judged wrong because the machine
        # running it had been quietly crippled, which is X46's `STEP refused
        # to enter walls` all over again.
        if st.pos < len(tape):
            return st.copy(stack=st.stack + (tape[st.pos],)), fuel - 1
        return st, fuel - 1
    if expr == "POP":
        return st.copy(stack=st.stack[:-1]), fuel - 1
    if expr == "LOAD":
        if st.pos < len(tape):
            return st.copy(reg=tape[st.pos]), fuel - 1
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
            if nxt.key() == st.key() or fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


def emit(expr, tape):
    res, _ = run(expr, tape, St(0))
    return "".join(tape[i] for i in res.out)


# ------------------------------------------------------------- witnesses
#
# A hand-written program per task, or None where the machine class is claimed
# not to contain one. The witness answers EXPRESSIBILITY; search answers
# whether it can be found. X58-X60 spent three experiments conflating those.

E, A, P, O, L = "EMIT", "ADV", "PUSH", "POP", "LOAD"
EA = ("SEQ", E, A)
HASH, OPEN, CLOSE = ("AT", 0, "#"), ("AT", 0, "("), ("AT", 0, ")")
EMPTY, M0 = ("EMPTY",), ("MATCH", 0)

WITNESS = {
    "strip comment": ("LOOP", ("IF", HASH, "NOP", EA)),
    "capture quoted": ("LOOP", ("IF", HASH,
                                ("IF", EMPTY, ("SEQ", P, A), ("SEQ", O, A)),
                                ("IF", EMPTY, A, EA))),
    "dedupe adjacent": ("LOOP", ("IF", M0, A, ("SEQ", E, ("SEQ", L, A)))),
    "emit matching first": ("SEQ", L, ("SEQ", A,
                                       ("LOOP", ("IF", M0, EA, A)))),
    "capture brackets": ("LOOP", ("IF", OPEN, ("SEQ", P, A),
                                  ("IF", CLOSE, ("SEQ", O, A),
                                   ("IF", EMPTY, A, EA)))),
    "balanced prefix": ("LOOP", ("IF", CLOSE,
                                 ("IF", EMPTY, "NOP",
                                  ("SEQ", O, EA)),
                                 ("IF", OPEN, ("SEQ", P, EA), EA))),
    "delayed copy": ("LOOP", ("IF", ("AT", 2, "$"), "NOP", EA)),
    # These four are claimed NOT expressible by this machine class:
    #   first occurrence only / emit if seen before  need a SET of seen bytes
    #     (2^|alphabet| states); one register holds |alphabet|+1
    #   reverse  needs to emit FROM the stack; EMIT only reads the head
    #   substitute  needs a key->value map
    "first occurrence only": None,
    "emit if seen before": None,
    "reverse": None,
    "substitute": None,
}


class Space:
    def __init__(self, tapes, alpha):
        self.tapes, self.alpha = tapes, list(alpha)
        stacks = [()]
        for k in range(1, DEPTH + 1):
            stacks += list(itertools.product(self.alpha, repeat=k))
        regs = [NONE] + self.alpha
        self.sits, self.index = [], {}
        for ti, tp in enumerate(tapes):
            for pos in range(len(tp) + 1):
                for stk in stacks:
                    for r in regs:
                        self.index[(ti, pos, stk, r)] = len(self.sits)
                        self.sits.append((ti, pos, stk, r))
        self.n = len(self.sits)
        self.base, off = [], 0
        for tp in tapes:
            self.base.append(off)
            off += len(tp)
        self.w = off
        self.width = 2 * self.n + self.n * self.w
        self.ident = np.arange(self.n, dtype=np.int32)
        self.atoms = {a: self._atom(a) for a in ACTS}

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
        for i, (ti, pos, stk, r) in enumerate(self.sits):
            tp = self.tapes[ti]
            p, s2, r2 = pos, stk, r
            if name == "HALT":
                halt[i] = True
            elif name == "ADV":
                p = min(pos + 1, len(tp))
            elif name == "PUSH" and pos < len(tp) and len(stk) < DEPTH:
                s2 = stk + (tp[pos],)
            elif name == "POP":
                s2 = stk[:-1]
            elif name == "LOAD" and pos < len(tp):
                r2 = tp[pos]
            elif name == "EMIT" and pos < len(tp):
                cnt[i, self.base[ti] + pos] = 1
            end[i] = self.index[(ti, p, s2, r2)]
        return self.pack(end, halt, cnt)

    def pred(self, p):
        return np.array([test_pred(p, self.tapes[ti], St(pos, stk, r))
                         for ti, pos, stk, r in self.sits], dtype=bool)

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


# ---------------------------------------------------------------- the audit

TAPES = ["a#b#a", "(aa)("]
HELD_OUT = ["abc#de", "((xy))", "zz#z", "(p)(q)", "a#b#c#d", "xyzzy",
            "#a#b#c", "(((z)))"]
BUDGET, PROBE = 120, 25


def generalises(prog, f):
    """Longer inputs and bytes the evidence never contained."""
    return all(emit(prog, t) == f(t) for t in HELD_OUT)


def synthesise(space, witness, preds, masks, bodies):
    """Target is the WITNESS's behaviour -- the same convention every earlier
    experiment used. Shapes include a prologue, X60's lesson."""
    target = space.table(witness)
    k = max(len(t) for t in space.tapes) + 2
    shapes = [(lambda t: t, lambda e: e, BUDGET),
              (lambda t: space.loop(t, k), lambda e: ("LOOP", e), BUDGET)]
    for a in ACTS:
        shapes.append((
            (lambda at: (lambda t: space.seq(at, space.loop(t, k))))(
                space.atoms[a]),
            (lambda aa: (lambda e: ("SEQ", aa, ("LOOP", e))))(a), PROBE))
    total = 0
    for w, rebuild, budget in shapes:
        expr, _, evals, _, _ = R.frontier(space, target, bodies, preds, masks,
                                          w, budget)
        total += evals
        if expr is not None:
            return rebuild(S50.polish(space, expr, preds, target, w)), total
    return None, total


def main() -> int:
    t0 = time.perf_counter()
    ns = (1, 2, 3, 4, 5)
    print("X62: memory-class audit\n")
    print(f"alphabet {ALPHA!r}; machine = head + stack(depth {DEPTH}) + "
          f"1 register + emit\n")

    print("1. RESIDUAL STATE COMPLEXITY (diagonal) AND WHAT IT IMPLIES")
    head = (f'{"family":12} {"task":22} ' + " ".join(f"{n:>6}" for n in ns)
            + f'  {"bits":>5} {"growth":>12} {"memory class":>18}')
    print(head + "\n" + "-" * len(head))
    classes = {}
    for fam, tasks in FAMILIES.items():
        for name, f in tasks:
            idx = [nerode_index(f, n, ALPHA) for n in ns]
            growth, mem = classify(idx, ALPHA)
            classes[name] = (idx, growth, mem)
            bits = math.ceil(math.log2(idx[-1])) if idx[-1] > 1 else 0
            print(f"{fam:12} {name:22} " + " ".join(f"{v:>6,}" for v in idx)
                  + f"  {bits:>5} {growth:>12} {mem:>18}")

    print("\n2. EXPRESSIBILITY IN THE AUDITED MACHINE")
    print("   witness = a hand-written program; None = claimed impossible.")
    print(f'{"task":22} {"witness":>10} {"held-out":>10}  why not')
    print("-" * 68)
    expressible = {}
    for fam, tasks in FAMILIES.items():
        for name, f in tasks:
            wit = WITNESS[name]
            if wit is None:
                why = {
                    "first occurrence only":
                        f"needs 2^{len(ALPHA)}={2**len(ALPHA)} set states; "
                        f"1 register holds {len(ALPHA)+1}",
                    "emit if seen before":
                        f"needs 2^{len(ALPHA)}={2**len(ALPHA)} set states; "
                        f"1 register holds {len(ALPHA)+1}",
                    "reverse": "EMIT reads the head; nothing emits FROM the stack",
                    "substitute": "needs a key->value map; 1 register holds one byte",
                }[name]
                print(f"{name:22} {'none':>10} {'-':>10}  {why}")
                expressible[name] = False
                continue
            ok_train = all(emit(wit, t) == f(t) for t in TAPES)
            ok_held = generalises(wit, f)
            expressible[name] = ok_train and ok_held
            print(f"{name:22} {('ok' if ok_train else 'WRONG'):>10} "
                  f"{('ok' if ok_held else 'FAILS'):>10}")

    print("\n3. SYNTHESIS COST, for the tasks the machine can express")
    space = Space(TAPES, sorted(set("".join(TAPES))))
    print(f"   {space.n:,} situations, {space.width*4/1e6:.3f} MB per behaviour")
    alpha = sorted(set("".join(TAPES)))
    preds = [("AT", o, c) for o in (0, 1, 2) for c in alpha + ["$"]]
    preds += [("EMPTY",), ("FULL",), ("MATCH", 0), ("MATCH", 1)]
    preds += [("TOP", c) for c in alpha]
    keep, masks, seen = [], [], set()
    for p in preds:
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        keep.append(p)
        masks.append(m)
    bodies = [(a, space.atoms[a]) for a in ACTS]
    bodies += [(("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b]))
               for a in ACTS for b in ACTS if a != b]
    print(f"   {len(keep)} tests, {len(bodies)} rule bodies\n")
    print(f'{"task":22} {"found":>7} {"evals":>10} {"generalises":>12}')
    print("-" * 54)
    found = {}
    for fam, tasks in FAMILIES.items():
        for name, f in tasks:
            if not expressible.get(name):
                continue
            prog, evals = synthesise(space, WITNESS[name], keep, masks, bodies)
            g = prog is not None and generalises(prog, f)
            found[name] = (prog is not None, g)
            print(f"{name:22} {('yes' if prog else 'no'):>7} {evals:>10,} "
                  f"{('ok' if g else ('FAILS' if prog else '-')):>12}")

    print("\n4. THE PRE-REGISTERED DECISION")
    by_fam = {}
    for fam, tasks in FAMILIES.items():
        ok = sum(1 for n, _ in tasks if expressible.get(n))
        by_fam[fam] = (ok, len(tasks))
    for fam, (ok, tot) in by_fam.items():
        print(f"  {fam:12} expressible {ok}/{tot}")
    weak = [f for f, (ok, tot) in by_fam.items() if ok == 0]
    if weak:
        print(f"\n  Families the machine cannot express AT ALL: {', '.join(weak)}")
        print("  The pre-registered rule says the next mechanism is a SPARSE")
        print("  MUTABLE STORE -- one that executes only the keys a candidate")
        print("  touches -- not a bigger substrate and not a writable tape,")
        print(f"  which X59 priced at 7,338 GB per behaviour.")
    else:
        print("\n  Every family is expressible; the pre-registered rule says")
        print("  proceed to tool- and code-oriented work on this machine.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
