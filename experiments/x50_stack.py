"""X50: what a counter cannot remember, and a stack that never sees its bound.

X49 broke the window wall with a bounded counter and called unbounded nesting
the next one. Removing the clamp is not that wall: a counter with no bound is
still a counter. The wall one place further out is knowing WHICH bracket you
are inside, and it has a certificate exactly like X49's, one term stronger.

TWO CERTIFICATES, BOTH COMPUTED FROM THE EVIDENCE. `copy inside []` emits a
character whose innermost enclosing bracket is square:

    window 'abc' at index 4 of 'x[(abc)]'  depth 2 -- do NOT emit
    window 'abc' at index 4 of 'x[[abc]]'  depth 2 -- EMIT

The first certificate (window alone) is X49's. The second adds the depth and
still collides, which rules out every counter as well: those two states differ
only in which bracket is open, and that is exactly what a counter discards.
The certificate also has to be able to stay silent, and it does -- on
`copy inside any`, which a counter CAN do, no collision exists.

Confirmed by construction too. The same machine with TOP(c) removed, so the
stack can only be asked whether it is empty -- precisely X49's power -- finds
0 hits in 55,064 candidates.

The primitives stay domain-agnostic: PUSH pushes the character under the head,
POP removes the top, TOP(c) and EMPTY read it. Nothing is named `open`,
`close`, or `match`.

MEASURED (5 targets, 262s, 4 arms):

    true rule        nodes    size  random similar   cover
    copy all             4      6k      8k      8k      8k
    halt on close        8      6k      9k      8k      9k
    strip brackets      14     39k     47k     40k     40k
    copy inside any     21      --      --      --      --
    copy inside []      21     58k     84k     75k     74k
    ------------------------------------------------------
                            4/5     4/5     4/5     4/5

UNBOUNDED, MEASURED RATHER THAN ASSERTED. The search runs over a bounded
abstraction -- stacks of depth <= 2 -- and the search tapes are checked to
never reach it, so the abstraction is exact on the evidence rather than an
approximation of it. The recovered programs are then run by an interpreter
with a real unbounded stack:

    depth 6 nesting, 3x what the abstraction allows and 3x anything seen:
        copy all OK, halt on close OK, strip brackets OK, copy inside [] OK

A program carrying no depth constant does not care how deep the input goes,
and that is now measured instead of argued.

BUT IT DOES NOT GENERALISE OVER THE ALPHABET, and separating the two axes is
the point of testing them apart:

    copy inside []   FAILS on '[[qmn]]': got '' want 'qmn'

The emit test came back as an ENUMERATION -- ('a'&TOP'[')|('b'&TOP'[')|... --
over the characters the evidence happened to contain, where TOP'[' alone
belongs. X49's polish pass cannot remove it here, because the rule that
handles `]` ended up in the DEFAULT, behind the emit rule, so the general test
really would emit a closing bracket. The enumeration is load-bearing GIVEN
THAT ORDER, and the pass reorders chain rules but cannot promote a default
into one. So: depth-general, alphabet-specific, and the gap has a named cause
rather than a shrug.

THE FIX THAT MATTERED, AND THE EXPLANATION THAT WAS WRONG FIRST. Prepend-only
list growth finds nothing here. The first explanation written for that was
that the TOP rule is a no-op before anything pushes -- a plateau. Measuring
the truth's own construction path, innermost-first, says otherwise:

    default ADV                        agreement 0.9555
    + IF TOP'['   -> (SEQ EMIT ADV)              0.9514
    + IF (']'|')') -> (SEQ POP ADV)              0.9944
    + IF ('['|'(') -> (SEQ PUSH ADV)             1.0000

The correct first step makes agreement WORSE. It is a valley, not a plateau,
and the best available prepend scores 0.9800, so the climb leaves the correct
chain at step one and never returns. No tie-break or plateau tolerance would
have helped. Letting the DEFAULT be refined as well removes the forced order,
and that single change took `copy inside []` from unreachable to recovered.

WHAT THIS DOES NOT SHOW. Every arm scores 4/5, as in X49 -- selection is again
not the lever, and this run says nothing about the proposer. `copy inside any`
is missed by all four arms despite being the EASIER of the two stack tasks,
which is a greedy-path failure and is not explained here. Five targets, eight
bytes, three tapes. And the stack alphabet drives the state space as
|alphabet|^depth, which is why the tapes reuse letters rather than reading
naturally -- a real constraint on how far this scales as written.
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x47_priced_vocabulary import Logistic, cover_k, top_k  # noqa: E402

ACTIONS = ("NOP", "ADV", "EMIT", "HALT", "PUSH", "POP")
STACK_ABSTRACTION = 2       # search-time bound; never reached on the evidence


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
    if k == "EQTOP":
        # Variable binding, not a literal: does the character at this offset
        # equal whatever is on top of the stack? No alphabet appears in it.
        i = st.pos + p[1]
        return (bool(st.stack) and 0 <= i < len(tape)
                and tape[i] == st.stack[-1])
    _, off, ch = p                                     # ("AT", offset, char)
    i = st.pos + off
    if ch == "$":
        return not (0 <= i < len(tape))
    return 0 <= i < len(tape) and tape[i] == ch


def run(expr, tape, st, fuel=8192, bound=None):
    """`bound` is the search-time abstraction. None means a real stack."""
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
        # Pushes the character under the head. It does not know what a
        # bracket is; that has to be synthesised.
        if st.pos < len(tape) and (bound is None or len(st.stack) < bound):
            return replace(st, stack=st.stack + (tape[st.pos],)), fuel - 1
        return st, fuel - 1
    if expr == "POP":
        return replace(st, stack=st.stack[:-1]), fuel - 1
    h = expr[0]
    if h == "SEQ":
        st, fuel = run(expr[1], tape, st, fuel, bound)
        return run(expr[2], tape, st, fuel, bound)
    if h == "IF":
        return run(expr[2] if test_pred(expr[1], tape, st) else expr[3],
                   tape, st, fuel - 1, bound)
    if h == "LOOP":
        for _ in range(len(tape) + 2):
            nxt, fuel = run(expr[1], tape, st, fuel, bound)
            if nxt == st or fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


COMPOUND = ("OR", "IF", "LOOP", "SEQ", "BOTH")


def size_of(e) -> int:
    if isinstance(e, str):
        return 1
    # Anything that is not a compound is an atomic test, whatever its arity.
    # Listing atom names instead meant a later file's test -- X59's
    # SAME(o1, o2) -- recursed into its integer offsets and crashed polish.
    if not isinstance(e, tuple) or e[0] not in COMPOUND:
        return 1
    return 1 + sum(size_of(p) for p in e[1:])


def render(e) -> str:
    if isinstance(e, str):
        return e
    k = e[0]
    if k == "EMPTY":
        return "EMPTY"
    if k == "EQTOP":
        return f"EQTOP@{e[1]:+d}"
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


def output(expr, tape, bound=None):
    res, _ = run(expr, tape, St(pos=0), bound=bound)
    return "".join(tape[p] for p in res.out), res.pos, res.live


# ------------------------------------------------------------- the space


class Space:
    """Situations are (tape, head position, stack contents).

    The stack is bounded HERE and only here, at depth 2, and the search tapes
    are chosen so that bound is never reached -- so the abstraction is exact
    on the evidence rather than an approximation of it. The interpreter that
    verifies the answer has no bound at all.
    """

    def __init__(self, tapes, alpha, bound=STACK_ABSTRACTION):
        self.tapes, self.bound = tapes, bound
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
                    s2 = stk + (tp[pos],)
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
                         bound=self.bound)
            end[i] = self.index[(ti, res.pos, res.stack)]
            halt[i] = not res.live
            for q in res.out:
                cnt[i, self.base[ti] + q] += 1
        return self.pack(end, halt, cnt)


# ------------------------------------------------- certificates, computed


OPENERS, CLOSERS = "[(", "])"


def scan_depth(tape, upto):
    d = 0
    for c in tape[:upto]:
        d += (c in OPENERS) - (c in CLOSERS)
    return d


def certificate(tapes, truth, use_depth):
    """Two positions a machine of the given power cannot tell apart.

    use_depth=False reproduces X49's certificate: same window, opposite
    decisions, so the window alone is not enough. use_depth=True is the new
    one: same window AND the same bracket depth, opposite decisions, so no
    COUNTER is enough either -- the two states differ only in which bracket
    is open, which is precisely what a counter throws away.
    """
    seen = {}
    for tape in tapes:
        res, _ = run(truth, tape, St(pos=0))
        emitted = set(res.out)
        for p in range(len(tape)):
            win = tuple(tape[p + d] if 0 <= p + d < len(tape) else "$"
                        for d in (-1, 0, 1))
            key = (win, scan_depth(tape, p)) if use_depth else (win,)
            seen.setdefault(key, {}).setdefault(p in emitted, []).append((tape, p))
    for key, byd in seen.items():
        if len(byd) > 1:
            return key, byd
    return None, None


# ---------------------------------------------------------- predicates


def families(space, alpha):
    """Partitions to derive from.

    Three of them, and the third is the one this experiment needs: the
    PRODUCT of "what character is under the head" with "what is on top of the
    stack". A product of partitions is a partition, so the derivation stays
    one pass and stays checkable -- and the event "emit here" for the typed
    bracket task lives exactly in that product.
    """
    out = {}
    for off in (-1, 0, 1):
        out[f"char@{off:+d}"] = [(("AT", off, c), space.pred(("AT", off, c)))
                                 for c in list(alpha) + ["$"]]
    tops = [("TOP", c) for c in alpha] + [("EMPTY",)]
    out["top"] = [(t, space.pred(t)) for t in tops]
    out["char@0 x top"] = [(("BOTH", ("AT", 0, c), t),
                            space.pred(("BOTH", ("AT", 0, c), t)))
                           for c in list(alpha) + ["$"] for t in tops]
    return out


def derive(space, target, fams):
    end, halted, cnt = space.unpack(target)
    own = np.zeros(space.n, dtype=bool)
    for i, (ti, pos, _) in enumerate(space.sits):
        if pos < len(space.tapes[ti]):
            own[i] = cnt[i, space.base[ti] + pos] > 0
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
            out.append((f"{ev}/{fname}", or_chain(parts), bool((union == b).all())))
    return sorted(out, key=lambda r: (not r[2], size_of(r[1])))


def features(v, t, space, size):
    ev, hv, cv = space.unpack(v)
    et, ht, ct = space.unpack(t)
    sv, st_ = (ev == space.ident) & ~hv, (et == space.ident) & ~ht
    moved = (~ht) & (~st_)
    bv, bt = cv > 0, ct > 0

    def frac(mask, sub):
        k = mask.sum()
        return float(sub[mask].mean()) if k else 0.0

    return np.array([
        float((ev == et).mean()), float((bv == bt).mean()),
        float((hv == ht).mean()), float(hv.mean()), float(ht.mean()),
        float(sv.mean()), float(st_.mean()),
        frac(moved, ev == et), frac(ht, hv), frac(st_, sv),
        float((bv & ~bt).mean()), float((~bv & bt).mean()), size / 16.0,
    ], dtype=np.float64)


def depth1(space, preds, pmasks, acts=ACTIONS):
    for a in acts:
        for b in acts:
            if a != b:
                yield ("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b])
    for p, pm in zip(preds, pmasks):
        for a in acts:
            for b in acts:
                if a != b:
                    yield ("IF", p, a, b), space.branch(pm, space.atoms[a],
                                                        space.atoms[b])


def _split(expr):
    chain, node = [], expr
    while isinstance(node, tuple) and node[0] == "IF":
        chain.append([node[1], node[2]])
        node = node[3]
    return chain, node


def _join(chain, default):
    e = default
    for p, b in reversed(chain):
        e = ("IF", p, b, e)
    return e


def polish(space, expr, preds, target, wrap):
    """Delete dead rules, reorder, then shrink each test. To a fixed point.

    Shrinking alone is not enough, and the recovered bracket program showed
    why: it kept a 15-node ENUMERATION of (letter AND top-of-stack) where
    TOP'[' belonged, because the rule handling `]` sat AFTER the emit rule, so
    the general test really would have emitted a closing bracket. The
    enumeration was load-bearing GIVEN THAT ORDER. Move the closing rule in
    front of the emit rule and it stops being load-bearing.

    So generality is not only about which test you pick; it is about where the
    rule sits. Both have to be searched, and only once the chain is complete.
    """
    order = sorted(preds, key=size_of)

    def matches(chain, default):
        return np.array_equal(wrap(space.table(_join(chain, default))), target)

    chain, default = _split(expr)
    if not chain:
        return expr
    changed = True
    while changed:
        changed = False
        for i in range(len(chain) - 1, -1, -1):        # dead rules
            trial = chain[:i] + chain[i + 1:]
            if matches(trial, default):
                chain, changed = trial, True
        for i in range(len(chain)):                    # reorder
            for j in range(len(chain)):
                if i == j:
                    continue
                trial = chain[:i] + chain[i + 1:]
                trial.insert(j, chain[i])
                if trial == chain or not matches(trial, default):
                    continue
                before = sum(size_of(p) for p, _ in chain)
                probe = [list(r) for r in trial]
                for k in range(len(probe)):
                    keep = probe[k][0]
                    for cand in order:
                        if size_of(cand) >= size_of(keep):
                            break
                        probe[k][0] = cand
                        if matches(probe, default):
                            keep = cand
                            break
                        probe[k][0] = keep
                if sum(size_of(p) for p, _ in probe) < before:
                    chain, changed = probe, True
                    break
            if changed:
                break
        for k in range(len(chain)):                    # shrink each test
            keep = chain[k][0]
            for cand in order:
                if size_of(cand) >= size_of(keep):
                    break
                chain[k][0] = cand
                if matches(chain, default):
                    keep, changed = cand, True
                    break
                chain[k][0] = keep
    return _join(chain, default)


def simplify_rules(space, expr, preds, target, wrap):
    return polish(space, expr, preds, target, wrap)


def decision_list(space, blocks, preds, pmasks, target, wrap, rounds=6):
    """Grow a decision list, refining EITHER end of it.

    Prepending alone fails, and the reason is measured rather than assumed --
    the first explanation written here was wrong. Building the TRUE chain
    innermost-first, which is the only order a prepend-only builder can use:

        default ADV                        agreement 0.9555
        + IF TOP'['   -> (SEQ EMIT ADV)              0.9514
        + IF (']'|')') -> (SEQ POP ADV)              0.9944
        + IF ('['|'(') -> (SEQ PUSH ADV)             1.0000

    The correct first step makes agreement WORSE, not equal: emitting on a
    populated stack is wrong until the pop rule exists to stop it running past
    the closing bracket. And a hill climb has somewhere better to go -- the
    best available prepend scores 0.9800 -- so it leaves the correct chain at
    step one and never returns. A valley, not a plateau, which is why no
    tie-break or plateau tolerance would have helped.

    Refining the DEFAULT as well removes the forced order. Push and pop can go
    on the front, where each improves immediately, and the TOP test enters at
    the back once there is a stack worth reading.
    """
    cost, agree = 0, lambda sig: float((sig == target).mean())
    order = sorted(range(len(preds)), key=lambda j: size_of(preds[j]))
    preds = [preds[j] for j in order]
    pmasks = [pmasks[j] for j in order]

    dflt_e, dflt_t, cur_a, cur_s = None, None, -1.0, 10 ** 9
    for e, t in blocks:
        cost += 1
        a = agree(wrap(t))
        if (a, -size_of(e)) > (cur_a, -cur_s):
            dflt_e, dflt_t, cur_a, cur_s = e, t, a, size_of(e)
    if dflt_e is None:
        return [], cost
    chain = []          # [(pred, mask, block_expr, block_table)], outermost first

    def assemble(ch, de, dt):
        expr, tab = de, dt
        for p, pm, be, bt in reversed(ch):
            expr, tab = ("IF", p, be, expr), space.branch(pm, bt, tab)
        return expr, tab

    cur_e, cur_t = assemble(chain, dflt_e, dflt_t)
    if np.array_equal(wrap(cur_t), target):
        return [cur_e], cost

    for _ in range(rounds):
        best = None
        for p, pm in zip(preds, pmasks):
            for be, bt in blocks:
                for front in (True, False):
                    cost += 1
                    if front:
                        ch, de, dt = [(p, pm, be, bt)] + chain, dflt_e, dflt_t
                    else:
                        ch = chain
                        de, dt = ("IF", p, be, dflt_e), space.branch(pm, bt, dflt_t)
                    e2, t2 = assemble(ch, de, dt)
                    a = agree(wrap(t2))
                    sz = size_of(p) + size_of(be)
                    if best is None or (a, -sz) > (best[0], -best[1]):
                        best = (a, sz, ch, de, dt, e2, t2)
        if best is None or best[0] <= cur_a:
            break
        cur_a, chain, dflt_e, dflt_t = best[0], best[2], best[3], best[4]
        cur_e, cur_t = best[5], best[6]
        if np.array_equal(wrap(cur_t), target):
            return [simplify_rules(space, cur_e, preds, target, wrap)], cost
    return [], cost


class _Shim:
    def __init__(self, width):
        self.n = width


def search(space, pool, preds, pmasks, target, score_fn, select_fn,
           keep, beam, budget, passes=4, acts=ACTIONS):
    cost, first, hits = 0, 0, []
    shim = _Shim(space.width)
    passes_n = max(len(t) for t in space.tapes) + 2

    scores = np.empty(len(pool))
    for i, (expr, tab) in enumerate(pool):
        cost += 1
        if np.array_equal(tab, target):
            hits.append(expr)
            first = first or cost
        scores[i] = score_fn(tab, target, space, size_of(expr))
    if hits:
        return hits, first, 1

    blocks = [pool[i] for i in select_fn(shim, pool, scores, target, keep)]
    blocks += [(a, space.atoms[a]) for a in acts]
    short = [(a, space.atoms[a]) for a in acts]
    short += [(("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b]))
              for a in acts for b in acts if a != b]
    rules = short + [b for b in blocks if b[0] not in {e for e, _ in short}]

    for wrap, tag in ((lambda t: t, None),
                      (lambda t: space.loop(t, passes_n), "LOOP")):
        found, c = decision_list(space, rules, preds, pmasks, target, wrap)
        cost += c
        if found:
            return [found[0] if tag is None else ("LOOP", found[0])], cost, 0

    best = {}
    for i, (expr, _) in enumerate(pool):
        if expr[0] == "IF" and scores[i] > best.get(expr[1], -1e18):
            best[expr[1]] = scores[i]
    o = sorted(range(len(preds)), key=lambda j: -best.get(preds[j], -1e18))
    preds, pmasks = [preds[j] for j in o], [pmasks[j] for j in o]

    share = max(1, (budget - cost) // max(1, passes - 1))
    level = blocks
    for d in range(2, passes + 1):
        nxt = []
        used = 0
        for ea, ta in level:
            for eb, tb in blocks:
                if used >= share:
                    break
                used += 1
                cost += 1
                tab = space.seq(ta, tb)
                if np.array_equal(tab, target):
                    hits.append(("SEQ", ea, eb))
                    first = first or cost
                nxt.append((("SEQ", ea, eb), tab))
        for expr, tab in list(nxt):
            cost += 1
            if np.array_equal(space.loop(tab, passes_n), target):
                hits.append(("LOOP", expr))
                first = first or cost
        if hits:
            return hits, first, d
        if cost >= budget or not nxt:
            return [], cost, 0
        s = np.array([score_fn(t, target, space, size_of(e)) for e, t in nxt])
        level = [nxt[i] for i in select_fn(shim, nxt, s, target, beam)]
    return [], cost, 0


# ----------------------------------------------------------- the truths

IS_OPEN = ("OR", ("AT", 0, "["), ("AT", 0, "("))
IS_CLOSE = ("OR", ("AT", 0, "]"), ("AT", 0, ")"))
PUSH_ON = ("SEQ", "PUSH", "ADV")
POP_ON = ("SEQ", "POP", "ADV")
EMIT_ON = ("SEQ", "EMIT", "ADV")

TRUTHS = {
    "copy all":        ("LOOP", EMIT_ON),
    "halt on close":   ("SEQ", "ADV", ("IF", IS_CLOSE, "HALT", "NOP")),
    "strip brackets":  ("LOOP", ("IF", IS_OPEN, "ADV",
                                 ("IF", IS_CLOSE, "ADV", EMIT_ON))),
    # Needs only "is the stack empty" -- a counter is enough.
    "copy inside any": ("LOOP", ("IF", IS_OPEN, PUSH_ON,
                                 ("IF", IS_CLOSE, POP_ON,
                                  ("IF", ("EMPTY",), "ADV", EMIT_ON)))),
    # Needs to know WHICH bracket is open. No counter can do this.
    "copy inside []":  ("LOOP", ("IF", IS_OPEN, PUSH_ON,
                                 ("IF", IS_CLOSE, POP_ON,
                                  ("IF", ("TOP", "["), EMIT_ON, "ADV")))),
}
TYPED = "copy inside []"

# Three tapes, one alphabet of eight bytes, nesting exactly 2. Two tapes was
# not enough: `(` appeared in a single context, so programs that keyed on that
# context matched the evidence and failed held-out. The stack STATE SPACE is
# |alphabet|^depth, so every extra byte is expensive -- these reuse letters
# deliberately rather than reading naturally.
TAPES = ["x[[abc]]", "x[(abc)]", "a(b[c]a)"]
EVAL = ["a[bc]a", "x(ab)x", "[[ab]]", "([ab])", "a[b(c)a]", "ab[c](",
        "(a[b]c)", "[a(b)c]"]
DEEP = ["[[[[abc]]]]", "x[([[ab]])]x", "[[[[[[c]]]]]]", "[(((a)))]",
        "[[[a](b)c]]", "[([[[c]]])]"]
# Same shapes, characters the evidence never contained. Depth generality and
# alphabet generality are different claims and have to be measured apart.
FRESH = ["[[qmn]]", "[(qmn)]", "[[[wz]]]", "q[w(e)r]t", "[[[[mn]]]]"]
BUDGET, KEEP, BEAM = 600_000, 60, 30


def training_set(space, pool, preds, pmasks, forbidden, n_tasks, rng, lp):
    xs, ys, dropped = [], [], 0
    npool = len(pool)
    for _ in range(n_tasks):
        i, j = rng.integers(0, npool, 2)
        (_, ta), (_, tb) = pool[int(i)], pool[int(j)]
        form = int(rng.integers(0, 3))
        if form == 0:
            task = space.seq(ta, tb)
        elif form == 1:
            task = space.loop(ta, lp)
        else:
            task = space.branch(pmasks[int(rng.integers(0, len(preds)))], ta, tb)
        if any(np.array_equal(task, f) for f in forbidden):
            dropped += 1
            continue
        parts = {int(i)} if form == 1 else {int(i), int(j)}
        if rng.random() < 0.5:
            deeper = space.loop(task, lp) if rng.random() < 0.5 else \
                space.seq(task, pool[int(rng.integers(0, npool))][1])
            if not np.array_equal(deeper, task):
                xs.append(features(task, deeper, space, 8))
                ys.append(1.0)
                for idx in parts:
                    e, t = pool[idx]
                    xs.append(features(t, deeper, space, size_of(e)))
                    ys.append(1.0)
                for idx in rng.integers(0, npool, 6):
                    e, t = pool[int(idx)]
                    xs.append(features(t, deeper, space, size_of(e)))
                    ys.append(0.0)
        for idx in parts:
            e, t = pool[idx]
            xs.append(features(t, task, space, size_of(e)))
            ys.append(1.0)
        for idx in rng.integers(0, npool, 8):
            if int(idx) in parts:
                continue
            e, t = pool[int(idx)]
            xs.append(features(t, task, space, size_of(e)))
            ys.append(0.0)
    return np.array(xs), np.array(ys), dropped


def main() -> int:
    t0 = time.perf_counter()
    alpha = sorted({c for t in TAPES for c in t})
    space = Space(TAPES, alpha)
    lp = max(len(t) for t in TAPES) + 2

    print("X50: what a counter cannot remember\n")
    print("primitives: NOP ADV EMIT HALT PUSH POP  SEQ IF LOOP  "
          "AT(offset,char) TOP(char) EMPTY OR")
    print("PUSH pushes the character under the head. Nothing is named "
          "`open`, `close` or `match`.")
    print(f"tapes {TAPES}   alphabet {''.join(alpha)!r}")
    print(f"situations {space.n:,} (tape x position x stack, |stack| <= "
          f"{space.bound}), width {space.width:,}")

    bad = [e for e in TRUTHS.values()
           if not np.array_equal(space.table(e), space.interpret(e))]
    print(f"table/interpreter agreement: {len(TRUTHS)-len(bad)}/{len(TRUTHS)}"
          f"{'  <-- FAST PATH IS WRONG' if bad else ''}")
    if bad:
        return 1
    over = [t for t in TAPES
            if max(scan_depth(t, i) for i in range(len(t) + 1)) > space.bound]
    print(f"search tapes reaching the abstraction bound: {len(over)} "
          f"(must be 0, or the abstraction is lossy)")
    if over:
        return 1

    print("\nCERTIFICATES, computed from the evidence")
    for use_depth, claim in ((False, "the window alone"),
                             (True, "the window PLUS any counter")):
        key, byd = certificate(TAPES, TRUTHS[TYPED], use_depth)
        if key is None:
            print(f"  no collision -- {claim} is NOT ruled out on these tapes")
            continue
        what = f"window {''.join(key[0])!r}" + (f", depth {key[1]}" if use_depth else "")
        print(f"  {claim} is insufficient: {what} occurs with both decisions")
        for dec, where in sorted(byd.items()):
            tape, pos = where[0]
            print(f"      emit={str(dec):5} at index {pos} of {tape!r}")

    fams = families(space, alpha)
    base, masks, seen = [], [], set()
    for p in ([("AT", o, c) for o in (-1, 0, 1) for c in list(alpha) + ["$"]]
              + [("TOP", c) for c in alpha] + [("EMPTY",)]):
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        base.append(p)
        masks.append(m)
    print(f"\n{len(base)} live atomic tests after dedup")

    forbidden = [space.interpret(e) for e in TRUTHS.values()]
    rng = np.random.default_rng(4)
    xs, ys, dropped = training_set(space, list(depth1(space, base, masks)),
                                   base, masks, forbidden, 150, rng, lp)
    model = Logistic().fit(xs, ys)
    print(f"proposer: {len(xs):,} examples, {dropped} discarded\n")

    rr = np.random.default_rng(3)
    arms = {
        "size": (lambda v, t, s, n: -n, top_k),
        "random": (lambda v, t, s, n: float(rr.random()), top_k),
        "similar": (lambda v, t, s, n: float((v == t).mean()), top_k),
        "cover": (lambda v, t, s, n: model.score(features(v, t, s, n)), cover_k),
    }
    # X49 measured every arm tying, so two of the six are dropped here to buy
    # runtime for a third tape. `random` stays -- an experiment with no arm
    # that can fail measures nothing.
    head = f'{"true rule":16} {"nodes":>5} ' + " ".join(f"{a:>9}" for a in arms)
    print(head + "\n" + "-" * len(head))
    tally = {a: Counter() for a in arms}
    built = {}
    for name, truth in TRUTHS.items():
        ttab = space.interpret(truth)
        dv = derive(space, ttab, fams)
        preds = base + [d[1] for d in dv]
        pmasks = masks + [space.pred(d[1]) for d in dv]
        pool = list(depth1(space, preds, pmasks))
        cells = []
        for arm, (fn, sel) in arms.items():
            hits, cost, depth = search(space, pool, preds, pmasks, ttab, fn,
                                       sel, KEEP, BEAM, BUDGET)
            ok = [h for h in hits
                  if all(output(h, tp) == output(truth, tp) for tp in EVAL)]
            if ok:
                tally[arm]["exact"] += 1
                built.setdefault(name, {})[arm] = min(ok, key=size_of)
                cells.append(f"{'d'+str(depth)+' '+str(int(cost/1000))+'k':>9}")
            elif hits:
                tally[arm]["overfit"] += 1
                cells.append(f'{"OVERFIT":>9}')
            else:
                tally[arm]["missed"] += 1
                cells.append(f'{"--":>9}')
        print(f"{name:16} {size_of(truth):>5} " + " ".join(cells))

    print()
    for arm in arms:
        c = tally[arm]
        print(f"  {arm:>9}: {c['exact']}/{len(TRUTHS)} exact"
              + (f", {c['overfit']} overfit" if c["overfit"] else "")
              + (f", {c['missed']} missed" if c["missed"] else ""))

    print(f"\nTHE COUNTER CONTROL: the same machine with TOP(c) removed, so the")
    print("stack can only be asked whether it is empty -- which is exactly the")
    print(f"power X49 had. Target: {TYPED!r}.")
    cpreds = [p for p in base if p[0] != "TOP"]
    cmasks = [m for p, m in zip(base, masks) if p[0] != "TOP"]
    ttab = space.interpret(TRUTHS[TYPED])
    cdv = [d for d in derive(space, ttab, {k: v for k, v in fams.items()
                                           if "top" not in k})]
    cpreds += [d[1] for d in cdv]
    cmasks += [space.pred(d[1]) for d in cdv]
    cpool = list(depth1(space, cpreds, cmasks))
    for arm in ("cover",):
        fn, sel = arms[arm]
        hits, cost, _ = search(space, cpool, cpreds, cmasks, ttab, fn, sel,
                               KEEP, BEAM, BUDGET)
        ok = [h for h in hits
              if all(output(h, tp) == output(TRUTHS[TYPED], tp) for tp in EVAL)]
        print(f"  {arm:>9}: {len(hits)} hits, {len(ok)} survive held-out "
              f"({cost:,} candidates, pool {len(cpool):,})")

    print("\nDOES IT GENERALISE PAST THE BOUND IT WAS SEARCHED UNDER?")
    print(f"  search abstraction: |stack| <= {space.bound}; search tapes nest "
          f"at most {max(max(scan_depth(t, i) for i in range(len(t)+1)) for t in TAPES)}.")
    print("  Below, the recovered programs run on an UNBOUNDED interpreter over")
    print("  tapes nesting far deeper than anything they were built from.")
    bestarm = max(arms, key=lambda a: tally[a]["exact"])
    for name, truth in TRUTHS.items():
        e = built.get(name, {}).get(bestarm)
        if e is None:
            print(f"  {name:16} -- not recovered --")
            continue
        wrong = [tp for tp in DEEP
                 if output(e, tp, bound=None) != output(truth, tp, bound=None)]
        depths = max(max(scan_depth(t, i) for i in range(len(t) + 1)) for t in DEEP)
        print(f"  {name:16} {'OK' if not wrong else f'FAILS on {wrong[0]!r}'}"
              f"   (deepest nesting tested: {depths})")
    print("\n  ...AND PAST THE ALPHABET IT WAS BUILT FROM?")
    print("  The same shapes with characters the evidence never contained.")
    for name, truth in TRUTHS.items():
        e = built.get(name, {}).get(bestarm)
        if e is None:
            continue
        wrong = [tp for tp in FRESH
                 if output(e, tp, bound=None) != output(truth, tp, bound=None)]
        if wrong:
            tp = wrong[0]
            print(f"  {name:16} FAILS on {tp!r}: got "
                  f"{output(e, tp, bound=None)[0]!r} want "
                  f"{output(truth, tp, bound=None)[0]!r}")
        else:
            print(f"  {name:16} OK")

    print(f"\n  {TYPED} recovered as:")
    e = built.get(TYPED, {}).get(bestarm)
    print(f"    {render(e) if e is not None else '-- not recovered --'}")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
