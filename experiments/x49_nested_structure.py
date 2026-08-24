"""X49: structure no window can see, and where the leverage actually was.

X48 scaled the substrate to a token stream. The obvious next step is "bigger
alphabet, harder tasks". One of those tasks turned out not to be a search
problem at all, and chasing it moved the whole result somewhere unexpected.

THE WALL, AS A COMPUTED CERTIFICATE. Every program here reads a fixed window
around the head and has no state but the head position, so its decision at a
position is a function of that window. Copying what lies inside brackets
needs the nesting DEPTH, which is in no window. The proof is one pass over
the evidence, not an argument:

    window 'xay' occurs at position 1 of 'xay[xay]0b1c' -- do NOT emit
    window 'xay' occurs at position 5 of the SAME tape  -- DO emit

Same window, opposite decisions. No window-only program can be correct, at
any budget, alphabet, or ranking. Confirmed empirically too: with the counter
removed, 1,542,456 candidates produce 0 hits.

The fix is not vocabulary, it is one general primitive -- a bounded counter,
INC / DEC / DEEP -- which knows nothing about brackets, exactly as LOOP knows
nothing about sliding.

MEASURED (6 targets, 86s). `nodes` is the size of the rule being recovered:

  true rule        nodes    size  random similar learned  cover cover-rnd
  copy all             4      8k     10k     10k     10k    10k       10k
  scan to open         5     33k     34k     27k     10k    27k       26k
  strip brackets      10     28k     71k     36k     71k    36k       36k
  halt on digit       12      8k     10k     10k     10k    10k       10k
  copy digits         13     21k     62k     28k     62k    28k       28k
  copy inside []      17     50k     75k     66k     66k    84k       82k
  ----------------------------------------------------------------------
                          6/6     6/6     6/6     6/6    6/6       6/6

    copy inside []  (LOOP (IF '['@0 (SEQ ADV INC)
                         (IF ']'@0 (SEQ ADV DEC)
                          (IF DEEP (SEQ EMIT ADV) ADV))))

Every program survives 600 SYNTHESISED adversarial tapes drawn from the
observed bytes plus bytes the evidence never showed -- a harder test than a
fixed held-out set, because the attacker is hunting for the gap.

THE RESULT THAT ARGUES AGAINST X47 AND X48. Every arm scores 6/6. Random
ties with the trained proposer. In X47 that gap was 6/6 against 1/6 and the
claim was "selection is the lever"; here selection does not matter at all,
because two things removed the need for it:

  a DECISION LIST is a shape binary composition cannot reach. `copy inside []`
  is a four-deep IF chain, so a search that carries only SEQ results between
  passes can never assemble it however large the budget. Grown one rule at a
  time it costs six cheap rounds. Before this, every arm scored 0 on it;
  after, every arm scores 6/6 and the whole run got 4x faster.

  the RULE POOL should never have been ranked. Decision-list rules are short
  action sequences -- SEQ(INC, ADV), SEQ(ADV, HALT) -- and there are only
  |acts| + |acts|^2 of them. Making 49 blocks compete for a top-120 slot
  against two thousand predicate-headed ones is a filter with nothing to
  gain. Admitting them unconditionally took every arm from 3/6 to 5/6.

So the honest reading is that X47's lever was real for the regime it was
measured in, and this regime is not that one. When the right SHAPE is
available, ranking stops being the bottleneck. Reporting a 6/6 as if the
proposer earned it would be false.

OCCAM HAS TO BE APPLIED AFTER THE LIST IS BUILT, NOT DURING. Greedy
construction first recovered `copy inside []` with a 60-node ENUMERATION of
the observed (character, depth) pairs where one-node DEEP belonged. Both were
exact on the evidence; only one was right, and the held-out tape proved it:
`got 'bc' want 'bce'`, because `e` had never appeared inside brackets. A
tie-break cannot catch this -- when that rule is added the bracket rules are
not in the chain yet, so the enumeration genuinely scores BETTER. Only after
the chain is complete can each test be swapped for the smallest one that
still matches. That pass turns the enumeration into DEEP and the program
generalises.

WHAT A 256-BYTE ALPHABET IS ACTUALLY FOR. Not atoms: a byte the evidence
never shows has an empty test and dedups away, so 21 observed bytes give 46
live tests and the other 235 contribute nothing. What the byte space buys is
the ability to PROBE with a symbol never seen. That is the only way to
discover that a derived character class was a list of observations rather
than a rule -- and it worked: against the enumerated predicate the probe found
a splitting tape in 2 tries, `'1ov76t[cxto#'`, candidate 'cxto' against truth
'cxto#'.

AND THE HONEST NOTE ABOUT THAT LOOP: in the final system it fires zero times,
because simplification fixes the program before any probe is needed. It
earned its place as a diagnostic and as verification, not as a contributor.
Saying otherwise would credit it with the simplifier's work.

WHAT WAS REMOVED. BACK, and with it `began` from the state: it multiplies the
situation space by the tape length and nothing in X46-X48 ever used it -- even
X46's `pushback` came back in the proactive form without UNDO. Tasks needing
backtracking are now out of scope. That is a real narrowing.

WHAT THIS DOES NOT SHOW. Six targets, 21 bytes, tapes of length 12, nesting
bounded at 3. The counter is bounded, so this is a finite-state machine with
a wider state, not a pushdown automaton; genuinely unbounded nesting is still
out of reach. And with every arm at 6/6 this run says nothing about the
proposer either way -- it is a ceiling, not a separation.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x47_priced_vocabulary import Logistic, cover_k, top_k  # noqa: E402

MAXDEPTH = 3
ACTIONS = ("NOP", "ADV", "EMIT", "HALT", "HOME", "INC", "DEC")
PLAIN = ("NOP", "ADV", "EMIT", "HALT", "HOME")


@dataclass(frozen=True, slots=True)
class St:
    pos: int
    depth: int = 0
    out: tuple = ()
    live: bool = True


def test_pred(pred, tape, st) -> bool:
    """The single way any program looks at the world.

    Three shapes, and only three: a character at an offset, the nesting
    counter being non-zero, and a disjunction. Nothing named `bracket`.
    """
    k = pred[0]
    if k == "OR":
        return test_pred(pred[1], tape, st) or test_pred(pred[2], tape, st)
    if k == "DEEP":
        return st.depth > 0
    if k == "AT":                       # (AT, offset, char, depth or None)
        _, off, ch, d = pred
        i = st.pos + off
        if d is not None and st.depth != d:
            return False
        if ch == "$":
            return not (0 <= i < len(tape))
        return 0 <= i < len(tape) and tape[i] == ch
    raise ValueError(pred)


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
    if expr == "HOME":
        return replace(st, pos=0), fuel - 1
    if expr == "INC":
        return replace(st, depth=min(st.depth + 1, MAXDEPTH)), fuel - 1
    if expr == "DEC":
        return replace(st, depth=max(st.depth - 1, 0)), fuel - 1
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
    if e[0] in ("AT", "DEEP"):
        return 1
    return 1 + sum(size_of(p) for p in e[1:])


def render(e) -> str:
    if isinstance(e, str):
        return e
    k = e[0]
    if k == "DEEP":
        return "DEEP"
    if k == "AT":
        _, off, ch, d = e
        base = ("END" if ch == "$" else repr(ch)) + f"@{off:+d}"
        return base if d is None else f"{base}&d{d}"
    if k == "OR":
        return f"({render(e[1])}|{render(e[2])})"
    if k == "IF":
        return f"(IF {render(e[1])} {render(e[2])} {render(e[3])})"
    if k == "LOOP":
        return f"(LOOP {render(e[1])})"
    return f"(SEQ {render(e[1])} {render(e[2])})"


def or_chain(parts):
    term = parts[-1]
    for p in reversed(parts[:-1]):
        term = ("OR", p, term)
    return term


def output(expr, tape):
    res, _ = run(expr, tape, St(pos=0))
    return "".join(tape[p] for p in res.out), res.pos, res.depth, res.live


# ------------------------------------------------------------ the space


class Space:
    def __init__(self, tapes, depths=MAXDEPTH + 1):
        self.tapes, self.depths = tapes, depths
        self.sits, self.index = [], {}
        for ti, tp in enumerate(tapes):
            for pos in range(len(tp) + 1):
                for d in range(depths):
                    self.index[(ti, pos, d)] = len(self.sits)
                    self.sits.append((ti, pos, d))
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
        for i, (ti, pos, d) in enumerate(self.sits):
            L = len(self.tapes[ti])
            p, dd = pos, d
            if name == "HALT":
                halt[i] = True
            elif name == "ADV":
                p = min(pos + 1, L)
            elif name == "HOME":
                p = 0
            elif name == "INC":
                dd = min(d + 1, self.depths - 1)
            elif name == "DEC":
                dd = max(d - 1, 0)
            elif name == "EMIT" and pos < L:
                cnt[i, self.base[ti] + pos] = 1
            end[i] = self.index[(ti, p, dd)]
        return self.pack(end, halt, cnt)

    def pred(self, p):
        return np.array([test_pred(p, self.tapes[ti], St(pos=pos, depth=d))
                         for ti, pos, d in self.sits], dtype=bool)

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
        for i, (ti, pos, d) in enumerate(self.sits):
            res, _ = run(e, self.tapes[ti], St(pos=pos, depth=d))
            end[i] = self.index[(ti, res.pos, res.depth)]
            halt[i] = not res.live
            for q in res.out:
                cnt[i, self.base[ti] + q] += 1
        return self.pack(end, halt, cnt)


# ------------------------------------------- the wall, as a certificate


def window_certificate(tapes, truth, radius=1):
    """Two positions with the SAME window where the truth disagrees.

    A program here decides what to do from the window around the head and
    nothing else, so such a pair is a proof of impossibility for the whole
    window-only fragment -- no budget, alphabet or ranking can cross it.
    """
    seen = {}
    for tape in tapes:
        res, _ = run(truth, tape, St(pos=0))
        emitted = set(res.out)
        for p in range(len(tape)):
            win = tuple(tape[p + d] if 0 <= p + d < len(tape) else "$"
                        for d in range(-radius, radius + 1))
            seen.setdefault(win, {}).setdefault(p in emitted, []).append((tape, p))
    for win, bydecision in seen.items():
        if len(bydecision) > 1:
            return win, bydecision
    return None, None


# ---------------------------------------------------------- predicates


def families(space, chars, offsets):
    """Partitions to derive from -- plain, and the PRODUCT with depth.

    X48 could derive any event that was a union of character atoms. "Emit
    here" for the bracket task is a CONJUNCTION -- inside a bracket and not
    itself a bracket -- which no union of character tests expresses. A
    product of two partitions is still a partition, so pairing each character
    with each depth keeps the derivation exact and one pass long.
    """
    out = {}
    for off in offsets:
        out[f"char@{off:+d}"] = [(("AT", off, c, None), space.pred(("AT", off, c, None)))
                                 for c in chars + ["$"]]
        out[f"char@{off:+d}xdepth"] = [
            (("AT", off, c, d), space.pred(("AT", off, c, d)))
            for c in chars + ["$"] for d in range(space.depths)]
    return out


def derive(space, target, fams):
    end, halted, cnt = space.unpack(target)
    own = np.zeros(space.n, dtype=bool)
    for i, (ti, pos, _) in enumerate(space.sits):
        if pos < len(space.tapes[ti]):
            own[i] = cnt[i, space.base[ti] + pos] > 0
    events = {
        "emitted-here": own,
        "halted": halted,
        "head-stayed": (end == space.ident) & ~halted,
    }
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
        float((bv & ~bt).mean()), float((~bv & bt).mean()),
        size / 16.0,
    ], dtype=np.float64)


def depth1(space, preds, pmasks, acts=ACTIONS):
    for a in acts:
        yield ("LOOP", a), space.loop(space.atoms[a], 14)
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


def combine(space, left, right, preds, pmasks, budget):
    used = 0
    for ea, ta in left:
        for eb, tb in right:
            if used >= budget:
                return
            used += 1
            yield ("SEQ", ea, eb), space.seq(ta, tb)
    for p, pm in zip(preds, pmasks):
        for ea, ta in left:
            for eb, tb in right:
                if used >= budget:
                    return
                used += 1
                yield ("IF", p, ea, eb), space.branch(pm, ta, tb)


class _CoverShim:
    def __init__(self, width):
        self.n = width


def simplify_rules(space, expr, preds, target, wrap):
    """Replace each rule's test with the smallest one that still matches.

    Greedy construction is myopic about generality. When the third rule is
    added the bracket rules are not in the chain yet, so a 60-node
    ENUMERATION of the observed (character, depth) pairs genuinely scores
    better than the one-node DEEP that means the same thing once the earlier
    rules are in place. It is a strict improvement at that step, so no
    tie-break can catch it -- the simplification has to happen after the list
    is complete, when the rest of the chain exists to be simplified against.

    An enumeration is exact on the evidence and wrong on the next tape. This
    is the pass that turns one into a rule.
    """
    chain, node = [], expr
    while isinstance(node, tuple) and node[0] == "IF":
        chain.append([node[1], node[2]])
        node = node[3]
    if not chain:
        return expr

    def rebuild():
        e = node
        for p, b in reversed(chain):
            e = ("IF", p, b, e)
        return e

    order = sorted(preds, key=size_of)
    for i in range(len(chain)):
        keep = chain[i][0]
        for cand_p in order:
            if size_of(cand_p) >= size_of(keep):
                break
            chain[i][0] = cand_p
            if np.array_equal(wrap(space.table(rebuild())), target):
                keep = cand_p
                break
            chain[i][0] = keep
    return rebuild()


def decision_list(space, blocks, preds, pmasks, target, wrap, rounds=6):
    """Build IF(p1, b1, IF(p2, b2, ... default)) greedily.

    A decision list is the shape "handle each character class differently"
    actually takes, and binary composition cannot reach it in bounded depth:
    `copy inside []` is a four-deep IF chain, so a search that carries only
    SEQ results between passes can never assemble it no matter how large the
    budget. Grown one rule at a time it is six cheap rounds.

    Each round scores every (predicate, block) pair by how much closer it
    brings the WHOLE program to the target -- under `wrap`, which is either
    identity or LOOP, so the same builder covers looping and straight-line
    rules.
    """
    cost, agree = 0, lambda sig: float((sig == target).mean())
    # Smallest tests first, so a rule that ties on the evidence is won by the
    # simpler one. Without this the builder picked a 60-node ENUMERATION of
    # the (character, depth) pairs it had actually observed over the one-node
    # DEEP that means the same thing on this evidence and keeps meaning it on
    # the next tape. Both were exact; only one was right.
    order = sorted(range(len(preds)), key=lambda j: size_of(preds[j]))
    preds = [preds[j] for j in order]
    pmasks = [pmasks[j] for j in order]

    cur_a, cur_e, cur_t, cur_s = -1.0, None, None, 10 ** 9
    for e, t in blocks:
        cost += 1
        a = agree(wrap(t))
        if (a, -size_of(e)) > (cur_a, -cur_s):
            cur_a, cur_e, cur_t, cur_s = a, e, t, size_of(e)
    if cur_e is None:
        return [], cost
    if np.array_equal(wrap(cur_t), target):
        return [cur_e], cost
    for _ in range(rounds):
        best = None
        for p, pm in zip(preds, pmasks):
            for eb, tb in blocks:
                cost += 1
                cand = space.branch(pm, tb, cur_t)
                a = agree(wrap(cand))
                sz = size_of(p) + size_of(eb)
                if best is None or (a, -sz) > (best[0], -best[3]):
                    best = (a, ("IF", p, eb, cur_e), cand, sz)
        if best is None or best[0] <= cur_a:
            break
        cur_a, cur_e, cur_t = best[0], best[1], best[2]
        if np.array_equal(wrap(cur_t), target):
            return [simplify_rules(space, cur_e, preds, target, wrap)], cost
    return [], cost


def search(space, pool, preds, pmasks, target, score_fn, select_fn,
           keep, beam, budget, passes=5, acts=ACTIONS):
    cost, first, hits = 0, 0, []
    shim = _CoverShim(space.width)

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

    best = {}
    for i, (expr, _) in enumerate(pool):
        if expr[0] == "IF" and scores[i] > best.get(expr[1], -1e18):
            best[expr[1]] = scores[i]
    order = sorted(range(len(preds)), key=lambda j: -best.get(preds[j], -1e18))
    preds = [preds[j] for j in order]
    pmasks = [pmasks[j] for j in order]

    # Decision lists first: they are cheap and they are the one shape
    # binary composition cannot reach.
    #
    # Their rules are SHORT ACTION SEQUENCES -- SEQ(INC, ADV), SEQ(ADV, HALT) --
    # and there are only |acts| + |acts|^2 of those in total, so making them
    # compete for a top-120 slot against two thousand predicate-headed blocks
    # is a filter with nothing to gain. Every one of them is admitted here
    # regardless of rank; that is not a widened budget, it is the removal of a
    # ranking that never should have applied.
    short = [(a, space.atoms[a]) for a in acts]
    short += [(("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b]))
              for a in acts for b in acts if a != b]
    rule_pool = short + [b for b in blocks if b[0] not in {e for e, _ in short}]

    for wrap, tag in ((lambda t: t, None), (lambda t: space.loop(t, 14), "LOOP")):
        found, c = decision_list(space, rule_pool, preds, pmasks, target, wrap)
        cost += c
        if found:
            e = found[0] if tag is None else ("LOOP", found[0])
            return [e], cost, 0

    share = max(1, (budget - cost) // max(1, passes - 1))
    level = blocks
    for depth in range(2, passes + 1):
        nxt = []
        for expr, tab in combine(space, level, blocks, preds, pmasks, share):
            cost += 1
            if np.array_equal(tab, target):
                hits.append(expr)
                first = first or cost
            elif expr[0] == "SEQ":
                nxt.append((expr, tab))
        # LOOP is UNARY: wrapping every result of this level costs |level|
        # candidates, not |level| x |blocks|, so it never competes for a beam
        # slot. `copy inside brackets` is a LOOP over a four-deep body.
        for expr, tab in list(nxt):
            cost += 1
            if np.array_equal(space.loop(tab, 14), target):
                hits.append(("LOOP", expr))
                first = first or cost
        if hits:
            return hits, first, depth
        if cost >= budget or not nxt:
            return [], cost, 0
        s = np.array([score_fn(t, target, space, size_of(e)) for e, t in nxt])
        level = [nxt[i] for i in select_fn(shim, nxt, s, target, beam)]
    return [], cost, 0


# ----------------------------------------------------------- the truths

OPEN, CLOSE = ("AT", 0, "[", None), ("AT", 0, "]", None)
IS_DIGIT = or_chain([("AT", 0, c, None) for c in "0123"])

TRUTHS = {
    "copy all":        ("LOOP", ("SEQ", "EMIT", "ADV")),
    "scan to open":    ("LOOP", ("IF", OPEN, "NOP", "ADV")),
    "strip brackets":  ("LOOP", ("IF", OPEN, "ADV",
                                 ("IF", CLOSE, "ADV", ("SEQ", "EMIT", "ADV")))),
    "halt on digit":   ("SEQ", "ADV", ("IF", IS_DIGIT, "HALT", "NOP")),
    "copy digits":     ("LOOP", ("SEQ", ("IF", IS_DIGIT, "EMIT", "NOP"), "ADV")),
    # The one that needs memory. Nothing here is called `bracket`: INC and DEC
    # move a counter, DEEP asks whether it is non-zero.
    "copy inside []":  ("LOOP", ("IF", OPEN, ("SEQ", "INC", "ADV"),
                                 ("IF", CLOSE, ("SEQ", "DEC", "ADV"),
                                  ("IF", ("DEEP",), ("SEQ", "EMIT", "ADV"),
                                   "ADV")))),
}
NEEDS_COUNTER = {"copy inside []"}


def training_set(space, pool, preds, pmasks, forbidden, n_tasks, rng):
    xs, ys, dropped = [], [], 0
    npool = len(pool)
    for _ in range(n_tasks):
        i, j = rng.integers(0, npool, 2)
        (_, ta), (_, tb) = pool[int(i)], pool[int(j)]
        form = int(rng.integers(0, 3))
        if form == 0:
            task = space.seq(ta, tb)
        elif form == 1:
            task = space.loop(ta, 14)
        else:
            k = int(rng.integers(0, len(preds)))
            task = space.branch(pmasks[k], ta, tb)
        if any(np.array_equal(task, f) for f in forbidden):
            dropped += 1
            continue
        parts = {int(i)} if form == 1 else {int(i), int(j)}
        if rng.random() < 0.5:
            if rng.random() < 0.5:
                deeper, extra = space.loop(task, 14), set()
            else:
                k = int(rng.integers(0, npool))
                deeper, extra = space.seq(task, pool[k][1]), {k}
            if (not any(np.array_equal(deeper, f) for f in forbidden)
                    and not np.array_equal(deeper, task)):
                xs.append(features(task, deeper, space, 8))
                ys.append(1.0)
                for idx in parts | extra:
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



def prepare(tapes, chars, rng):
    """Everything that depends on the current evidence."""
    space = Space(tapes)
    fams = families(space, chars, (-1, 0, 1))
    preds, masks, seen = [], [], set()
    for p in [("AT", o, c, None) for o in (-1, 0, 1) for c in chars + ["$"]] \
            + [("DEEP",)]:
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        preds.append(p)
        masks.append(m)
    forbidden = [space.interpret(e) for e in TRUTHS.values()]
    base_pool = list(depth1(space, preds, masks))
    xs, ys, _ = training_set(space, base_pool, preds, masks, forbidden, 200, rng)
    return space, fams, preds, masks, Logistic().fit(xs, ys)


UNSEEN = list("efghjkvw56789!#-_=+")


def probe_disagreement(hits, truth, chars, rng, length=12, tries=600):
    """Write an input the candidates and the world disagree on.

    Only the truth is consulted -- that is the environment answering a query
    the system composed. The held-out tapes are never touched here; using
    them to decide what to ask would be marking your own homework.
    """
    # The probe alphabet includes bytes the evidence has NEVER shown. This is
    # what "a 256-byte alphabet" actually buys: not 256 atoms, which dedup to
    # nothing, but the ability to ask about a symbol you have never seen --
    # the only way to discover that a derived character class was a list of
    # observations rather than a rule.
    alpha = list(chars) + UNSEEN
    for k in range(tries):
        tape = "".join(alpha[int(i)] for i in rng.integers(0, len(alpha), length))
        for h in hits:
            if output(h, tape) != output(truth, tape):
                return tape, k + 1
    return None, tries


TAPES = ["xay[xay]0b1c", "[p[qr]s]t,u2", "m[n]o[[p]]q3"]
EVAL = ["a[bc]d0[e]1", "[[x]y]z2,w3", "q[r[st]u]v4", "0a[b]c[[d]]e",
        "[m]n[o[p]q]r", "zz[y[x]w]v1"]
BUDGET, KEEP, BEAM, PASSES = 3_000_000, 120, 40, 5
MAX_ROUNDS = 4


def main() -> int:
    t0 = time.perf_counter()
    chars = sorted({c for t in TAPES for c in t})
    space = Space(TAPES)

    print("X49: structure the window cannot see\n")
    print("primitives: NOP ADV EMIT HALT HOME INC DEC  SEQ IF LOOP  "
          "AT(offset,char[,depth]) DEEP OR")
    print(f"tapes {TAPES}")
    print(f"{len(chars)} distinct bytes observed of 256 possible -- a byte the "
          "evidence never")
    print("shows has an empty test and dedups away, so the live alphabet is "
          "what the\ndata exercises, not what the encoding allows.")
    print(f"situations {space.n} (tape x position x depth 0..{MAXDEPTH}), "
          f"signature width {space.width:,}\n")

    checks = list(TRUTHS.values()) + [
        ("LOOP", ("IF", OPEN, ("SEQ", "INC", "ADV"), ("SEQ", "DEC", "EMIT"))),
        ("SEQ", ("LOOP", ("SEQ", "EMIT", "ADV")), ("IF", ("DEEP",), "HALT", "HOME")),
    ]
    bad = sum(0 if np.array_equal(space.table(e), space.interpret(e)) else 1
              for e in checks)
    print(f"table/interpreter agreement: {len(checks)-bad}/{len(checks)}"
          f"{'  <-- FAST PATH IS WRONG' if bad else ''}")
    if bad:
        return 1

    print("\nTHE WALL, AS A CERTIFICATE rather than an argument")
    win, byd = window_certificate(TAPES, TRUTHS["copy inside []"])
    if win is None:
        print("  no colliding window found -- the impossibility claim is NOT")
        print("  established on these tapes; do not make it.")
    else:
        print(f"  window {''.join(win)!r} occurs with BOTH decisions:")
        for dec, where in sorted(byd.items()):
            tape, pos = where[0]
            print(f"    emit={str(dec):5} at position {pos} of {tape!r}")
        print("  A program whose only state is the head position decides from")
        print("  the window alone, so no such program can be correct here --")
        print("  regardless of budget, alphabet, or ranking. It needs memory.")

    fams = families(space, chars, (-1, 0, 1))
    base_preds, base_masks, seen = [], [], set()
    for p in [("AT", o, c, None) for o in (-1, 0, 1) for c in chars + ["$"]] \
            + [("DEEP",)]:
        m = space.pred(p)
        if m.tobytes() in seen or not m.any():
            continue
        seen.add(m.tobytes())
        base_preds.append(p)
        base_masks.append(m)
    print(f"\n{len(base_preds)} live atomic tests after dedup "
          f"(3 offsets x {len(chars)} bytes + END + DEEP)")

    truth_tabs = [space.interpret(e) for e in TRUTHS.values()]
    base_pool = list(depth1(space, base_preds, base_masks))
    rng = np.random.default_rng(4)
    xs, ys, dropped = training_set(space, base_pool, base_preds, base_masks,
                                   truth_tabs, 300, rng)
    model = Logistic().fit(xs, ys)
    print(f"proposer: {len(xs):,} examples, {dropped} discarded for matching "
          f"a target\n")

    rr = np.random.default_rng(3)
    sc = {"size": lambda v, t, sp, n: -n,
          "random": lambda v, t, sp, n: float(rr.random()),
          "similar": lambda v, t, sp, n: float((v == t).mean()),
          "learned": lambda v, t, sp, n: model.score(features(v, t, sp, n))}
    arms = {"size": (sc["size"], top_k), "random": (sc["random"], top_k),
            "similar": (sc["similar"], top_k), "learned": (sc["learned"], top_k),
            "cover": (sc["learned"], cover_k), "cover-rnd": (sc["random"], cover_k)}

    head = (f'{"true rule":16} {"nodes":>5} {"asked":>6} '
            + " ".join(f"{a:>9}" for a in arms))
    print(head + "\n" + "-" * len(head))
    tally = {a: Counter() for a in arms}
    built = {}
    for name, truth in TRUTHS.items():
        tapes, asked = list(TAPES), 0
        # The evidence is grown by the SYSTEM, once per target, using one
        # reference arm -- growing it separately per arm would give the arms
        # different evidence and make the comparison meaningless.
        for rnd in range(MAX_ROUNDS):
            sp, fm, pr, pm, md = prepare(tapes, chars, np.random.default_rng(4))
            tt = sp.interpret(truth)
            dv = derive(sp, tt, fm)
            pr2, pm2 = pr + [d[1] for d in dv], pm + [sp.pred(d[1]) for d in dv]
            pool = list(depth1(sp, pr2, pm2))
            hits, _, _ = search(sp, pool, pr2, pm2, tt,
                                lambda v, t, s_, n: md.score(features(v, t, s_, n)),
                                cover_k, KEEP, BEAM, BUDGET, PASSES)
            if not hits or len(tapes) >= 6:
                break
            tape, _ = probe_disagreement(hits, truth, chars,
                                         np.random.default_rng(500 + rnd))
            if tape is None:
                break
            tapes.append(tape)
            asked += 1

        space2, fams2, preds2, masks2, model2 = prepare(
            tapes, chars, np.random.default_rng(4))
        ttab = space2.interpret(truth)
        dv = derive(space2, ttab, fams2)
        preds3 = preds2 + [d[1] for d in dv]
        masks3 = masks2 + [space2.pred(d[1]) for d in dv]
        pool = list(depth1(space2, preds3, masks3))
        armfns = {"size": (lambda v, t, s_, n: -n, top_k),
                  "random": (lambda v, t, s_, n: float(rr.random()), top_k),
                  "similar": (lambda v, t, s_, n: float((v == t).mean()), top_k),
                  "learned": (lambda v, t, s_, n: model2.score(features(v, t, s_, n)),
                              top_k),
                  "cover": (lambda v, t, s_, n: model2.score(features(v, t, s_, n)),
                            cover_k),
                  "cover-rnd": (lambda v, t, s_, n: float(rr.random()), cover_k)}
        cells = []
        for arm in arms:
            fn, sel = armfns[arm]
            hits, cost, depth = search(space2, pool, preds3, masks3, ttab, fn,
                                       sel, KEEP, BEAM, BUDGET, PASSES)
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
        print(f"{name:16} {size_of(truth):>5} {asked:>6} " + " ".join(cells))

    print()
    for arm in arms:
        c = tally[arm]
        print(f"  {arm:>9}: {c['exact']}/6 exact"
              + (f", {c['overfit']} overfit" if c["overfit"] else "")
              + (f", {c['missed']} missed" if c["missed"] else ""))

    # The counter-free control: same task, same everything, no INC/DEC/DEEP.
    print("\nTHE SAME TASK WITH THE COUNTER REMOVED (no INC, DEC or DEEP)")
    flat = Space(TAPES, depths=1)
    fpreds, fmasks, seen2 = [], [], set()
    for p in [("AT", o, c, None) for o in (-1, 0, 1) for c in chars + ["$"]]:
        m = flat.pred(p)
        if m.tobytes() in seen2 or not m.any():
            continue
        seen2.add(m.tobytes())
        fpreds.append(p)
        fmasks.append(m)
    ftab = flat.interpret(TRUTHS["copy inside []"])
    fderived = derive(flat, ftab, families(flat, chars, (-1, 0, 1)))
    fpreds += [d[1] for d in fderived]
    fmasks += [flat.pred(d[1]) for d in fderived]
    fpool = list(depth1(flat, fpreds, fmasks, acts=PLAIN))
    for arm in ("learned", "cover"):
        fn, sel = arms[arm]
        hits, cost, depth = search(flat, fpool, fpreds, fmasks, ftab, fn, sel,
                                   KEEP, BEAM, BUDGET, PASSES, acts=PLAIN)
        ok = [h for h in hits if all(
            output(h, tp) == output(TRUTHS["copy inside []"], tp) for tp in EVAL)]
        print(f"  {arm:>9}: {len(hits)} hits, {len(ok)} survive held-out "
              f"({cost:,} candidates, pool {len(fpool):,})")

    best = max(arms, key=lambda a: tally[a]["exact"])
    print(f"\nWHAT THE BEST ARM ({best}) BUILT, AND WHETHER IT SURVIVES ATTACK")
    print("  600 synthesised tapes per program, drawn from the observed bytes")
    print("  PLUS bytes the evidence never showed -- a far harder test than a")
    print("  fixed held-out set, because the attacker is looking for the gap.")
    for name, truth in TRUTHS.items():
        e = built.get(name, {}).get(best)
        if e is None:
            print(f"  {name:16} -- not found --")
            continue
        tape, tries = probe_disagreement([e], truth, chars,
                                         np.random.default_rng(9))
        verdict = (f"SPLIT after {tries} by {tape!r}" if tape
                   else "survived all 600")
        print(f"  {name:16} {verdict}")
        print(f"  {'':16} {render(e)[:78]}")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
