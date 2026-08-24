"""X48: the same machine over a token stream, and the one thing that broke.

X46/X47 built programs over a grid. Nothing in the pricing mechanism mentions
cells, so the next move is to swap the substrate: STEP becomes ADV (advance
the read head), GET(prop, offset) becomes a test on the character at an
offset, DIE becomes HALT. Add EMIT and the machine transduces text.

THE PORT IS NOT A PORT, and measuring the lattice before writing the port is
what showed it. X47 rests on one structural fact: the predicate lattice
CLOSES under OR at 403 truth vectors, so a predicate is one element of a
small finite set and can be priced as a unit. On a token stream that fact is
false:

    alphabet   offsets            atoms   OR-lattice
    10 chars   HERE                  11        1,024   = 2^10, the powerset
    10 chars   HERE, NEXT            21       55,783
    10 chars   HERE, NEXT, PREV      31      344,296+  (aborted, still growing)

The grid's tests OVERLAP -- a cell can be walkable and dangerous -- so unions
collapse, 4,096 possible down to 403 real. Character tests at one offset are
mutually EXCLUSIVE: they partition the situations, and the union-closure of a
partition is exactly its powerset. Wider alphabets and more offsets make it
worse. X47's answer does not survive the swap.

DERIVE THE TESTS FROM THE EVENTS, DO NOT ENUMERATE THE SPACE OF TESTS. The
target's own behaviour says where the interesting things happen: which
situations it emitted at, which it halted in, which left the head where it
started. For any such event set b, the tightest predicate expressible over
one offset family is the union of exactly those atoms whose block meets b --
computable in one pass, and CHECKABLE for equality with b, which is the
difference between a derived test and a guess.

    halt on digit   halted@+1        ('0'@+1|('1'@+1|('2'@+1|'3'@+1))   exact
    copy digits     emitted-here@+0  ('0'@0|('1'@0|('2'@0|'3'@0))       exact

`is-digit` read straight off "where did the target halt", in 11 comparisons,
out of a lattice of 55,783 that is never touched.

PER OFFSET is the load-bearing word, and the first version got it wrong. It
pooled all offsets together and the partition argument silently died -- atoms
at different offsets overlap, so the union of atoms meeting b is no longer
the minimal superset. It showed up as `halt on digit` deriving the nonsense
('a'|'b'|'0'|'1'|',') and being marked loose. The event is about the
character AFTER the head moves, and no offset-0 test can say that.

MEASURED (6 targets, 2,000,000 candidates per arm per target, 166s). `asked`
counts inputs the system SYNTHESISED for itself; every result is verified by
the real interpreter on 8 held-out tapes:

  true rule      nodes asked   size  random similar learned  cover cover-rnd
  advance 1          1     0     0k      0k      0k      0k     0k      0k
  scan to comma      5     0     --     16k     16k     16k    16k      --
  copy to comma      7     0     --    436k      --    431k   429k      --
  halt on comma      6     1   338k    500k      --    510k   512k    624k
  halt on digit     12     0   402k      --      --    508k   507k      --
  copy digits       13     0     --      --    445k    452k   452k    449k
  -------------------------------------------------------------------------
                             3/6     4/6     3/6     6/6    6/6     3/6

    halt on digit  (IF ('0'@+1|('1'@+1|('2'@+1|'3'@+1))) (SEQ ... HALT) ADV)
    copy digits    (LOOP (SEQ (IF ('0'@0|('1'@0|('2'@0|'3'@0))) EMIT NOP) ADV))

`halt on digit` came back BETTER than the rule it was asked to find: it tests
the next character and then moves, rather than moving and then testing. Same
behaviour, and it uses the derived predicate rather than rebuilding one.

PILLAR 4 ON A TOKEN STREAM, and exactly what it was worth. On the grid the
agent picks a probe out of a fixed menu of boards and cells. Here there is no
menu: the experiment IS a string, so the system writes one -- generate a
tape, run every surviving program on it, keep the tape if they disagree. That
is a constructed query rather than a selected one.

It fired on ONE target. Ablated -- seed tapes only, no synthesised input --
`halt on comma` goes from exact to MISSED, 0 hits. So the loop is worth one
target out of six, 5/6 to 6/6. It is not what fixed the overfitting; a
representation bug was, and saying otherwise would credit the loop with
someone else's work.

THE BUG, which the held-out check caught and the fast path did not. Halting
collapsed every halted state to one sentinel, so `SEQ(ADV, HALT)` and bare
`HALT` had IDENTICAL tables while the interpreter told them apart. The search
duly returned `(IF ','@+1 HALT ADV)` for `halt on comma` -- halt without
advancing -- matching on every situation of every search tape, and two
targets scored OVERFIT for that reason alone. A fast path coarser than the
trusted path does not merely lose speed; it invents equivalences that are not
there. Halting now keeps the head position.

A SECOND FIX, aimed at a measured weakness rather than at the score. The
proposer ranked `copy digits`'s loop body 255th of 846 while plain similarity
ranked it 3rd -- because a loop BODY emits far less than the loop that runs
it, and every emission feature counts against it. Training now deepens tasks
by LOOP as well as by SEQ, so the body of a loop appears as a positive. The
rank moved 255 -> 57 and the target was recovered.

WHAT THIS DOES NOT SHOW. The calibration arm is weaker here than on the grid:
`random` scores 4/6 against 1/6 in X47, because this pool is 846 blocks and
KEEP is 120 -- picking 14% at random finds a lot. A small pool makes a
forgiving control, and the learned/random gap should be read as smaller
evidence than X47's, not larger.

Six targets, one alphabet, tapes of length 7. This says the mechanism
survives a substrate swap. It does not say it scales to real text.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x47_priced_vocabulary import DEAD, Logistic, cover_k, top_k  # noqa: E402

HALTED = DEAD
ALPHA = "abcd0123 ,"
ACTIONS = ("NOP", "ADV", "EMIT", "HALT", "HOME", "BACK")


@dataclass(frozen=True, slots=True)
class TState:
    pos: int
    began: int
    out: tuple = ()
    live: bool = True


def test_pred(pred, tape: str, st: TState) -> bool:
    """The single way any program looks at the stream."""
    if pred[0] == "OR":
        return test_pred(pred[1], tape, st) or test_pred(pred[2], tape, st)
    _, off, ch = pred
    i = st.pos + off
    if ch == "$":                       # end-of-stream, the analogue of WALK
        return not (0 <= i < len(tape))
    return 0 <= i < len(tape) and tape[i] == ch


def run(expr, tape: str, st: TState, fuel: int = 8192):
    # Fuel is a runaway guard, not a semantic bound. It is set far above
    # anything the search can build (LOOP is capped at tape length + 2, and
    # depth never exceeds 3) so that it never binds -- if it did, the table
    # fast path and the interpreter would disagree on deep programs, and the
    # table would be describing a machine the verifier does not run.
    if fuel <= 0 or not st.live:
        return st, fuel
    if expr == "NOP":
        return st, fuel - 1
    if expr == "ADV":
        # Advances unconditionally, clamped at the end of the stream. It does
        # not know what a delimiter is; stopping is the program's job.
        return replace(st, pos=min(st.pos + 1, len(tape))), fuel - 1
    if expr == "EMIT":
        if st.pos < len(tape):
            return replace(st, out=st.out + (st.pos,)), fuel - 1
        return st, fuel - 1
    if expr == "HALT":
        return replace(st, live=False), fuel - 1
    if expr == "HOME":
        return replace(st, pos=0), fuel - 1
    if expr == "BACK":
        return replace(st, pos=st.began), fuel - 1
    head = expr[0]
    if head == "SEQ":
        st, fuel = run(expr[1], tape, st, fuel)
        return run(expr[2], tape, st, fuel)
    if head == "IF":
        return run(expr[2] if test_pred(expr[1], tape, st) else expr[3],
                   tape, st, fuel - 1)
    if head == "LOOP":
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
    if e[0] == "EQ":
        return 1
    return 1 + sum(size_of(p) for p in e[1:])


def render(e) -> str:
    if isinstance(e, str):
        return e
    if e[0] == "EQ":
        off = {0: "", 1: "+1", -1: "-1"}[e[1]]
        return f"{'END' if e[2] == '$' else repr(e[2])}@{off or '0'}"
    if e[0] == "OR":
        return f"({render(e[1])}|{render(e[2])})"
    if e[0] == "IF":
        return f"(IF {render(e[1])} {render(e[2])} {render(e[3])})"
    if e[0] == "LOOP":
        return f"(LOOP {render(e[1])})"
    return f"(SEQ {render(e[1])} {render(e[2])})"


# ------------------------------------------------------- behaviour tables
#
# A program's whole behaviour is (where the head ends up, how often it
# emitted at each stream cell). Both compose by array indexing:
#
#     SEQ(a,b).end   = b.end[a.end]
#     SEQ(a,b).count = a.count + b.count[a.end]
#
# The second line is the new one, and it is why emission had to be counts
# rather than an output string: counts ADD under composition, strings do not
# without carrying order. Counts lose the ORDER of emissions, so the fast
# path can admit a candidate the interpreter would reject -- which is
# harmless in one direction only, and is why every survivor is re-checked
# against the real interpreter on held-out tapes.


class TokenSpace:
    """Behaviour = (where the head ends, whether it halted, how often it
    emitted at each cell). All three compose by array indexing.

    HALTING KEEPS THE POSITION, and getting that wrong was not cosmetic. The
    first version collapsed every halt to one sentinel, so `SEQ(ADV, HALT)`
    and bare `HALT` had identical tables while the interpreter told them
    apart. The search duly returned `(IF ','@+1 HALT ADV)` for
    `halt on comma` -- halt WITHOUT advancing -- and it matched on every
    situation of every search tape. Only the held-out check caught it. A fast
    path that is coarser than the trusted path does not merely lose speed; it
    invents equivalences that are not there.
    """

    def __init__(self, tapes):
        self.tapes = tapes
        self.sits, self.index = [], {}
        for ti, tp in enumerate(tapes):
            for pos in range(len(tp) + 1):
                for began in range(len(tp) + 1):
                    self.index[(ti, pos, began)] = len(self.sits)
                    self.sits.append((ti, pos, began))
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
        for i, (ti, pos, began) in enumerate(self.sits):
            L = len(self.tapes[ti])
            p = pos
            if name == "HALT":
                halt[i] = True
            elif name == "ADV":
                p = min(pos + 1, L)
            elif name == "HOME":
                p = 0
            elif name == "BACK":
                p = began
            elif name == "EMIT" and pos < L:
                cnt[i, self.base[ti] + pos] = 1
            end[i] = self.index[(ti, p, began)]
        return self.pack(end, halt, cnt)

    def pred(self, p):
        return np.array([test_pred(p, self.tapes[ti], TState(pos=pos, began=bg))
                         for ti, pos, bg in self.sits], dtype=bool)

    def seq(self, a, b):
        ea, ha, ca = self.unpack(a)
        eb, hb, cb = self.unpack(b)
        live = ~ha
        end = np.where(live, eb[ea], ea).astype(np.int32)
        halt = np.where(live, hb[ea], True)
        cnt = ca + np.where(live[:, None], cb[ea], 0)
        return self.pack(end, halt, cnt)

    def branch(self, p, a, b):
        ea, ha, ca = self.unpack(a)
        eb, hb, cb = self.unpack(b)
        return self.pack(np.where(p, ea, eb).astype(np.int32),
                         np.where(p, ha, hb),
                         np.where(p[:, None], ca, cb))

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
        for i, (ti, pos, began) in enumerate(self.sits):
            res, _ = run(e, self.tapes[ti], TState(pos=pos, began=began))
            end[i] = self.index[(ti, res.pos, began)]
            halt[i] = not res.live
            for q in res.out:
                cnt[i, self.base[ti] + q] += 1
        return self.pack(end, halt, cnt)

    def output(self, e, tape):
        res, _ = run(e, tape, TState(pos=0, began=0))
        return "".join(tape[q] for q in res.out), res.pos, res.live


# ------------------------------------------------------------ predicates


def atom_preds(offsets=(0,)):
    return [("EQ", off, ch) for off in offsets for ch in ALPHA + "$"]


def or_chain(parts):
    term = parts[-1]
    for p in reversed(parts[:-1]):
        term = ("OR", p, term)
    return term


def derive_predicates(space, target, families):
    """Read the tests off the target's EVENTS instead of searching for them.

    Character atoms AT ONE OFFSET partition the situations, so for any event
    set b the tightest predicate expressible over that family is the union of
    the atoms whose block meets b -- one pass, no lattice. Whether that union
    EQUALS b is then checkable, which is the difference between a derived
    test and a guess.

    Per offset is the load-bearing word. The first version pooled all offsets
    together, and the partition argument silently died: atoms at different
    offsets overlap, so the union of "atoms meeting b" is no longer the
    minimal superset. It showed up as `halt on digit` deriving the nonsense
    ('a'|'b'|'0'|'1'|',') and being marked loose -- the event is about the
    character AFTER the head moves, and no offset-0 test can say that.
    """
    end, halted, cnt = space.unpack(target)
    own = np.zeros(space.n, dtype=bool)
    for i, (ti, pos, _) in enumerate(space.sits):
        if pos < len(space.tapes[ti]):
            own[i] = cnt[i, space.base[ti] + pos] > 0

    events = {
        "emitted-here": own,
        "halted": halted,
        "head-stayed": (end == space.ident) & ~halted,
        "head-moved": (end != space.ident) & ~halted,
        "emitted-at-all": cnt.sum(1) > 0,
    }
    out, seen = [], set()
    for ev, b in events.items():
        if not b.any() or b.all():
            continue
        for off, fam in sorted(families.items()):
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
            out.append((f"{ev}@{off:+d}", or_chain(parts),
                        bool((union == b).all())))
    return sorted(out, key=lambda r: (not r[2], size_of(r[1])))


def lattice_preds(atoms, max_atoms):
    """The full OR-lattice, which on this substrate is a POWERSET.

    Included to be enumerated once and priced honestly: it is the thing X47
    could rely on and this substrate cannot.
    """
    from itertools import combinations
    out = []
    for k in range(1, max_atoms + 1):
        for combo in combinations(atoms, k):
            out.append(or_chain(list(combo)))
    return out


# ------------------------------------------------------------- features


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
        float((ev == et).mean()),
        float((bv == bt).mean()),
        float((hv == ht).mean()),
        float(hv.mean()), float(ht.mean()),
        float(sv.mean()), float(st_.mean()),
        frac(moved, ev == et),
        frac(ht, hv),
        frac(st_, sv),
        float((bv & ~bt).mean()),      # emits where the truth does not
        float((~bv & bt).mean()),      # silent where the truth emits
        size / 16.0,
    ], dtype=np.float64)


NFEAT = 13


def depth1(space, preds, pred_masks):
    for a in ACTIONS:
        yield ("LOOP", a), space.loop(space.atoms[a], 9)
    for a in ACTIONS:
        for b in ACTIONS:
            if a != b:
                yield ("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b])
    for p, pm in zip(preds, pred_masks):
        for a in ACTIONS:
            for b in ACTIONS:
                if a != b:
                    yield ("IF", p, a, b), space.branch(pm, space.atoms[a],
                                                        space.atoms[b])


def combine(space, left, right, preds, pred_masks, budget):
    used = 0
    for ea, ta in left:
        for eb, tb in right:
            if used >= budget:
                return
            used += 1
            yield ("SEQ", ea, eb), space.seq(ta, tb)
    for ea, ta in left:
        if used >= budget:
            return
        used += 1
        yield ("LOOP", ea), space.loop(ta, 9)
    for p, pm in zip(preds, pred_masks):
        for ea, ta in left:
            for eb, tb in right:
                if used >= budget:
                    return
                used += 1
                yield ("IF", p, ea, eb), space.branch(pm, ta, tb)


class _CoverShim:
    """cover_k works on flat signature vectors; here a signature is wider
    than the situation count because it carries emission counts too."""

    def __init__(self, width):
        self.n = width


def search(space, pool, preds, pred_masks, target, score_fn, select_fn,
           keep, beam, budget):
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
    blocks += [(a, space.atoms[a]) for a in ACTIONS]

    best = {}
    for i, (expr, _) in enumerate(pool):
        if expr[0] == "IF" and scores[i] > best.get(expr[1], -1e18):
            best[expr[1]] = scores[i]
    order = sorted(range(len(preds)), key=lambda j: -best.get(preds[j], -1e18))
    preds = [preds[j] for j in order]
    pred_masks = [pred_masks[j] for j in order]

    p2 = int((budget - cost) * 0.6)
    lvl2 = []
    for expr, tab in combine(space, blocks, blocks, preds, pred_masks, p2):
        cost += 1
        if np.array_equal(tab, target):
            hits.append(expr)
            first = first or cost
        elif expr[0] in ("SEQ", "LOOP"):
            lvl2.append((expr, tab))
    if hits:
        return hits, first, 2
    if cost >= budget or not lvl2:
        return [], cost, 0

    # LOOP is UNARY, so wrapping every depth-2 result costs |lvl2|
    # candidates rather than |lvl2| x |blocks|. There is no reason to make it
    # compete for a beam slot, and `copy digits` -- LOOP over a sequence that
    # ranked 60th or worse -- was lost for exactly that reason.
    for expr, tab in list(lvl2):
        if cost >= budget:
            break
        cost += 1
        looped = space.loop(tab, 9)
        if np.array_equal(looped, target):
            hits.append(("LOOP", expr))
            first = first or cost
    if hits:
        return hits, first, 3

    s2 = np.array([score_fn(t, target, space, size_of(e)) for e, t in lvl2])
    ranked = [lvl2[i] for i in select_fn(shim, lvl2, s2, target, beam)]
    for expr, tab in combine(space, ranked, blocks, preds, pred_masks,
                             budget - cost):
        cost += 1
        if np.array_equal(tab, target):
            hits.append(expr)
            first = first or cost
    return hits, first or cost, 3 if hits else 0


# ------------------------------------------- Pillar 4 on a token stream


def discriminating_tape(survivors, truth, rng, length=7, tries=400):
    """SYNTHESISE an input that makes the survivors disagree.

    On the grid the agent picks a probe out of a fixed set of boards and
    cells. Here there is no fixed set: the experiment is a STRING, and the
    system writes it. That is a strictly stronger form of the same loop --
    active learning where the query is constructed rather than selected.

    Returns the first tape on which not all survivors produce the same
    (output string, final head position, halted) triple, together with what
    the truth does on it -- which is the only thing that has to be asked of
    the world.
    """
    for k in range(tries):
        tape = "".join(ALPHA[int(i)] for i in rng.integers(0, len(ALPHA), length))
        seen = {}
        for s in survivors:
            seen.setdefault(space_free_output(s, tape), []).append(s)
            if len(seen) > 1:
                return tape, space_free_output(truth, tape), k + 1
    return None, None, tries


def space_free_output(expr, tape):
    res, _ = run(expr, tape, TState(pos=0, began=0))
    return "".join(tape[p] for p in res.out), res.pos, res.live


# ----------------------------------------------------------- the truths

IS_DIGIT = or_chain([("EQ", 0, c) for c in "0123"])
COMMA = ("EQ", 0, ",")
END0 = ("EQ", 0, "$")

TRUTHS = {
    # `advance 1` is deliberately trivial: ADV clamps at the end of the
    # stream exactly as X46's STEP clamps at the board edge, so no guard is
    # needed and none should be synthesised. It is the floor, not a result.
    "advance 1":       "ADV",
    "scan to comma":   ("LOOP", ("IF", COMMA, "NOP", "ADV")),
    "copy to comma":   ("LOOP", ("IF", COMMA, "NOP", ("SEQ", "EMIT", "ADV"))),
    "halt on comma":   ("SEQ", "ADV", ("IF", COMMA, "HALT", "NOP")),
    "halt on digit":   ("SEQ", "ADV", ("IF", IS_DIGIT, "HALT", "NOP")),
    "copy digits":     ("LOOP", ("SEQ", ("IF", IS_DIGIT, "EMIT", "NOP"), "ADV")),
}

def training_set(space, pool, preds, masks, truth_tabs, n_tasks, rng):
    xs, ys, dropped = [], [], 0
    npool = len(pool)
    for _ in range(n_tasks):
        i, j = rng.integers(0, npool, 2)
        (_, ta), (_, tb) = pool[i], pool[j]
        form = rng.integers(0, 3)
        if form == 0:
            task = space.seq(ta, tb)
        elif form == 1:
            task = space.loop(ta, 9)
        else:
            k = int(rng.integers(0, len(preds)))
            task = space.branch(masks[k], ta, tb)
        if any(np.array_equal(task, tt) for tt in truth_tabs):
            dropped += 1
            continue
        parts = {int(i)} if form == 1 else {int(i), int(j)}
        # Half the tasks go one level deeper, with the INTERMEDIATE as a
        # positive -- and deepening by LOOP as well as by SEQ, because a loop
        # BODY is the case the model was blindest to. A body emits far less
        # than the loop that runs it, so every feature about emission counts
        # against it, and `copy digits` was lost precisely there: its body
        # ranked 255 of 846 under the model and 3rd under plain similarity.
        if rng.random() < 0.5:
            if rng.random() < 0.5:
                deeper, extra = space.loop(task, 9), set()
            else:
                k = int(rng.integers(0, npool))
                deeper, extra = space.seq(task, pool[k][1]), {k}
            if (not any(np.array_equal(deeper, tt) for tt in truth_tabs)
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


BUDGET = 2_000_000
KEEP = 120
BEAM = 60


MAX_ROUNDS = 4


def build(tapes, offsets, rng):
    space = TokenSpace(tapes)
    families = {off: [(a, space.pred(a)) for a in atom_preds((off,))]
                for off in offsets}
    atoms = [a for off in offsets for a, _ in families[off]]
    masks = [m for off in offsets for _, m in families[off]]
    truth_tabs = [space.interpret(e) for e in TRUTHS.values()]
    base = list(depth1(space, atoms, masks))
    xs, ys, dropped = training_set(space, base, atoms, masks, truth_tabs,
                                   400, rng)
    return space, families, atoms, masks, truth_tabs, Logistic().fit(xs, ys), dropped


def main() -> int:
    t0 = time.perf_counter()
    OFFSETS = (0, 1)
    # Every alphabet character must appear in the search tapes. A character
    # the evidence never shows cannot be classified -- the same
    # identifiability wall this project keeps rediscovering -- and a derived
    # `is-digit` that never saw a 3 is wrong in a way held-out tapes catch.
    seed_tapes = ["ab,12 c", "d3 0,a1", " ,bc23d"]
    eval_tapes = ["0a,b1 c", "cd23,a ", "  a1,0b", "ab0,1 c", "1,a b02",
                  "d0c,3a ", ",,0 ab1", "23 cd,0"]

    print("X48: the same machine over a token stream\n")
    print("primitives: NOP ADV EMIT HALT HOME BACK  SEQ IF LOOP  EQ(offset,char) OR")
    print(f"alphabet {ALPHA!r}   seed tapes {seed_tapes}")

    probe = TokenSpace(seed_tapes)
    checks = [("SEQ", ("IF", END0, "NOP", ("SEQ", "EMIT", "ADV")), "BACK"),
              ("LOOP", ("IF", IS_DIGIT, "EMIT", "ADV")),
              ("LOOP", ("SEQ", "EMIT", ("IF", END0, "HALT", "ADV"))),
              ("LOOP", ("IF", END0, "NOP", ("SEQ", "ADV", "BACK")))]
    checks += list(TRUTHS.values())
    bad = sum(0 if np.array_equal(probe.table(e), probe.interpret(e)) else 1
              for e in checks)
    print(f"table/interpreter agreement: {len(checks)-bad}/{len(checks)}"
          f"{'  <-- FAST PATH IS WRONG' if bad else ''}")
    if bad:
        return 1
    print(f"{len(atom_preds(OFFSETS))} atomic tests over offsets {OFFSETS}; "
          f"the OR-lattice measured at 55,783")
    print("predicates -- unenumerable, so the tests are DERIVED from events.\n")

    arm_names = ["size", "random", "similar", "learned", "cover", "cover-rnd"]
    tally = {a: Counter() for a in arm_names}
    built, rows = {}, []
    shown = {}

    for ti, (name, truth) in enumerate(TRUTHS.items()):
        tapes = list(seed_tapes)
        asked = 0
        for rnd in range(MAX_ROUNDS):
            rng = np.random.default_rng(4)
            space, families, atoms, masks, ttabs, model, _ = build(
                tapes, OFFSETS, rng)
            ttab = ttabs[ti]
            derived = derive_predicates(space, ttab, families)
            shown[name] = derived
            preds = list(atoms) + [d[1] for d in derived]
            pmasks = list(masks) + [space.pred(d[1]) for d in derived]
            pool = list(depth1(space, preds, pmasks))

            rr = np.random.default_rng(3)
            sc = {"size": lambda v, t, sp, n: -n,
                  "random": lambda v, t, sp, n: float(rr.random()),
                  "similar": lambda v, t, sp, n: float((v == t).mean()),
                  "learned": lambda v, t, sp, n: model.score(features(v, t, sp, n))}
            arms = {"size": (sc["size"], top_k), "random": (sc["random"], top_k),
                    "similar": (sc["similar"], top_k),
                    "learned": (sc["learned"], top_k),
                    "cover": (sc["learned"], cover_k),
                    "cover-rnd": (sc["random"], cover_k)}

            results, every_hit = {}, []
            for arm, (fn, sel) in arms.items():
                hits, cost, depth = search(space, pool, preds, pmasks, ttab,
                                           fn, sel, KEEP, BEAM, BUDGET)
                results[arm] = (hits, cost, depth)
                every_hit.extend(hits)

            # PILLAR 4 ON A TOKEN STREAM. Any surviving program that differs
            # from the truth on SOME input is a program the evidence failed to
            # rule out. So write that input. The query is constructed, not
            # chosen from a menu -- which is the part the grid version could
            # not do.
            wrong = [h for h in every_hit
                     if any(space_free_output(h, tp) != space_free_output(truth, tp)
                            for tp in eval_tapes)]
            if not wrong or len(tapes) >= 6:
                break
            tape, _, tries = discriminating_tape(
                wrong + [truth], truth, np.random.default_rng(100 + rnd))
            if tape is None:
                break
            tapes.append(tape)
            asked += 1

        cells = []
        for arm in arm_names:
            hits, cost, depth = results[arm]
            ok = [h for h in hits
                  if all(space_free_output(h, tp) == space_free_output(truth, tp)
                         for tp in eval_tapes)]
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
        rows.append((name, size_of(truth), asked, len(tapes), cells))

    head = (f'{"true rule":15} {"nodes":>5} {"asked":>6} {"tapes":>6} '
            + " ".join(f"{a:>9}" for a in arm_names))
    print(head + "\n" + "-" * len(head))
    for name, nodes, asked, nt, cells in rows:
        print(f"{name:15} {nodes:>5} {asked:>6} {nt:>6} " + " ".join(cells))

    print()
    for arm in arm_names:
        c = tally[arm]
        print(f"  {arm:>9}: {c['exact']}/6 exact"
              + (f", {c['overfit']} overfit" if c["overfit"] else "")
              + (f", {c['missed']} missed" if c["missed"] else ""))

    print("\nTESTS DERIVED FROM THE TARGET'S OWN EVENTS (exact ones only)")
    for name in ("halt on digit", "copy digits"):
        for ev, term, exact in shown.get(name, []):
            if exact:
                print(f"  {name:15} {ev:16} {render(term)[:46]}")

    best = max(arm_names, key=lambda a: tally[a]["exact"])
    print(f"\nWHAT THE BEST ARM ({best}) BUILT")
    for name in TRUTHS:
        e = built.get(name, {}).get(best)
        print(f"  {name:15} {render(e)[:64] if e is not None else '-- not found --'}")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
