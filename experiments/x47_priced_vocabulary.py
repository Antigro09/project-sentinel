"""X47: break the 15-node wall by re-PRICING the vocabulary, not enlarging it.

X46 unified motion and hazards into one substrate and paid for it: the
composite `step + proximity` is 15 nodes, and enumeration by size cannot get
there. Measured, not asserted --

    size <= 7      8,539 programs     2.2s
    size <= 8     27,424 programs     8.0s
    size <= 9    124,764 programs    45.0s
    size <= 10   606,807 programs   238.6s

a factor of ~5 per node, so size 15 is ~2e9 programs and ~200 hours. Blind
enumeration here is not slow, it is impossible.

A CORRECTION TO X46's DIAGNOSIS, found by measuring the pool instead of
believing the story. X46 said the proximity rule was out of reach because
`near` was too deep to build. It is not. The predicate lattice CLOSES under
OR at 403 truth vectors (12, 70, 200, 335, 396, 403, 403) and `near` is in
the pool the whole time. What is out of reach is the PROGRAM, because size
charges 7 nodes for a predicate that is one element of a finite lattice. The
wall was never expressiveness. It was accounting.

Size is a prior, not a law. Bottom-up enumeration explores in order of a
weight, and the weight merely happens to be node count. Re-price the
building blocks and the same enumerator, at the same cost, reaches different
programs:

    pass 1   every depth-1 composition over the closed lattice -- 8,085
             blocks, identical for every arm
    price    score each against the target's behaviour, keep 400
    pass 2+  compose the kept blocks under a fixed candidate budget

`step + proximity` is then depth 2 -- a SEQ of two depth-1 blocks. The
question stops being "can it be built" and becomes "can the scorer find two
needles in 8,085 straws". That is selection, and selection is what a learned
model is for.

MEASURED (5,000,000 candidate tables per arm per target, keep 400, beam 60;
474s total). Cost is at FIRST hit; every hit is re-verified with X46's own
interpreter on 7x7 boards the search never saw:

  true rule        nodes    size   random  similar  learned   cover  cover-rnd
  step 1               4      0k       0k       0k       0k      0k       0k
  slide                5    172k       --     172k     172k    172k     172k
  step + kill          9     90k       --       --      36k      8k       --
  step + pushback      9     90k       --       --     212k    212k       --
  step + proximity    15      --       --       --       9k      8k       --
  slide + kill        10      --       --    3004k       --   3003k       --
  ------------------------------------------------------------------------
                            4/6      1/6      3/6      5/6     6/6      2/6

    step + proximity   (SEQ (IF WALK@AHEAD STEP NOP)
                            (IF (HAZ@N|(HAZ@S|(HAZ@E|HAZ@W))) DIE NOP))

recovered at 9,000 candidates against the ~2e9 that size-first enumeration
would need. The wall is not a wall once the vocabulary is priced.

WHY THE RANDOM ARM MATTERS. It scores 1/6 -- the one target that needs no
composition at all. Selection of 400 from 8,085 by chance does not find the
pairs, so the ordering is doing the work rather than the budget or the pool.
An experiment where every arm succeeds has measured nothing.

TWO LEVERS, BOTH LOAD-BEARING, AND THE CONTROL THAT SEPARATES THEM.

  learned 5/6 vs similar 3/6.  Ranking blocks by resemblance to the target
  is the obvious heuristic and it is beaten, because the component that
  explains the DEATHS does not resemble a target that mostly moves. The
  rank diagnostic is blunt: for `step + proximity`, similarity puts the
  motion block at rank 2 and the hazard block at 1641 of 8,085; the trained
  model puts them at 18 and 169. Its heaviest learned feature is
  "dies where the target dies" (+1.34), not "looks like the target".

  cover 6/6 vs learned 5/6.  A leaderboard of 400 is still dozens of
  near-identical motion blocks, because motion is most of what the target
  does. Greedy coverage asks a different question at each pick -- which
  block agrees where nothing chosen so far agrees -- so once motion is
  covered, the next pick is forced to be about dying.

  cover-rnd 2/6.  THE CONTROL. Same coverage objective, random tie-break.
  Coverage saturates after a few dozen picks and the remaining slots are
  filled by score; filling them at random costs four targets. So the claim
  is not "coverage works" and not "the model works" -- it is that coverage
  chooses the basis and the model fills the rest, and removing either one
  is measurable.

WHAT THIS DOES NOT SHOW.

  Six targets from one generator. A 6/6 says this pricing handles THESE
  compositions, not that it handles composition.

  The proposer is a 13-feature logistic regression, deliberately -- small
  enough that a win cannot be credited to capacity. It is NOT the 1.4M
  transformer core, which reads grid episodes and emits rule-set labels and
  has no interface to program terms. Retrofitting it is the next question,
  not a settled one.

  The depth-2 IF sweep is truncated at roughly 4.5% of its space by the
  budget, identically for every arm, so IF-form results depend on predicate
  order. Predicates are therefore priced by the arm too -- a predicate is
  worth what the best block built on it is worth -- rather than inheriting
  lattice order.

  The depth-3 beam is drawn from SEQ and LOOP results only; an IF-form
  intermediate cannot be carried to depth 3. `slide + kill` is reachable
  anyway because SEQ(slide, kill) exists, but a rule that is only expressible
  as a LOOP over an IF-intermediate is out of reach here.

A BUG THIS FILE FOUND IN ITSELF. The table fast path composes behaviours by
array indexing instead of re-running X46's interpreter -- a second
implementation, so a second chance to be wrong, so a self-check against the
first. It failed 11 of 60 immediately: `LOOP` ran its body one extra time,
because starting at `out = a` is already the first pass. Programs that settle
hid it -- a fixed point is a fixed point however often you reach it -- but a
body that OSCILLATES, STEP then UNDO, lands elsewhere on an odd pass than an
even one. 401 of 5,000 situations disagreed. tests/test_unified_substrate.py
now pins it.
"""

from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x46_unified_substrate as X
from x46_unified_substrate import DIRS, Board, State, make_board, render, size_of

DEAD = -1
rng_global = np.random.default_rng(0)


# ------------------------------------------------------- situation tables
#
# A program's whole behaviour is a lookup table: situation -> situation (or
# DEAD). That works because no primitive changes `began` or `dir`, so a
# situation is (board, pos, began, dir) and every primitive maps one to
# another with the last two fixed. Composition then becomes array indexing:
#
#     SEQ(a, b)[s]    = b[a[s]]
#     IF(p, a, b)[s]  = a[s] where p else b[s]
#     LOOP(a)         = iterate until fixed
#
# which is ~20x faster than re-running the interpreter per candidate. It is
# an ACCELERATOR ONLY: every survivor reported at the end is re-checked with
# X46's interpreter on boards the search never saw.


def build_situations(boards):
    sits, index = [], {}
    for bi, b in enumerate(boards):
        cells = [(x, y) for y in range(b.size) for x in range(b.size)]
        for pos in cells:
            for began in cells:
                for aid in DIRS:
                    index[(bi, pos, began, aid)] = len(sits)
                    sits.append((bi, pos, began, aid))
    return sits, index


class Space:
    def __init__(self, boards):
        self.boards = boards
        self.sits, self.index = build_situations(boards)
        self.n = len(self.sits)
        self.ident = np.arange(self.n, dtype=np.int32)
        self.atoms = {name: self._atom(name) for name in X.ACTIONS}

    def _atom(self, name: str) -> np.ndarray:
        out = np.empty(self.n, dtype=np.int32)
        for i, (bi, pos, began, aid) in enumerate(self.sits):
            b = self.boards[bi]
            if name == "NOP":
                out[i] = i
            elif name == "DIE":
                out[i] = DEAD
            elif name == "HOME":
                out[i] = self.index[(bi, b.start, began, aid)]
            elif name == "UNDO":
                out[i] = self.index[(bi, began, began, aid)]
            else:  # STEP -- moves unconditionally, only the board stops it
                dx, dy = DIRS[aid]
                cell = (pos[0] + dx, pos[1] + dy)
                ok = 0 <= cell[0] < b.size and 0 <= cell[1] < b.size
                out[i] = self.index[(bi, cell if ok else pos, began, aid)]
        return out

    def pred(self, p) -> np.ndarray:
        out = np.empty(self.n, dtype=bool)
        for i, (bi, pos, began, aid) in enumerate(self.sits):
            out[i] = X.test(p, self.boards[bi],
                            State(pos=pos, began=began, dir=DIRS[aid]))
        return out

    def seq(self, a, b):
        out = np.where(a == DEAD, DEAD, b[np.where(a == DEAD, 0, a)])
        return out.astype(np.int32)

    def loop(self, a, passes):
        # X46 runs the body at most `board.size + 1` times TOTAL. Starting
        # from `out = a` is already the first pass, so only passes-1 remain.
        # Getting this off by one is not cosmetic: a body that oscillates --
        # STEP then UNDO -- lands somewhere different on an odd pass than on
        # an even one, and the self-check caught it on exactly that shape.
        out = a.copy()
        for _ in range(passes - 1):
            nxt = self.seq(out, a)
            if np.array_equal(nxt, out):
                break
            out = nxt
        return out

    def table(self, expr) -> np.ndarray:
        """Behaviour of any expression, by composition."""
        if isinstance(expr, str):
            return self.atoms[expr]
        head = expr[0]
        if head == "SEQ":
            return self.seq(self.table(expr[1]), self.table(expr[2]))
        if head == "LOOP":
            return self.loop(self.table(expr[1]),
                             max(b.size for b in self.boards) + 1)
        p = self.pred(expr[1])
        return np.where(p, self.table(expr[2]), self.table(expr[3])).astype(np.int32)

    def interpret(self, expr) -> np.ndarray:
        """The same thing via X46's interpreter -- the trusted path."""
        out = np.empty(self.n, dtype=np.int32)
        for i, (bi, pos, began, aid) in enumerate(self.sits):
            b = self.boards[bi]
            st = State(pos=pos, began=began, dir=DIRS[aid])
            res, _ = X.run(expr, b, st)
            out[i] = DEAD if not res.alive else self.index[(bi, res.pos, began, aid)]
        return out


# ------------------------------------------------------------- features
#
# What a scorer is allowed to see: the candidate's behaviour and the
# target's behaviour, nothing else. No rule names, no axis labels, no hint
# about which domain the task came from.


def features(v: np.ndarray, t: np.ndarray, ident: np.ndarray, size: int) -> np.ndarray:
    v_dead, t_dead = v == DEAD, t == DEAD
    v_stay, t_stay = v == ident, t == ident
    t_move = ~t_dead & ~t_stay

    def frac(mask, sub):
        k = mask.sum()
        return float(sub[mask].mean()) if k else 0.0

    return np.array([
        float((v == t).mean()),
        float((v_dead == t_dead).mean()),
        float(v_dead.mean()),
        float(t_dead.mean()),
        float(v_stay.mean()),
        float(t_stay.mean()),
        frac(t_move, v == t),
        frac(t_dead, v_dead),
        frac(t_stay, v_stay),
        float((v_dead & ~t_dead).mean()),     # kills where truth does not
        float((~v_dead & t_dead).mean()),     # spares where truth kills
        float((~v_dead & ~v_stay & t_move).mean()),
        size / 16.0,
    ], dtype=np.float64)


NFEAT = 13


class Logistic:
    """A deliberately small model: 13 features, one weight each.

    Small enough that a win cannot be attributed to capacity, and that the
    learned prices can be read off and argued with.
    """

    def __init__(self):
        self.w = np.zeros(NFEAT + 1)
        self.mu = np.zeros(NFEAT)
        self.sd = np.ones(NFEAT)

    def fit(self, xs, ys, epochs=300, lr=0.5):
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        self.mu, self.sd = xs.mean(0), xs.std(0) + 1e-9
        z = np.hstack([(xs - self.mu) / self.sd, np.ones((len(xs), 1))])
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-z @ self.w))
            self.w -= lr * (z.T @ (p - ys)) / len(ys)
        return self

    def score(self, x):
        z = np.append((x - self.mu) / self.sd, 1.0)
        return float(1.0 / (1.0 + np.exp(-z @ self.w)))


# ----------------------------------------------------------- the pool

def depth1(space: Space, preds, pred_tabs):
    """Every depth-1 composition, with the predicate lattice priced as units.

    This is the move that makes the wall an accounting question. A predicate
    is one element of a CLOSED 334-element lattice, so charging its node
    count -- 7 for `near` -- prices a lattice lookup as if it were a tree
    that had to be discovered. Here each predicate costs one.
    """
    for a in X.ACTIONS:
        yield ("LOOP", a), space.loop(space.atoms[a], 6)
    for a in X.ACTIONS:
        for b in X.ACTIONS:
            if a != b:
                yield ("SEQ", a, b), space.seq(space.atoms[a], space.atoms[b])
    for p, pt in zip(preds, pred_tabs):
        for a in X.ACTIONS:
            for b in X.ACTIONS:
                if a == b:
                    continue
                yield (("IF", p, a, b),
                       np.where(pt, space.atoms[a], space.atoms[b]).astype(np.int32))


def combine(space: Space, left, right, preds, pred_tabs, budget):
    """Every way to put two priced blocks together. Yields (expr, table)."""
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
        yield ("LOOP", ea), space.loop(ta, 6)
    for p, pt in zip(preds, pred_tabs):
        for ea, ta in left:
            for eb, tb in right:
                if used >= budget:
                    return
                used += 1
                yield ("IF", p, ea, eb), np.where(pt, ta, tb).astype(np.int32)


# ------------------------------------------------------------- training
#
# The model is trained on RANDOM tasks from the same substrate, never on the
# six evaluation targets. Any random task whose behaviour matches one of the
# six is thrown away and counted. Sharing sub-structure with the targets is
# not leakage -- it is the entire thing being tested, since a proposer that
# cannot transfer a building block has learned nothing.


def training_set(space, pool, preds, pred_tabs, truth_tabs, n_tasks, rng):
    xs, ys, dropped = [], [], 0
    npool = len(pool)
    for _ in range(n_tasks):
        i, j = rng.integers(0, npool, 2)
        (ea, ta), (eb, tb) = pool[i], pool[j]
        form = rng.integers(0, 3)
        if form == 0:
            task = space.seq(ta, tb)
        elif form == 1:
            task = space.loop(ta, 6)
        else:
            k = int(rng.integers(0, len(preds)))
            task = np.where(pred_tabs[k], ta, tb).astype(np.int32)
        if any(np.array_equal(task, tt) for tt in truth_tabs):
            dropped += 1
            continue
        parts = {i} if form == 1 else {i, j}
        # Half the tasks go one level deeper, with the INTERMEDIATE as a
        # positive. Without this the model only ever sees depth-1 blocks and
        # is guessing when it ranks a partially built program -- which is
        # exactly where `slide + kill` was lost: `slide` itself failed to make
        # the depth-2 beam.
        if rng.random() < 0.5:
            k = int(rng.integers(0, npool))
            deeper = space.seq(task, pool[k][1])
            if not any(np.array_equal(deeper, tt) for tt in truth_tabs):
                xs.append(features(task, deeper, space.ident, 8))
                ys.append(1.0)
                for idx in parts | {k}:
                    e, t = pool[idx]
                    xs.append(features(t, deeper, space.ident, size_of(e)))
                    ys.append(1.0)
                for idx in rng.integers(0, npool, 6):
                    e, t = pool[int(idx)]
                    xs.append(features(t, deeper, space.ident, size_of(e)))
                    ys.append(0.0)
        for idx in parts:
            e, t = pool[idx]
            xs.append(features(t, task, space.ident, size_of(e)))
            ys.append(1.0)
        for idx in rng.integers(0, npool, 8):
            if int(idx) in parts:
                continue
            e, t = pool[int(idx)]
            xs.append(features(t, task, space.ident, size_of(e)))
            ys.append(0.0)
    return np.array(xs), np.array(ys), dropped


# ---------------------------------------------------------------- arms


def make_scorers(model, rng):
    def size_score(v, t, ident, size):
        return -size

    def random_score(v, t, ident, size):
        return float(rng.random())

    def similar_score(v, t, ident, size):
        return float((v == t).mean())

    def learned_score(v, t, ident, size):
        return model.score(features(v, t, ident, size))

    return {"size": size_score, "random": random_score,
            "similar": similar_score, "learned": learned_score}


def top_k(space, pool, scores, target, keep):
    return list(np.argsort(-scores)[:keep])


def cover_k(space, pool, scores, target, keep):
    """Pick a BASIS, not a leaderboard.

    Ranking every block against the whole target selects for blocks that
    resemble it, and the top of that list is dozens of near-identical motion
    blocks -- because motion is most of what the target does. The component
    that explains the DEATHS scores lower by construction and is crowded
    out, which is exactly what the rank diagnostic showed.

    Greedy coverage asks a different question at each step: which block
    agrees with the target where nothing chosen so far agrees? The first
    pick is a motion block, and the situations left over are the deaths --
    so the second pick is forced to be about dying. Same information as the
    other arms, different objective.
    """
    agree = np.empty((len(pool), space.n), dtype=np.float32)
    for i, (_, tab) in enumerate(pool):
        agree[i] = tab == target
    unc = np.ones(space.n, dtype=np.float32)
    tiny = 1e-4 * (scores - scores.min()) / (np.ptp(scores) + 1e-9)
    chosen, taken = [], np.zeros(len(pool), dtype=bool)
    for _ in range(keep):
        gains = agree @ unc
        gains[taken] = -1.0
        best = int(np.argmax(gains + tiny))
        if gains[best] <= 0:
            break
        taken[best] = True
        unc = unc * (1.0 - agree[best])
        chosen.append(best)
    if len(chosen) < keep:          # nothing left to cover: fall back to rank
        for i in np.argsort(-scores):
            if len(chosen) >= keep:
                break
            if not taken[i]:
                chosen.append(int(i))
    return chosen


def search(space, pool, preds, pred_tabs, target, score_fn, select_fn,
           keep, beam, budget):
    """Price the vocabulary, then compose.

    Cost is counted at the FIRST hit, not at the end of the pass. A pass is
    not abandoned when it finds something -- a later candidate may be the one
    that survives the held-out boards -- but the effort number has to mean
    "what it took to get there".
    """
    cost, first, hits = 0, 0, []

    scores = np.empty(len(pool))
    for i, (expr, tab) in enumerate(pool):
        cost += 1
        if np.array_equal(tab, target):
            hits.append(expr)
            first = first or cost
        scores[i] = score_fn(tab, target, space.ident, size_of(expr))
    if hits:
        return hits, first, 1

    blocks = [pool[i] for i in select_fn(space, pool, scores, target, keep)]
    blocks += [(a, space.atoms[a]) for a in X.ACTIONS]

    # Price the PREDICATES too, on the same yardstick: a predicate is worth
    # what the best block built on it is worth. Without this the IF sweep runs
    # in lattice order -- fewest leaves first -- which is a size prior nobody
    # chose and which happens to front-load exactly the useful single-cell
    # tests. Every arm should have to earn that ordering rather than inherit
    # it, especially since the sweep is truncated long before it finishes.
    best = {}
    for i, (expr, _) in enumerate(pool):
        if expr[0] == "IF" and scores[i] > best.get(expr[1], -1e18):
            best[expr[1]] = scores[i]
    order = sorted(range(len(preds)), key=lambda j: -best.get(preds[j], -1e18))
    preds = [preds[j] for j in order]
    pred_tabs = [pred_tabs[j] for j in order]

    # Split the remaining budget between the two composition passes. Without
    # this the depth-2 IF sweep swallows everything and depth 3 never runs --
    # which would silently mean `slide + kill` was never attempted, not that
    # it was attempted and missed.
    p2 = int((budget - cost) * PASS2_SHARE)

    lvl2 = []
    for expr, tab in combine(space, blocks, blocks, preds, pred_tabs, p2):
        cost += 1
        if np.array_equal(tab, target):
            hits.append(expr)
            first = first or cost
        elif expr[0] in ("SEQ", "LOOP"):
            lvl2.append((expr, tab))
    if hits:
        return hits, first, 2
    if cost >= budget:
        return [], cost, 0

    # The beam uses the arm's SELECTOR, not a fixed top-k. An arm whose whole
    # claim is that ranking crowds out rare-event components should not be
    # forced back into ranking one level up.
    lvl2_scores = np.array([score_fn(t, target, space.ident, size_of(e))
                            for e, t in lvl2])
    ranked = [lvl2[i] for i in select_fn(space, lvl2, lvl2_scores, target, beam)]
    for expr, tab in combine(space, ranked, blocks, preds, pred_tabs, budget - cost):
        cost += 1
        if np.array_equal(tab, target):
            hits.append(expr)
            first = first or cost
    return hits, first or cost, 3 if hits else 0


BUDGET = 5_000_000
KEEP = 400          # priced blocks kept out of the 8,085-block depth-1 pool
BEAM = 60           # depth-2 blocks carried into depth 3
PASS2_SHARE = 0.60  # so depth 3 is genuinely reached, not merely declared


def main() -> int:
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)

    search_boards = [make_board(11, 5), make_board(12, 5)]
    X.PROBE_BOARDS = search_boards
    space = Space(search_boards)
    eval_boards = [make_board(s) for s in range(1, 3)]

    print("X47: breaking the 15-node wall by pricing, not by enlarging\n")
    preds = X.enumerate_preds(6)
    pred_tabs = [space.pred(p) for p in preds]
    print(f"predicate lattice: {len(preds)} truth vectors (CLOSED under OR)")
    print(f"situations: {space.n:,}  ({len(search_boards)} boards, "
          f"pos x began x dir)")

    # The fast path must equal the trusted path, or every number below is
    # about a different machine than X46's.
    bad = 0
    for _ in range(60):
        p = preds[int(rng.integers(0, len(preds)))]
        a, b = (X.ACTIONS[int(rng.integers(0, 5))] for _ in range(2))
        e = ("SEQ", ("IF", p, a, b), ("LOOP", ("IF", p, b, a)))
        if not np.array_equal(space.table(e), space.interpret(e)):
            bad += 1
    print(f"table/interpreter agreement: {60-bad}/60"
          f"{'  <-- FAST PATH IS WRONG' if bad else ''}")
    if bad:
        return 1

    t = time.perf_counter()
    pool = list(depth1(space, preds, pred_tabs))
    print(f"depth-1 pool: {len(pool):,} blocks ({time.perf_counter()-t:.1f}s)\n")

    truths = list(X.TRUTHS.items())
    truth_tabs = [space.interpret(e) for _, e in truths]

    t = time.perf_counter()
    xs, ys, dropped = training_set(space, pool, preds, pred_tabs, truth_tabs,
                                   n_tasks=500, rng=rng)
    model = Logistic().fit(xs, ys)
    print(f"proposer trained on {len(xs):,} examples from random tasks "
          f"({time.perf_counter()-t:.1f}s)")
    print(f"  {dropped} random tasks discarded for matching an evaluation "
          f"target's behaviour")
    order = np.argsort(-np.abs(model.w[:NFEAT]))
    names = ["exact", "dead-agree", "v-dead", "t-dead", "v-stay", "t-stay",
             "on-moves", "on-deaths", "on-stays", "over-kill", "under-kill",
             "both-move", "size"]
    top = "  ".join(f"{names[i]}{model.w[i]:+.2f}" for i in order[:5])
    print(f"  heaviest features: {top}\n")

    sc = make_scorers(model, np.random.default_rng(3))
    arms = {
        "size":    (sc["size"], top_k),
        "random":  (sc["random"], top_k),
        "similar": (sc["similar"], top_k),
        "learned": (sc["learned"], top_k),
        "cover":   (sc["learned"], cover_k),
        # Does coverage need the model at all, or is the objective doing all
        # the work? Same selector, random tie-break. If this also scores 6/6
        # the honest claim is about coverage, not about a trained proposer.
        "cover-rnd": (sc["random"], cover_k),
    }
    print(f"equal budget: {BUDGET:,} candidate tables per arm per target; "
          f"keep {KEEP}, beam {BEAM}\n")

    header = f'{"true rule":18} {"nodes":>5} ' + \
        " ".join(f"{a:>10}" for a in arms)
    print(header)
    print("-" * len(header))

    tally = {a: Counter() for a in arms}
    found_expr = {}
    for (name, truth), ttab in zip(truths, truth_tabs):
        cells = []
        for arm, (fn, sel) in arms.items():
            hits, cost, depth = search(space, pool, preds, pred_tabs, ttab,
                                       fn, sel, KEEP, BEAM, BUDGET)
            if not hits:
                tally[arm]["missed"] += 1
                cells.append(f'{"--":>10}')
                continue
            # Held-out check with X46's own interpreter, on 7x7 boards the
            # search never saw.
            ok = [h for h in hits if all(
                X.transition(h, b, (x, y), a) == X.transition(truth, b, (x, y), a)
                for b in eval_boards
                for y in range(b.size) for x in range(b.size) for a in DIRS)]
            if ok:
                tally[arm]["exact"] += 1
                found_expr.setdefault(name, {})[arm] = min(ok, key=size_of)
                cells.append(f"{'d'+str(depth)+' '+f'{cost/1000:.0f}k':>10}")
            else:
                tally[arm]["overfit"] += 1
                cells.append(f'{"OVERFIT":>10}')
        print(f"{name:18} {size_of(truth):>5} " + " ".join(cells))

    print()
    for arm in arms:
        c = tally[arm]
        print(f"  {arm:>8}: {c['exact']}/6 exact"
              + (f", {c['overfit']} overfit" if c['overfit'] else "")
              + (f", {c['missed']} missed" if c['missed'] else ""))

    print("\nWHAT THE BEST ARM ACTUALLY BUILT")
    best_arm = max(arms, key=lambda a: tally[a]["exact"])
    print(f"  (arm: {best_arm})")
    for name in X.TRUTHS:
        e = found_expr.get(name, {}).get(best_arm)
        print(f"  {name:18} {render(e)[:68] if e is not None else '-- not found --'}")

    print("\nREADING")
    ex = {a: tally[a]["exact"] for a in arms}
    heur = max(ex["size"], ex["similar"])
    if ex["random"] >= max(ex["learned"], ex["cover"]):
        print("  the RANDOM arm matches the best learned arm. The pricing is")
        print("  decoration: the depth-1 pool and the budget are doing the")
        print("  work, not the scorer. Do not claim guidance.")
    elif max(ex["learned"], ex["cover"]) > heur:
        print(f"  learned {ex['learned']}/6 and cover {ex['cover']}/6 beat the")
        print(f"  heuristics (similar {ex['similar']}/6, size {ex['size']}/6) and")
        print(f"  the calibration arm (random {ex['random']}/6) at an identical")
        print("  budget. Same enumerator, same pool, same cost -- only the")
        print("  ORDER changes, and the order is what the model supplies.")
    else:
        print(f"  the trained proposer (learned {ex['learned']}/6, cover "
              f"{ex['cover']}/6) does not beat")
        print(f"  a one-line heuristic (similar {ex['similar']}/6, size "
              f"{ex['size']}/6). Report the heuristic.")
    if ex["cover"] > ex["learned"]:
        print(f"\n  cover {ex['cover']}/6 > learned {ex['learned']}/6: ranking every block")
        print("  against the whole target crowds out the component that")
        print("  explains the rare events. Selection has to build a basis.")
    print(f"\ntotal {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
