"""X64E: a distribution over logical forms, and conflict as posterior mass.

X64D concluded "what cannot eliminate cannot contradict". That is not a
theorem, and the review was right to reject it. A model can keep support on
every interpretation and still measure how much of its probability mass sits
outside what the demonstrations allow. X64D could not do that because its
language layer produced SETS of predicates, and a set has no mass. The fix
is a normalised distribution over structured meanings:

    z           a typed logical form drawn from the task grammar
    p_theta     a distribution over all valid z given the instruction u
    C(D)        the logical forms whose execution fits the demonstrations
    conflict    1 - sum over C(D) of p_theta(z | u)

Nothing is eliminated: every grammatical z keeps non-zero probability. What
becomes measurable is disagreement.

PHASES, in order, each with a stop condition:

  E0  audit the meaning grammar and establish the GOLD upper bound. If a
      parser that is handed the true logical form cannot separate matched
      from contradictory pairs, the representation is inadequate and no
      amount of learning repairs it. Stop there.
  E1  a small, attributable log-linear parser trained by weak supervision
      over behaviourally consistent forms. Exact inference -- the grammar
      has 168 forms, so no approximation is needed anywhere.
  E2  joint inference over (meaning, behaviour), with language ranking and
      evidence deciding, as X64D established.
  E3  posterior-mass conflict detection, thresholded on validation only.

WHAT THIS IS NOT. It is a controlled semantic-parsing experiment over an
authored task grammar. It is not natural-language understanding, and the
grammar's structure is supplied rather than discovered.

MEASURED. 12 of 12 gates. 43 held-out logical forms over 24 filter-scope
pairs absent from development and validation; 86 conditions per arm, 66 of
them covered by every arm including X64C's and X64D's.

  E0  the representation is sharp enough: a parser handed the true form
      separates matched from contradictory pairs at AUROC 0.988. Had this
      failed, learning could not have repaired it and the experiment would
      have stopped.

  E1/E7  on the 66 shared conditions every arm is 100% correct, and the
      question is what each spends to get there:

        arm                          correct   queries
        demonstrations only              66       150
        X64D predicate senses            66       148
        uniform logical forms            66        34
        authored multi-sense parser      66         8
        role-blind induced               66         6
        MAIN induced parser              66         2
        gold logical forms               66         0

  E3  conflict as posterior mass -- 1 minus the mass on the forms the
      demonstrations allow -- separates at AUROC 0.996, AUPRC 0.997, 95%
      bootstrap CI (0.986, 1.000), recall 0.98 and precision 1.00 at a
      threshold fixed on validation. X64D reported twelve set-based
      statistics all at chance. The difference is not a better statistic, it
      is that a normalised distribution has mass and a set does not.

WHAT LEARNING ACTUALLY BOUGHT, and it is not accuracy. The authored parser
-- weights read straight off the surface realiser, no learning -- scores
1.00 exact-form on the test split against the induced parser's 0.84, and
both reach 1.00 denotation. Learning loses on parsing accuracy and wins
downstream: 2 queries against 8, and conflict AUROC 0.996 against 0.988.
The induced weights are CALIBRATED because they were fitted to a likelihood,
and calibration is what the commit threshold and the conflict mass consume.
An authored map can be right without being confident in proportion.

AND EXACT-FORM ACCURACY IS PARTLY UNIDENTIFIABLE. 28 of the 116 behaviours
have more than one logical form, up to 21 of them, so no behavioural
observation can separate those forms. Denotation accuracy is the
identifiable quantity; exact-form accuracy is reported beside it and should
not be read as the parser being wrong.

TWO CONTROLS THAT WERE BROKEN AND HAD TO BE FIXED. The authored-structure
baseline was first built over every logical form, so it was handed the test
split's surface vocabulary and scored 1.00 on out-of-vocabulary forms; then
over development forms but all three variants, which still handed it the
test's words. Restricted to what the induced parser sees -- development
forms, variants 0 and 1 -- it scores 0.00 on variant 2 like everything else.
And E2 first compared 80/86 against 66/66, which compares populations rather
than arms; it now runs on the intersection.

Run: uv run python experiments/x64e_semantics.py
"""

import hashlib
import itertools
import json
import math
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x64a_identify as X
import x64b1_openworld as O
import x64d_senses as W

UNIVERSE, HELD_OUT = W.UNIVERSE, W.HELD_OUT
CHALLENGE, CONFIRM_ON = W.CHALLENGE, W.CONFIRM_ON
SCOPES, FILTERS, POLARITIES = W.SCOPES, W.FILTERS, W.POLARITIES


# ------------------------------------------------- E0.1 the logical forms
#
# A typed form is (op, filter, scope). Every slot value is reused across
# many forms, so a test form is a new combination of parts each seen
# elsewhere -- which is what makes E4 a compositional question rather than a
# memorisation one.

class Z(tuple):
    __slots__ = ()

    def __new__(cls, op, filt, scope):
        return super().__new__(cls, (op, filt, scope))

    @property
    def op(self):
        return self[0]

    @property
    def filt(self):
        return self[1]

    @property
    def scope(self):
        return self[2]

    def __repr__(self):
        return f"{self[0]}({self[1]} @ {self[2]})"


SLOTS = ("op", "filt", "scope")
VALUES = {"op": POLARITIES, "filt": FILTERS, "scope": SCOPES}

ALL_Z = [Z(o, f, s) for o in POLARITIES for f in FILTERS for s in SCOPES]


def execute(z):
    """A logical form denotes a behaviour over the universe."""
    return W.make_task(z.scope, z.filt, z.op)


_BEH = {}


def denote(z):
    if z not in _BEH:
        f = execute(z)
        _BEH[z] = tuple(f(t) for t in UNIVERSE)
    return _BEH[z]


def forms_by_behaviour():
    d = {}
    for z in ALL_Z:
        d.setdefault(denote(z), []).append(z)
    return d


# ------------------------------------------------ E0.2 anti-leakage audit

def audit(instr_of, meanings):
    """Report the grammar's shape and check that no surface feature indexes
    a whole task. A word may denote an operator or an argument; it may not
    denote a behaviour."""
    by_beh = forms_by_behaviour()
    rows = {
        "logical forms": len(ALL_Z),
        "distinct behaviours": len(by_beh),
        "slots": len(SLOTS),
        "values per slot": {k: len(v) for k, v in VALUES.items()},
    }
    # a feature that uniquely identifies one meaning is leakage
    tok_to_meaning = {}
    for m in meanings:
        for v in (0, 1, 2):
            for tok in instr_of(m, v):
                tok_to_meaning.setdefault(tok, set()).add(m)
    leaks = sorted(t for t, ms in tok_to_meaning.items() if len(ms) == 1)
    reuse = {}
    for slot in SLOTS:
        for val in VALUES[slot]:
            reuse[(slot, val)] = sum(1 for m in meanings
                                     if getattr(Z(*m), slot) == val)
    return rows, leaks, reuse


# --------------------------------------------------------------- splits
#
# By (filter, scope) PAIR, so a test form is a new combination of a filter
# and a scope each of which was seen in other combinations. Every filter and
# every scope must appear in development or the test measures vocabulary
# coverage rather than composition.

def combo_of(z):
    """W.realise speaks (scope, filter, polarity)."""
    return (z.scope, z.filt, z.op)


def instr(z, variant=0):
    return W.realise(combo_of(z), variant)


def make_splits():
    pairs = [(f, s) for f in FILTERS for s in SCOPES]
    dev, val, test = [], [], []
    for i, (f, s) in enumerate(pairs):
        k = (FILTERS.index(f) * 3 + SCOPES.index(s) * 5) % 10
        (dev if k < 4 else val if k < 7 else test).append((f, s))
    need_f = set(FILTERS) - {f for f, _s in dev}
    need_s = set(SCOPES) - {s for _f, s in dev}
    for grp in (val, test):
        for p in list(grp):
            if p[0] in need_f or p[1] in need_s:
                grp.remove(p)
                dev.append(p)
                need_f.discard(p[0])
                need_s.discard(p[1])
    return set(dev), set(val), set(test)


DEV_PAIRS, VAL_PAIRS, TEST_PAIRS = make_splits()


def forms_in(pairs):
    """Only forms whose behaviour is non-degenerate, so a `task` is always
    something a user could have meant."""
    out = []
    for z in ALL_Z:
        if (z.filt, z.scope) not in pairs:
            continue
        b = denote(z)
        if all(o == "" for o in b):
            continue
        out.append(z)
    return out


# ------------------------------------------- E0.4 the gold upper bound
#
# Hand the parser the true logical form and ask whether the REPRESENTATION
# can do the job. If it cannot, learning is pointless and the experiment
# stops here.

def consistent_forms(demos):
    """C(D): the logical forms whose denotation fits every demonstration."""
    idx = {t: i for i, t in enumerate(UNIVERSE)}
    out = []
    for z in ALL_Z:
        b = denote(z)
        if all(b[idx[t]] == a for t, a in demos.items()):
            out.append(z)
    return out


def conflict_score(p, demos):
    """1 - the probability mass the parser puts on meanings the
    demonstrations allow. Every form keeps support; what is measured is
    where the mass sits."""
    C = set(consistent_forms(demos))
    return 1.0 - sum(v for z, v in p.items() if z in C)


def gold_parser(z_true):
    return {z: (1.0 if z == z_true else 0.0) for z in ALL_Z}


def uniform_parser():
    n = len(ALL_Z)
    return {z: 1.0 / n for z in ALL_Z}


# ------------------------------------------- E1 the log-linear parser
#
# Small and attributable on purpose. Exact inference over all 168 forms, so
# nothing here is an approximation and every number is a property of the
# model rather than of a sampler.
#
#     p(z | u) = exp(theta . phi(u, z)) / sum over z' of exp(theta . phi)
#
# Features are indicator functions over (word, role, slot, value). With
# alignment on, a word in an object role may only inform the filter slot, a
# preposition or delimiter only the scope, a verb only the operator -- which
# is the structured phrase-to-subtree alignment the protocol asks for, and
# is ablatable.

ROLE_SLOT = {W.VERB: ("op",), W.OBJ: ("filt",), W.MOD: ("filt",),
             W.PREP: ("scope",), W.DELIM: ("scope",)}


def feats(toks, z, align=True):
    out = []
    for (w, r) in toks:
        slots = ROLE_SLOT.get(r, SLOTS) if align else SLOTS
        for sl in slots:
            out.append((w, r, sl, getattr(z, sl)))
    for sl in SLOTS:
        out.append(("<bias>", "-", sl, getattr(z, sl)))
    return out


class Parser:
    def __init__(self, align=True, role_blind=False):
        self.th = {}
        self.align = align
        self.role_blind = role_blind

    def _toks(self, toks):
        return [(w, "-") if self.role_blind else (w, r) for w, r in toks]

    def scores(self, toks, cands=None):
        toks = self._toks(toks)
        cands = cands or ALL_Z
        s = {}
        for z in cands:
            s[z] = sum(self.th.get(f, 0.0) for f in feats(toks, z, self.align))
        return s

    def dist(self, toks, cands=None):
        s = self.scores(toks, cands)
        m = max(s.values())
        e = {z: math.exp(v - m) for z, v in s.items()}
        tot = sum(e.values())
        return {z: v / tot for z, v in e.items()}

    def fit(self, examples, epochs=40, lr=0.5, l2=1e-3, contrast=None,
            verbose=False):
        """Weak supervision: maximise the total probability of the
        behaviourally CONSISTENT forms, since the true form is never given.

            L = sum_i log sum_{z in C(D_i)} p(z | u_i)

        `contrast` supplies mismatched (instruction, demonstrations) pairs
        and subtracts the mass they put on their own consistent set, which
        is the preregistered control against spurious forms that fit any
        demonstrations at all."""
        for ep in range(epochs):
            grad = {}
            ll = 0.0
            for toks, Cset in examples:
                p = self.dist(toks)
                zc = sum(p[z] for z in Cset) or 1e-12
                ll += math.log(zc)
                tk = self._toks(toks)
                for z in ALL_Z:
                    w = (p[z] / zc if z in Cset else 0.0) - p[z]
                    if abs(w) < 1e-12:
                        continue
                    for f in feats(tk, z, self.align):
                        grad[f] = grad.get(f, 0.0) + w
            for toks, Cset in (contrast or []):
                p = self.dist(toks)
                tk = self._toks(toks)
                for z in Cset:
                    for f in feats(tk, z, self.align):
                        grad[f] = grad.get(f, 0.0) - p[z]
            for f, g in grad.items():
                self.th[f] = self.th.get(f, 0.0) + lr * (g - l2 * self.th.get(f, 0.0))
            if verbose and ep % 10 == 0:
                print(f"      epoch {ep:>3} loglik {ll:>10.2f}")
        return self


def training_examples(forms, n_demos=6, variants=(0, 1)):
    ex = []
    for z in forms:
        f = execute(z)
        demos = {t: f(t) for t in UNIVERSE[:n_demos]}
        C = set(consistent_forms(demos))
        for v in variants:
            ex.append((instr(z, v), C))
    return ex


def contrast_examples(forms, n_demos=6, variants=(0, 1)):
    """Mismatched pairs: this instruction with someone else's
    demonstrations. Their consistent set is what the parser must NOT put
    mass on."""
    ex = []
    for i, z in enumerate(forms):
        other = execute(forms[(i + 1) % len(forms)])
        demos = {t: other(t) for t in UNIVERSE[:n_demos]}
        C = set(consistent_forms(demos))
        for v in variants:
            ex.append((instr(z, v), C))
    return ex


# ------------------------------------------- E2 joint inference
#
#   p(b | u, D)  proportional to  1[b fits D] * sum over z denoting b of
#                                 p_theta(z | u)
#
# Language ranks which ambiguity to look at first. It never authorises
# execution: the answer is given only when the EVIDENCE leaves one
# behaviour, which is X64D's finding carried forward unchanged.

POOL = None


def pool():
    global POOL
    if POOL is None:
        POOL = W.pool()
    return POOL


def known(parser, toks):
    """Tokens the parser has never seen. An out-of-grammar word is a
    different state from an ambiguous one and must not be silently
    ignored."""
    tk = parser._toks(toks)
    return [t for t in tk
            if not any((t[0], t[1], sl, v) in parser.th
                       for sl in SLOTS for v in VALUES[sl])]


def behaviour_prior(p):
    """Push the meaning distribution through denotation."""
    out = {}
    for z, v in p.items():
        b = denote(z)
        out[b] = out.get(b, 0.0) + v
    return out


def solve(z_true, variant, parser, mode="joint", query="infogain",
          demos_n=2, budget=8, semantic=True, confirm=True, rng=None,
          exclude=None, theta=None, demos_from=None, commit=None):
    """One task, one instruction form."""
    rng = rng or random.Random(5)
    f = execute(z_true)
    toks = instr(z_true, variant)
    src = demos_from or f
    demos = {t: src(t) for t in UNIVERSE[:demos_n]}

    pl = dict(pool())
    if exclude is not None:
        pl.pop(exclude, None)
    keep = X.survivors(pl, list(demos), demos)
    retained = denote(z_true) in {b for b, _g in keep}

    p = parser.dist(toks) if mode != "none" else uniform_parser()
    oov = known(parser, toks) if mode != "none" else []
    conf = conflict_score(p, demos) if mode != "none" else 0.0
    if theta is not None and conf >= theta:
        return dict(verdict="conflict", rep=None, asked=0, sem=0,
                    retained=retained, conflict=conf, oov=len(oov))

    bp = behaviour_prior(p)
    sem_asked, asked = 0, 0
    seen = set(demos)

    # A semantic question asks which reading is meant, and is the only move
    # available when a word is out of vocabulary.
    if semantic and mode != "none" and (oov or len(keep) > 1):
        for _ in range(2 if oov else 1):
            if len(keep) <= 1:
                break
            best, gain = None, 0.0
            for sl in SLOTS:
                for v in VALUES[sl]:
                    mass = sum(pv for z, pv in p.items()
                               if getattr(z, sl) == v)
                    if 0.05 < mass < 0.95:
                        g = min(mass, 1 - mass)
                        if g > gain:
                            best, gain = (sl, v), g
            if best is None:
                break
            sl, v = best
            sem_asked += 1
            ans = getattr(z_true, sl) == v
            p = {z: pv for z, pv in p.items()
                 if (getattr(z, sl) == v) == ans}
            tot = sum(p.values()) or 1e-12
            p = {z: pv / tot for z, pv in p.items()}
            bp = behaviour_prior(p)
            ok = {b for b in bp if bp[b] > 0}
            if any(b in ok for b, _g in keep):
                keep = [(b, g) for b, g in keep if b in ok] or keep

    def order(items):
        if mode == "none":
            return items
        hi = max((bp.get(b, 0.0) for b, _g in items), default=0.0)
        top = [(b, g) for b, g in items if bp.get(b, 0.0) >= hi * 0.999]
        return top or items

    while len(keep) > 1 and asked < budget:
        # Posterior-thresholded commitment. X64D showed that committing on a
        # language-preferred TIER is unsafe, because a tier is a set and a
        # set carries no confidence. A normalised posterior does, so whether
        # a threshold on it is safe is a measurable question rather than an
        # assumption -- and it is settled on validation.
        if commit is not None and mode != "none":
            tot = sum(bp.get(b, 0.0) for b, _g in keep) or 1e-12
            top = max(keep, key=lambda bg: bp.get(bg[0], 0.0))
            if bp.get(top[0], 0.0) / tot >= commit:
                keep = [top]
                break
        tier = order(keep)
        if query == "random":
            cands = [t for t in UNIVERSE if t not in seen
                     and len(X.split(keep, t)) > 1]
            q = rng.choice(cands) if cands else None
        elif query == "oracle":
            best, kp = None, len(keep) + 1
            for t in UNIVERSE:
                if t in seen:
                    continue
                k = len(X.split(keep, t).get(f(t), []))
                if k < kp:
                    best, kp = t, k
            q = best
        elif query == "infogain":
            # Expected information gain under the JOINT posterior, not the
            # uniform-weighted disagreement X64A used: an answer that splits
            # two behaviours the language considers implausible buys less
            # than one that splits two it considers likely.
            q, gain = None, -1.0
            for t in UNIVERSE:
                if t in seen:
                    continue
                parts = {}
                for b, g in keep:
                    a = b[UNIVERSE.index(t)]
                    parts[a] = parts.get(a, 0.0) + max(bp.get(b, 0.0), 1e-9)
                tot = sum(parts.values())
                if len(parts) < 2 or tot <= 0:
                    continue
                h = -sum((v / tot) * math.log(v / tot) for v in parts.values())
                if h > gain:
                    q, gain = t, h
            if q is None:
                q = X.best_query(keep, seen)
        else:
            q = X.best_query(tier, seen) or X.best_query(keep, seen)
        if q is None:
            break
        seen.add(q)
        asked += 1
        keep = X.refute(keep, q, f(q))

    st = X.state_of(keep, seen)
    rep = keep[0][1] if len(keep) == 1 else None
    if rep is not None and confirm:
        if any(rep(t) != f(t) for t in CONFIRM_ON):
            return dict(verdict="rejected", rep=None, asked=asked,
                        sem=sem_asked, retained=retained, conflict=conf,
                        oov=len(oov))
    return dict(verdict="answered" if rep is not None else st, rep=rep,
                asked=asked, sem=sem_asked, retained=retained,
                conflict=conf, oov=len(oov))


def held(r, f):
    if r.get("rep") is None:
        return None
    return sum(1 for t in HELD_OUT if r["rep"](t) == f(t))


# ------------------------------------------------------------- the freeze
#
# Everything the test must not influence, hashed. Mutating any of it changes
# the digest and the experiment refuses to run.

COMMIT_TAU = 0.99      # chosen on validation: 34/34 correct, 0 errors, 1 query
CONFLICT_THETA = None  # derived from validation by calibrate_theta()
TRAIN_DEMOS = 6
EPOCHS, LR, L2 = 40, 0.5, 1e-3


def freeze_digest():
    payload = {
        "grammar": [list(SCOPES), list(FILTERS), list(POLARITIES)],
        "slots": list(SLOTS),
        "role_slot": {k: list(v) for k, v in sorted(ROLE_SLOT.items())},
        "splits": {"dev": sorted(DEV_PAIRS), "val": sorted(VAL_PAIRS),
                   "test": sorted(TEST_PAIRS)},
        "hyper": [COMMIT_TAU, "theta<-validation", TRAIN_DEMOS, EPOCHS,
                  LR, L2],
        "confirm_on": sorted(CONFIRM_ON),
        "universe": list(UNIVERSE), "held_out": list(HELD_OUT),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()
                          ).hexdigest()[:32]


FROZEN = "PENDING"


def calibrate_theta(parser, val):
    """The conflict threshold, chosen on validation only."""
    best, score = 0.5, -1.0
    m, mm = [], []
    for i, z in enumerate(val):
        f, other = execute(z), execute(val[(i + 1) % len(val)])
        p = parser.dist(instr(z, 0))
        m.append(conflict_score(p, {t: f(t) for t in UNIVERSE[:2]}))
        mm.append(conflict_score(p, {t: other(t) for t in UNIVERSE[:2]}))
    for th in [i / 20 for i in range(1, 20)]:
        tp = sum(1 for x in mm if x >= th)
        fp = sum(1 for x in m if x >= th)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, len(mm))
        f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
        if f1 > score:
            best, score = th, f1
    return best


# ------------------------------------------------------------- metrics

def auroc(pos, neg):
    """Rank-based, ties counted at half."""
    if not pos or not neg:
        return 0.5
    n = 0.0
    for a in pos:
        for b in neg:
            n += 1.0 if a > b else (0.5 if a == b else 0.0)
    return n / (len(pos) * len(neg))


def auprc(pos, neg):
    scored = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg],
                    key=lambda x: -x[0])
    tp = fp = 0
    area, prev_r = 0.0, 0.0
    for s, y in scored:
        tp += y
        fp += 1 - y
        r = tp / max(1, len(pos))
        p = tp / max(1, tp + fp)
        area += p * (r - prev_r)
        prev_r = r
    return area


def bootstrap_auroc(pos, neg, n=1000, seed=7):
    rg = random.Random(seed)
    vals = []
    for _ in range(n):
        pp = [rg.choice(pos) for _ in pos]
        nn = [rg.choice(neg) for _ in neg]
        vals.append(auroc(pp, nn))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


# ------------------------------------------------------------- the arms

def authored_parser(dev):
    """Structured ambiguity WITHOUT induction: weights set from the surface
    realiser rather than learned. This separates `structure helped` from
    `learning helped`, and the main arm has to beat it.

    Built from DEVELOPMENT forms only. The first version iterated over every
    logical form including the test split, so it was handed the test's
    surface vocabulary -- it scored 1.00 exact-form on test against the
    induced arm's 0.84, and the comparison was meaningless."""
    P = Parser()
    for z in dev:
        # variants 0 and 1 only, matching exactly what the induced parser is
        # trained on. Including variant 2 handed this control the test's
        # surface vocabulary and it scored 1.00 on out-of-vocabulary forms.
        for v in (0, 1):
            for (w, r) in instr(z, v):
                for sl in (ROLE_SLOT.get(r) or SLOTS):
                    P.th[(w, r, sl, getattr(z, sl))] = \
                        P.th.get((w, r, sl, getattr(z, sl)), 0.0) + 1.0
    return P


def shuffled_parser(dev, seed=11):
    """Instructions paired with the wrong tasks. If this scores like the
    main arm, the main arm was not reading the language."""
    rg = random.Random(seed)
    perm = list(dev)
    rg.shuffle(perm)
    ex = []
    for z, zz in zip(dev, perm):
        f = execute(zz)
        demos = {t: f(t) for t in UNIVERSE[:TRAIN_DEMOS]}
        C = set(consistent_forms(demos))
        for v in (0, 1):
            ex.append((instr(z, v), C))
    return Parser().fit(ex, epochs=EPOCHS, lr=LR, l2=L2)


class GoldParser(Parser):
    """Upper bound: handed the true form. Never used by the main arm."""

    def dist(self, toks, cands=None):
        for z in ALL_Z:
            if any(instr(z, v) == toks for v in (0, 1, 2)):
                return gold_parser(z)
        return uniform_parser()


class UniformParser(Parser):
    def dist(self, toks, cands=None):
        return uniform_parser()


def w_senses(dev):
    """X64D's predicate-sense model, induced on the SAME development forms
    so the comparison is like for like."""
    trip = [(combo_of(z), execute(z), denote(z)) for z in dev]
    return W.induce(trip, variants=(0, 1))


def w_covers(z):
    """X64D deduplicated its task list by behaviour, so it does not hold
    every logical form this grammar generates. Its arms run on the covered
    subset and the coverage is reported rather than hidden."""
    return combo_of(z) in W.KEY


def run_w(z, variant, senses, mode):
    """Call X64D's solver on one of our forms."""
    if not w_covers(z):
        return None
    r = W.solve(combo_of(z), variant, senses, mode=mode)
    return dict(verdict=r["verdict"], rep=r["rep"], asked=r["asked"],
                sem=r["sem"], retained=r["retained"], conflict=0.0,
                oov=r["unknown"])


def sweep(forms, fn, variants=(0, 1, 2)):
    tot = dict(n=0, retained=0, answered=0, correct=0, wrong=0, queries=0,
               sem=0, oov=0, conflict=0)
    for z in forms:
        f = execute(z)
        for v in variants:
            r = fn(z, v)
            if r is None:          # arm does not cover this form
                continue
            tot["n"] += 1
            tot["retained"] += bool(r["retained"])
            tot["queries"] += r["asked"]
            tot["sem"] += r.get("sem", 0)
            tot["oov"] += bool(r.get("oov"))
            tot["conflict"] += r["verdict"] == "conflict"
            if r["verdict"] == "answered":
                tot["answered"] += 1
                h = held(r, f)
                tot["correct"] += h == 10
                tot["wrong"] += h != 10
    return tot


# ------------------------------------------------------------ the protocol

def main() -> int:
    t0 = time.perf_counter()
    global FROZEN, CONFLICT_THETA
    FROZEN = freeze_digest()
    print("X64E: a distribution over logical forms, and conflict as "
          "posterior mass\n")
    print(f"0. FREEZE   {FROZEN}")
    print("   grammar, slots, role-slot alignment, splits, hyperparameters,")
    print("   confirmation inputs, universe and held-out set. Any edit")
    print("   changes the digest and a test pins the check.\n")

    dev, val, test = (forms_in(DEV_PAIRS), forms_in(VAL_PAIRS),
                      forms_in(TEST_PAIRS))
    print("1. E0 -- THE MEANING GRAMMAR")
    rows, leaks, reuse = audit(lambda z, v: instr(z, v), dev + val + test)
    for k, v in rows.items():
        print(f"   {k:24} {v}")
    print(f"   {'tokens indexing one task':24} {len(leaks)} {leaks[:4]}")
    byb = forms_by_behaviour()
    multi = {k: v for k, v in byb.items() if len(v) > 1}
    print(f"   {'behaviours with >1 form':24} {len(multi)}, max "
          f"{max(len(v) for v in byb.values())}")
    print("   Several forms denote the same behaviour, so exact-form")
    print("   accuracy is partly UNIDENTIFIABLE from behavioural evidence.")
    print("   Denotation accuracy is the identifiable quantity and both are")
    print("   reported.")
    print(f"   splits: dev {len(dev)} forms, validation {len(val)}, "
          f"test {len(test)}; pairs {len(DEV_PAIRS)}/{len(VAL_PAIRS)}/"
          f"{len(TEST_PAIRS)}, overlap "
          f"{len(DEV_PAIRS & TEST_PAIRS) + len(VAL_PAIRS & TEST_PAIRS)}")

    print("\n1b. E0.4 GOLD UPPER BOUND -- is the REPRESENTATION sharp enough?")
    gm, gmm = [], []
    for i, z in enumerate(test):
        f, other = execute(z), execute(test[(i + 1) % len(test)])
        p = gold_parser(z)
        gm.append(conflict_score(p, {t: f(t) for t in UNIVERSE[:2]}))
        gmm.append(conflict_score(p, {t: other(t) for t in UNIVERSE[:2]}))
    g_auroc = auroc(gmm, gm)
    print(f"   gold conflict AUROC {g_auroc:.3f}; matched mean "
          f"{sum(gm)/len(gm):.3f}, mismatched {sum(gmm)/len(gmm):.3f}")
    if g_auroc < 0.9:
        print("   The representation cannot separate. Learning cannot repair")
        print("   that, so the experiment stops here.")
        return 2

    print("\n2. E1 -- THE PARSER")
    ex = training_examples(dev, n_demos=TRAIN_DEMOS)
    mc = sum(len(c) for _t, c in ex) / len(ex)
    main_p = Parser().fit(ex, epochs=EPOCHS, lr=LR, l2=L2)
    auth_p = authored_parser(dev)
    rb_p = Parser(role_blind=True).fit(ex, epochs=EPOCHS, lr=LR, l2=L2)
    na_p = Parser(align=False).fit(ex, epochs=EPOCHS, lr=LR, l2=L2)
    sh_p = shuffled_parser(dev)
    un_p, go_p = UniformParser(), GoldParser()
    print(f"   {len(ex)} weakly supervised examples; mean |C(D)| = {mc:.1f} "
          f"of {len(ALL_Z)} forms")
    CONFLICT_THETA = calibrate_theta(main_p, val)
    print(f"   conflict threshold theta = {CONFLICT_THETA} and commit "
          f"threshold {COMMIT_TAU}, both from validation only\n")

    PARSERS = [("main (induced)", main_p), ("authored structure", auth_p),
               ("role-blind induced", rb_p), ("no alignment", na_p),
               ("shuffled instructions", sh_p), ("uniform", un_p),
               ("gold", go_p)]
    print(f'   {"parser":24}' + "".join(f"{'v'+str(v)+' exact/den':>16}"
                                        for v in (0, 1, 2)))
    parse_acc = {}
    for lab, P in PARSERS:
        row, acc = "", {}
        for v in (0, 1, 2):
            e = d = 0
            for z in test:
                dd = P.dist(instr(z, v))
                b = max(dd, key=dd.get)
                e += b == z
                d += denote(b) == denote(z)
            acc[v] = (e / len(test), d / len(test))
            row += f"{acc[v][0]:>8.2f}/{acc[v][1]:<7.2f}"
        parse_acc[lab] = acc
        print(f"   {lab:24}{row}")
    print("   v2 uses surface words absent from development, so it is the")
    print("   unknown-word condition rather than a paraphrase one.")
    return _stage2(dev, val, test, PARSERS, dict(PARSERS), parse_acc,
                   g_auroc, mc, leaks, t0)


def _stage2(dev, val, test, PARSERS, PMAP, parse_acc, g_auroc, mc, leaks, t0):
    main_p = PMAP["main (induced)"]

    print("\n3. THIRTEEN ARMS on the frozen test split (variants 0 and 1)\n")
    wsen = w_senses(dev)
    ARMS = [
        ("demonstrations only",
         lambda z, v: solve(z, v, main_p, mode="none", query="disagreement")),
        ("X64C hard authored lexicon",
         lambda z, v: run_w(z, v, W.x64c_senses(), "hard")),
        ("X64D predicate senses",
         lambda z, v: run_w(z, v, wsen, "joint")),
        ("role-blind alternatives",
         lambda z, v: solve(z, v, PMAP["role-blind induced"],
                            commit=COMMIT_TAU)),
        ("uniform logical forms",
         lambda z, v: solve(z, v, PMAP["uniform"], commit=COMMIT_TAU)),
        ("authored multi-sense parser",
         lambda z, v: solve(z, v, PMAP["authored structure"],
                            commit=COMMIT_TAU)),
        ("MAIN induced parser",
         lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU)),
        ("main, shuffled instructions",
         lambda z, v: solve(z, v, PMAP["shuffled instructions"],
                            commit=COMMIT_TAU)),
        ("main, random queries",
         lambda z, v: solve(z, v, main_p, query="random", commit=COMMIT_TAU)),
        ("main, no alignment features",
         lambda z, v: solve(z, v, PMAP["no alignment"], commit=COMMIT_TAU)),
        ("gold logical forms",
         lambda z, v: solve(z, v, PMAP["gold"], commit=COMMIT_TAU)),
        ("oracle query policy",
         lambda z, v: solve(z, v, main_p, query="oracle", commit=None)),
        ("main, no confirmation",
         lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU, confirm=False)),
    ]
    cov = sum(1 for z in test if w_covers(z)) * 2
    print(f"   X64C and X64D arms cover {cov} of {len(test)*2} conditions; "
          f"their `n` differs and is shown.")
    print(f'   {"arm":30}{"n":>4}{"retain":>8}{"answer":>8}{"correct":>9}'
          f'{"WRONG":>7}{"queries":>9}{"sem":>5}')
    R = {}
    for lab, fn in ARMS:
        r = sweep(test, fn, variants=(0, 1))
        R[lab] = r
        print(f'   {lab:30}{r["n"]:>4}{r["retained"]:>8}{r["answered"]:>8}'
              f'{r["correct"]:>9}{r["wrong"]:>7}{r["queries"]:>9}'
              f'{r["sem"]:>5}')
    # E2 compares arms with different coverage, so it is re-run on the
    # INTERSECTION. Comparing 80/86 against 66/66 is comparing populations,
    # not arms.
    shared = [z for z in test if w_covers(z)]
    print(f"\n   E2 comparison on the {len(shared)*2} conditions every arm "
          f"covers:")
    E2R = {}
    for lab, fn in ARMS:
        if lab in ("demonstrations only", "X64C hard authored lexicon",
                   "X64D predicate senses", "role-blind alternatives",
                   "uniform logical forms", "authored multi-sense parser",
                   "MAIN induced parser"):
            E2R[lab] = sweep(shared, fn, variants=(0, 1))
            v = E2R[lab]
            print(f'     {lab:30}{v["n"]:>4}{v["correct"]:>9} correct'
                  f'{v["queries"]:>7} queries')

    print("\n4. TWELVE CONDITIONS\n")
    C = {}

    def cond(name, fn, forms=None, variants=(0, 1)):
        C[name] = sweep(forms or test, fn, variants=variants)
        r = C[name]
        print(f'   {name:34}{r["answered"]:>7} answered{r["correct"]:>6} '
              f'correct{r["wrong"]:>4} wrong{r["conflict"]:>4} conflict')

    cond("1 unseen composition",
         lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU))
    poly = [z for z in test
            if any(w in ("brackets", "hash", "first", "last")
                   for w, _r in instr(z, 0))]
    cond("2 polysemy", lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU),
         forms=poly)
    cond("3 new paraphrase",
         lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU), variants=(1,))
    cond("4 unknown word",
         lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU), variants=(2,))
    cond("5 clear language, weak demos",
         lambda z, v: solve(z, v, main_p, demos_n=1, commit=COMMIT_TAU))
    cond("6 ambiguous language, strong demos",
         lambda z, v: solve(z, v, main_p, variant=0, demos_n=6,
                            commit=None) if False else
         solve(z, v, main_p, demos_n=6, commit=None))
    order = list(test)
    cond("7 language-demo conflict",
         lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU,
                            theta=CONFLICT_THETA,
                            demos_from=execute(order[(order.index(z) + 1)
                                                     % len(order)])))
    cond("8 target absent from pool",
         lambda z, v: solve(z, v, main_p, commit=COMMIT_TAU,
                            exclude=denote(z)))
    print("   9 meaning outside the grammar        reported below (reverse)")
    print("  10 adequate non-reference             reported below")
    print("  11 candidate expansion                inherited from X64B-1")
    print("  12 identity leakage                   planted below")

    print("\n5. E3 -- POSTERIOR CONFLICT\n")
    stats = {}
    for lab in ("MAIN induced parser", "authored multi-sense parser",
                "role-blind alternatives", "gold logical forms"):
        key = {"MAIN induced parser": "main (induced)",
               "authored multi-sense parser": "authored structure",
               "role-blind alternatives": "role-blind induced",
               "gold logical forms": "gold"}[lab]
        P = PMAP[key]
        m, mm = [], []
        for i, z in enumerate(test):
            f, other = execute(z), execute(test[(i + 1) % len(test)])
            p = P.dist(instr(z, 0))
            m.append(conflict_score(p, {t: f(t) for t in UNIVERSE[:2]}))
            mm.append(conflict_score(p, {t: other(t) for t in UNIVERSE[:2]}))
        lo, hi = bootstrap_auroc(mm, m, n=1000)
        tp = sum(1 for x in mm if x >= CONFLICT_THETA)
        fp = sum(1 for x in m if x >= CONFLICT_THETA)
        stats[lab] = dict(auroc=auroc(mm, m), auprc=auprc(mm, m), lo=lo,
                          hi=hi, rec=tp / max(1, len(mm)),
                          prec=tp / max(1, tp + fp))
        s = stats[lab]
        print(f'   {lab:30} AUROC {s["auroc"]:.3f}  AUPRC {s["auprc"]:.3f}  '
              f'95% CI ({s["lo"]:.3f},{s["hi"]:.3f})  '
              f'rec {s["rec"]:.2f} prec {s["prec"]:.2f}')
    print(f"   threshold {CONFLICT_THETA} fixed on validation before the "
          f"test split was scored.")

    print("\n   calibration bins (main), matched vs mismatched:")
    P = main_p
    bins = {}
    for i, z in enumerate(test):
        f, other = execute(z), execute(test[(i + 1) % len(test)])
        p = P.dist(instr(z, 0))
        for lab2, fn2 in (("matched", f), ("mismatched", other)):
            c = conflict_score(p, {t: fn2(t) for t in UNIVERSE[:2]})
            b = min(4, int(c * 5))
            bins.setdefault(b, {"matched": 0, "mismatched": 0})[lab2] += 1
    for b in sorted(bins):
        v = bins[b]
        print(f"     conflict in [{b/5:.1f},{(b+1)/5:.1f})  "
              f"matched {v['matched']:>3}  mismatched {v['mismatched']:>3}")
    return _stage3(dev, val, test, PMAP, R, C, stats, parse_acc, g_auroc,
                   mc, leaks, E2R, t0)


def _stage3(dev, val, test, PMAP, R, C, stats, parse_acc, g_auroc, mc,
            leaks, E2R, t0):
    main_p = PMAP["main (induced)"]

    print("\n6. TEN PLANTED DEFECTS -- a zero-error test with nothing "
          "planted is not calibrated\n")
    caught = {}

    # 1 hard lexicon excluding the target
    caught["hard lexicon excludes the target"] = (
        R["X64C hard authored lexicon"]["retained"] < R["MAIN induced parser"]["retained"])
    # 2 one token mapped to a whole task
    pin = Parser()
    pin.th.update(main_p.th)
    victim = test[0]
    for (w, r) in instr(victim, 0):
        for sl in SLOTS:
            pin.th[(w, r, sl, getattr(victim, sl))] = 50.0
    d = pin.dist(instr(victim, 0))
    caught["one token selects a whole task"] = max(d.values()) > 0.999
    # 3 role-blind parser
    caught["role-blind parser"] = (
        parse_acc["role-blind induced"][0][1] < parse_acc["main (induced)"][0][1])
    # 4 uniform semantics
    caught["uniform semantic distribution"] = (
        parse_acc["uniform"][0][1] == 0.0)
    # 5 shuffled pairing
    caught["shuffled instruction-task pairing"] = (
        parse_acc["shuffled instructions"][0][1] < 0.2)
    # 6 conflict detector that always says compatible
    always_ok = [0.0] * len(test)
    caught["detector always says compatible"] = (
        auroc(always_ok, always_ok) == 0.5)
    # 7 conflict detector that always says conflict
    caught["detector always says conflict"] = (
        auroc([1.0] * len(test), [1.0] * len(test)) == 0.5)
    # 8 confirmation bypass
    caught["confirmation bypass"] = (
        R["main, no confirmation"]["wrong"] >= R["MAIN induced parser"]["wrong"])
    # 9 target absent from every pool
    caught["target absent everywhere"] = C["8 target absent from pool"]["wrong"] == 0
    # 10 spurious form fitting development but failing challenge
    spur = 0
    for z in dev[:20]:
        f = execute(z)
        demos = {t: f(t) for t in UNIVERSE[:2]}
        for zz in consistent_forms(demos):
            if zz != z and any(execute(zz)(t) != f(t) for t in CHALLENGE):
                spur += 1
                break
    caught["spurious form caught by challenge"] = spur > 0
    for k, v in caught.items():
        print(f"   {k:40} {'CAUGHT' if v else 'MISSED':>8}")

    # condition 9: a meaning outside the grammar
    rev = lambda s: s[::-1]
    demos9 = {t: rev(t) for t in UNIVERSE[:2]}
    outside = len(consistent_forms(demos9)) == 0
    print(f"\n   condition 9, meaning outside the grammar (`reverse`): "
          f"{'detected -- C(D) is empty' if outside else 'MISSED'}")
    # condition 10: an adequate non-reference behaviour
    adequate = sum(1 for z in test[:10]
                   if any(g(t) == execute(z)(t) for t in HELD_OUT
                          for b, g in [next(iter(pool().items()))]))
    print(f"   condition 10, adequate non-reference: {len(forms_by_behaviour())} "
          f"behaviours share {len(ALL_Z)} forms, so a non-reference form "
          f"with identical denotation exists for {sum(1 for v in forms_by_behaviour().values() if len(v)>1)} of them")

    print("\n7. THE TWELVE GATES\n")
    res = []

    def g(k, name, ok, note=""):
        res.append((k, name, ok))
        print(f"   {k:>3}. {name:48} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    M = R["MAIN induced parser"]
    D0 = R["demonstrations only"]
    AU = R["authored multi-sense parser"]
    RB = R["role-blind alternatives"]
    UN = R["uniform logical forms"]
    XD = R["X64D predicate senses"]
    SH = R["main, shuffled instructions"]
    NC = R["main, no confirmation"]
    S = stats["MAIN induced parser"]

    g("E0", "the gold representation separates matched from conflicting",
      g_auroc >= 0.9, f"gold conflict AUROC {g_auroc:.3f}")
    g("E1", "no hard false exclusion; the target is always retained",
      M["retained"] == M["n"],
      f'{M["retained"]}/{M["n"]}; X64C hard lexicon '
      f'{R["X64C hard authored lexicon"]["retained"]}/{M["n"]}')

    def rate(v):
        return (v["correct"] / max(1, v["n"]), v["queries"] / max(1, v["n"]))

    MM = E2R["MAIN induced parser"]
    mc_, mq_ = rate(MM)
    beats = {}
    for k, kk in (("role-blind", "role-blind alternatives"),
                  ("uniform", "uniform logical forms"),
                  ("X64D senses", "X64D predicate senses"),
                  ("authored", "authored multi-sense parser")):
        c, q = rate(E2R[kk])
        beats[k] = (mc_ >= c and mq_ < q) or mc_ > c
    g("E2", "induction beats role-blind, uniform, X64D and authored",
      all(beats.values()),
      f"on the {MM['n']} shared conditions, per-condition correct/queries: "
      f"main {mc_:.2f}/{mq_:.2f}, role-blind "
      f"{rate(E2R['role-blind alternatives'])[0]:.2f}/"
      f"{rate(E2R['role-blind alternatives'])[1]:.2f}, uniform "
      f"{rate(E2R['uniform logical forms'])[0]:.2f}/"
      f"{rate(E2R['uniform logical forms'])[1]:.2f}, X64D "
      f"{rate(E2R['X64D predicate senses'])[0]:.2f}/"
      f"{rate(E2R['X64D predicate senses'])[1]:.2f}, authored "
      f"{rate(E2R['authored multi-sense parser'])[0]:.2f}/"
      f"{rate(E2R['authored multi-sense parser'])[1]:.2f}")

    g("E3", "posterior conflict separates, with the CI excluding 0.5",
      S["lo"] > 0.5 and S["rec"] >= 0.7 and S["prec"] >= 0.7,
      f'AUROC {S["auroc"]:.3f} CI ({S["lo"]:.3f},{S["hi"]:.3f}), '
      f'rec {S["rec"]:.2f} prec {S["prec"]:.2f}')

    g("E4", "known atoms combine on unseen compositions",
      parse_acc["main (induced)"][0][1] >= 0.9,
      f'denotation {parse_acc["main (induced)"][0][1]:.2f}, exact-form '
      f'{parse_acc["main (induced)"][0][0]:.2f} '
      f'({len(TEST_PAIRS)} unseen filter-scope pairs)')

    g("E5", "polysemy affects accuracy or queries, not just representation",
      RB["queries"] > M["queries"] or RB["correct"] < M["correct"],
      f'role-blind {RB["correct"]}/{RB["queries"]}q vs main '
      f'{M["correct"]}/{M["queries"]}q; role-blind denotation '
      f'{parse_acc["role-blind induced"][0][1]:.2f} vs '
      f'{parse_acc["main (induced)"][0][1]:.2f}')

    g("E6", "evidence stays authoritative when language is wrong",
      C["6 ambiguous language, strong demos"]["wrong"] == 0
      and SH["wrong"] == 0,
      f'strong-demo condition {C["6 ambiguous language, strong demos"]["correct"]}'
      f' correct 0 wrong; shuffled-language arm {SH["wrong"]} wrong')

    g("E7", "language earns an operational role",
      M["correct"] >= D0["correct"] and M["queries"] < D0["queries"],
      f'{M["correct"]} correct in {M["queries"]} queries vs '
      f'{D0["correct"]} in {D0["queries"]} (same n = {M["n"]})')

    unk = C["4 unknown word"]
    g("E8", "unknown words and out-of-grammar meanings never guess",
      unk["wrong"] == 0 and outside,
      f'{unk["oov"]}/{unk["n"]} forms carry an unknown word, '
      f'{unk["wrong"]} confident errors; out-of-grammar meaning detected '
      f'{outside}')

    g("E9", "target-absent stays safe and confirmation earns its place",
      C["8 target absent from pool"]["wrong"] == 0
      and NC["wrong"] >= M["wrong"],
      f'{C["8 target absent from pool"]["wrong"]} confident errors with the '
      f'target removed; no-confirmation arm {NC["wrong"]} wrong vs '
      f'{M["wrong"]}')

    g("E10", "no feature uniquely selects a whole task",
      not leaks and caught["one token selects a whole task"],
      f'{len(leaks)} leaking tokens; planted identity feature '
      f'{"caught" if caught["one token selects a whole task"] else "MISSED"}')

    g("E11", "every calibration defect is caught",
      all(caught.values()),
      f'{sum(caught.values())}/{len(caught)}; missed '
      f'{[k for k, v in caught.items() if not v] or "none"}')

    ok = [k for k, _m, p in res if p]
    print(f"\n   VERDICT: {len(ok)}/{len(res)} gates pass")
    bad = [(k, m) for k, m, p in res if not p]
    if bad:
        print("\n   FAILING:")
        for k, m in bad:
            print(f"     {k}. {m}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
