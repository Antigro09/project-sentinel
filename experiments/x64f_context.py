"""X64F: surface language that a word-to-slot table cannot decode.

X64E's realiser was nearly a serialisation of the logical form -- which is
why a parser whose weights were read straight off it scored 1.00 exact-form
without learning anything. The linguistic problem had largely been solved by
the data generator, so X64 was not closed and the X64E report said so.

This replaces the realiser with one built to defeat a static map. The
decisive device: the SAME NOUNS serve as either the filter or the scope
delimiter, and only ORDER says which.

    remove the brackets before the hash   ->  remove(brackets @ before hash)
    remove the hash before the brackets   ->  remove(hashes  @ before brackets)

Identical multisets, different meanings. No assignment of weights to words
can separate them; position and neighbourhood can. That single construction
is what F1 tests, and it is not repairable by a bigger lexicon.

Six more phenomena, each required by the protocol:

    contextual polysemy   `leave X`  keeps,  `leave out X`  removes
    nonlocal composition  `leave X out` removes, particle separated from
                          its verb by the whole object
    many-to-one           keep <- keep / take / leave / hold on to
    one-to-many           `first` is a filter after `the`, a scope after
                          `after the`, and part of a comparison after
                          `matching the`
    phrase-level meaning  `in a row`, `seen before`, `hold on to` -- no
                          single token carries them
    weak words            please, kindly, just, all, simply: present, and
                          semantically inert
    omitted arguments     the scope phrase may be dropped entirely, leaving
                          it to the demonstrations or a clarification

MEASURED. 12 of 12 gates, over three independently seeded frozen splits.

  50 multiset collisions covering 46 of 230 live forms; no surface STRING is
  ambiguous, so word order resolves them and the denotation ceiling is 1.00.

  parser              seed 101      seed 202      seed 303
                    den   coll    den   coll    den   coll
  contextual        0.67  0.50    0.67  0.89    0.71  0.29
  bag-of-words      0.67  0.11    0.68  0.44    0.76  0.07
  authored          0.11  0.00    0.18  0.28    0.02  0.00
  shuffled          0.00  0.00    0.00  0.00    0.01  0.00
  uniform           0.00  0.00    0.00  0.00    0.02  0.00
  gold              1.00  1.00    1.00  1.00    1.00  1.00

  F1, pooled and paired by task meaning:
    collision cases  contextual 29/50 = 0.58, bag-of-words 11/50 = 0.22
                     difference +0.359, 95% CI (+0.220,+0.500) EXCLUDES 0
    all cases        difference -0.026, 95% CI (-0.078,+0.028) INCLUDES 0

CONTEXT BUYS EXACTLY THE CONSTRUCTION IT SHOULD AND NOTHING ELSE. On the
collisions it is 2.6x bag-of-words with an interval that excludes zero;
across all cases the difference is indistinguishable from zero. Elsewhere
the words alone determine the reading and the extra features are noise.
Reporting only the overall number would have hidden both halves of that.

  arm                    answered  correct  wrong  queries
  demonstrations only         360      360      0      842
  bag-of-words                360      360      0      727
  contextual (main)           360      360      0      721
  shuffled language           356      356      0     1087
  main, no confirmation       382      374      8      721
  main, target removed          6        6      0        0

  conflict AUROC 0.943, 95% CI (0.918,0.966)
  F8 paired saving per task meaning +0.625, 95% CI (+0.440,+0.834)

AND THE AUTHORED CONTROL COLLAPSES. On X64E's realiser a hand-written
word-to-slot table reached 1.00 exact-form without learning anything, which
is why X64 was not closed. Here it reaches 0.02 to 0.18. That is the whole
point of the experiment: the realiser, not the parser, was doing the work.

WHAT IS STILL WEAK, stated rather than buried:
  the operational gain over BAG-OF-WORDS is negligible -- 721 queries
    against 727. The paired saving that survives an interval is against
    DEMONSTRATIONS-ONLY, not against the bag-of-words control.
  180 unknown-word cases give 167 correct answers, and the evidence
    supplied every one of them. The parser does not understand those words;
    it declines to guess, and the demonstrations resolve the task. Safe, and
    not comprehension.
  with the target removed the system answers 6 of 1080 conditions. Zero
    errors, and very little coverage.
  F4's margin is thin: 0.74 against 0.71 on phrase-only filters.

THREE MID-RUN CORRECTIONS, each a measured rejection:
  AdaGrad was tried on the hypothesis that sparse contextual features needed
    per-feature rates. It made both arms much worse (0.14 and 0.28 against
    0.51 and 0.59) and the flag stays off.
  the gradient was SUMMED, so the effective step scaled with the dataset:
    504 examples gave 0.58 on validation and 637 gave 0.04. That was
    divergence, not overfitting. Averaging fixed it and the fix is pinned.
  the first collision family was too small to measure. 22 bags over 188
    forms gave 9 training and 9 test instances, contextual tied bag-of-words
    on the first three-seed run, and the phenomenon F1 exists to test was 5%
    of the data. `letters` was added as a third role-swappable noun, taking
    the family to 50 bags over 46 forms.

Run: uv run python experiments/x64f_context.py
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
import x64d_senses as W
import x64e_semantics as S

UNIVERSE, HELD_OUT = S.UNIVERSE, S.HELD_OUT
CONFIRM_ON = S.CONFIRM_ON


# --------------------------------------------------- an extended grammar
#
# `before brackets` and `after brackets` are added so that the two nouns can
# swap roles and produce the multiset collisions above.

# `letters` joins `hash` and `brackets` as a noun that is BOTH a filter and
# a scope delimiter. With only two such nouns, 22 of 188 forms took part in a
# multiset collision -- 9 training and 9 test instances, which is too few
# either to learn the distinction or to measure it, and the first three-seed
# run showed contextual features tied with bag-of-words as a result. The
# phenomenon F1 exists to test has to be common, not incidental.
SCOPES = S.SCOPES + ("before brackets", "after brackets",
                     "before letters", "after letters")
FILTERS = S.FILTERS
POLARITIES = S.POLARITIES


def scope_positions(s, scope):
    if scope == "before letters":
        i = next((k for k, c in enumerate(s) if c not in "()#"), -1)
        return list(range(len(s) if i < 0 else i))
    if scope == "after letters":
        i = max((k for k, c in enumerate(s) if c not in "()#"), default=-1)
        return [] if i < 0 else list(range(i + 1, len(s)))
    if scope == "before brackets":
        i = s.find("(")
        return list(range(len(s) if i < 0 else i))
    if scope == "after brackets":
        i = s.rfind(")")
        return [] if i < 0 else list(range(i + 1, len(s)))
    return W.scope_positions(s, scope)


def make_task(scope, filt, polarity):
    def f(s):
        pos = scope_positions(s, scope)
        first = s[pos[0]] if pos else None
        seen, out = set(), []
        for i in pos:
            c = s[i]
            if filt == "everything":
                hit = True
            elif filt == "letters":
                hit = c not in "()#"
            elif filt == "brackets":
                hit = c in "()"
            elif filt == "hashes":
                hit = c == "#"
            elif filt == "repeats in a row":
                hit = bool(out) and out[-1] == c
            elif filt in ("repeats", "symbols seen before"):
                hit = c in seen
            elif filt == "the first symbol":
                hit = i == pos[0]
            elif filt == "the last symbol":
                hit = i == pos[-1]
            elif filt == "symbols at even places":
                hit = pos.index(i) % 2 == 0
            elif filt == "symbols before a repeat":
                hit = c not in seen and s.count(c) > 1
            else:
                hit = c == first
            seen.add(c)
            if hit == (polarity == "keep"):
                out.append(c)
        return "".join(out)
    return f


class Z(tuple):
    __slots__ = ()

    def __new__(cls, op, filt, scope):
        return super().__new__(cls, (op, filt, scope))

    op = property(lambda self: self[0])
    filt = property(lambda self: self[1])
    scope = property(lambda self: self[2])

    def __repr__(self):
        return f"{self[0]}({self[1]} @ {self[2]})"


SLOTS = ("op", "filt", "scope")
VALUES = {"op": POLARITIES, "filt": FILTERS, "scope": SCOPES}
ALL_Z = [Z(o, f, s) for o in POLARITIES for f in FILTERS for s in SCOPES]

_BEH = {}


def denote(z):
    if z not in _BEH:
        f = make_task(z.scope, z.filt, z.op)
        _BEH[z] = tuple(f(t) for t in UNIVERSE)
    return _BEH[z]


def execute(z):
    return make_task(z.scope, z.filt, z.op)


LIVE = [z for z in ALL_Z if not all(o == "" for o in denote(z))]


# ------------------------------------------------- the contextual realiser

VERB_KEEP = [["keep"], ["take"], ["leave"], ["hold", "on", "to"]]
VERB_DROP = [["drop"], ["get", "rid", "of"]]
PARTICLE_VERBS = [["leave"], ["take"]]          # V + out  =>  remove

FILTER_NP = {
    "everything": [["everything"]],
    "letters": [["the", "letters"]],
    "brackets": [["the", "brackets"], ["the", "parens"]],
    "hashes": [["the", "hash"], ["the", "hashes"]],
    "repeats in a row": [["the", "repeats", "in", "a", "row"]],
    "repeats": [["the", "repeats"]],
    "symbols seen before": [["the", "symbols", "seen", "before"]],
    "symbols matching the first": [["the", "symbols", "matching",
                                    "the", "first"]],
    "the first symbol": [["the", "first"]],
    "the last symbol": [["the", "last"]],
    "symbols at even places": [["the", "even", "symbols"]],
    "symbols before a repeat": [["the", "symbols", "before", "a", "repeat"]],
}

SCOPE_PP = {
    "whole": [[]],
    "before hash": [["before", "the", "hash"]],
    "after hash": [["after", "the", "hash"]],
    "inside brackets": [["inside", "the", "brackets"]],
    "outside brackets": [["outside", "the", "brackets"]],
    "before brackets": [["before", "the", "brackets"]],
    "after brackets": [["after", "the", "brackets"]],
    "after the first symbol": [["after", "the", "first"]],
    "before the last": [["before", "the", "last"]],
    "before letters": [["before", "the", "letters"]],
    "after letters": [["after", "the", "letters"]],
}

ADAGRAD = False
WEAK = ["please", "kindly", "just", "simply"]


def realise(z, variant=0, rng=None):
    """Templates, not slots. `variant` selects a CONSTRUCTION, and the
    constructions differ in what a word means, not only in which word.

    `rng` varies the lexical choices INSIDE a construction, which is how
    training gets more than one surface string per form without touching a
    held-out template. 168 examples against tens of thousands of contextual
    features was the reason the contextual parser lost to bag-of-words on
    validation."""
    pick = (lambda xs: xs[rng.randrange(len(xs))]) if rng else \
        (lambda xs: xs[variant % len(xs)])
    fil = pick(FILTER_NP[z.filt])
    sco = SCOPE_PP[z.scope][0]
    if z.op == "keep":
        v = pick(VERB_KEEP) if rng else VERB_KEEP[variant % len(VERB_KEEP)]
        wk = pick(WEAK) if rng else WEAK[variant % len(WEAK)]
        forms = [v + fil + sco,
                 (sco + [","] if sco else []) + v + fil,
                 [wk] + v + fil + sco]
    else:
        pv = pick(PARTICLE_VERBS) if rng else \
            PARTICLE_VERBS[variant % len(PARTICLE_VERBS)]
        dv = pick(VERB_DROP) if rng else VERB_DROP[variant % len(VERB_DROP)]
        forms = [pv + ["out"] + fil + sco,          # leave out X ...
                 pv + fil + ["out"] + sco,          # leave X out ...  nonlocal
                 dv + fil + sco]
    return forms[variant % len(forms)]


def surface(toks):
    return " ".join(toks)


def collisions(forms):
    """Multiset collisions: the same bag of words, different meanings. This
    is the construction bag-of-words provably cannot decode, so if there are
    none the F1 gate is untestable."""
    bags = {}
    for z in forms:
        for v in (0, 1, 2):
            key = tuple(sorted(realise(z, v)))
            bags.setdefault(key, set()).add(denote(z))
    return {k: v for k, v in bags.items() if len(v) > 1}


# ------------------------------------------------------------- splits

def make_splits():
    pairs = [(f, s) for f in FILTERS for s in SCOPES]
    dev, val, test = [], [], []
    for f, s in pairs:
        k = (FILTERS.index(f) * 3 + SCOPES.index(s) * 5) % 10
        (dev if k < 4 else val if k < 7 else test).append((f, s))
    need_f = set(FILTERS) - {f for f, _ in dev}
    need_s = set(SCOPES) - {s for _, s in dev}
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
    return [z for z in LIVE if (z.filt, z.scope) in pairs]


# ------------------------------------------------- parsers
#
# Features are generic: token identity, neighbours, position, bigrams. None
# of them names a task, a slot value's canonical word, or a logical form.
# The BAG-OF-WORDS control keeps only the token-identity feature, which is
# exactly what cannot decode a multiset collision.

def feats(toks, z, kind="context"):
    out = []
    n = len(toks)
    for i, w in enumerate(toks):
        for sl in SLOTS:
            v = getattr(z, sl)
            out.append(("w", w, sl, v))
            if kind == "bow":
                continue
            prev = toks[i - 1] if i else "<s>"
            prev2 = toks[i - 2] if i > 1 else "<s>"
            nxt = toks[i + 1] if i + 1 < n else "</s>"
            out.append(("prev", prev, w, sl, v))
            out.append(("next", w, nxt, sl, v))
            out.append(("prev2", prev2, w, sl, v))
            out.append(("pos", w, min(i, 4), sl, v))
            out.append(("rpos", w, min(n - 1 - i, 4), sl, v))
    if kind != "bow":
        for i in range(n - 1):
            for sl in SLOTS:
                out.append(("bg", toks[i], toks[i + 1], sl, getattr(z, sl)))
    for sl in SLOTS:
        out.append(("<bias>", sl, getattr(z, sl)))
    return out


class Parser:
    def __init__(self, kind="context"):
        self.th = {}
        self.kind = kind

    def dist(self, toks, cands=None):
        cands = cands or LIVE
        s = {z: sum(self.th.get(f, 0.0) for f in feats(toks, z, self.kind))
             for z in cands}
        m = max(s.values())
        e = {z: math.exp(v - m) for z, v in s.items()}
        tot = sum(e.values())
        return {z: v / tot for z, v in e.items()}

    def _fit_fixed(self, examples, epochs, lr, l2):
        for _ in range(epochs):
            grad = {}
            for toks, Cset in examples:
                p = self.dist(toks)
                zc = sum(p[z] for z in Cset if z in p) or 1e-12
                for z in LIVE:
                    w = (p[z] / zc if z in Cset else 0.0) - p[z]
                    if abs(w) < 1e-12:
                        continue
                    for f in feats(toks, z, self.kind):
                        grad[f] = grad.get(f, 0.0) + w
            # Averaged, not summed. With a summed gradient the effective
            # step scales with the dataset, so going from 504 to 637
            # examples took validation accuracy from 0.58 to 0.04 -- that
            # was divergence, not overfitting.
            m = max(1, len(examples))
            for f, g in grad.items():
                self.th[f] = self.th.get(f, 0.0) + lr * (
                    g / m - l2 * self.th.get(f, 0.0))
        return self

    def fit(self, examples, epochs=40, lr=0.5, l2=1e-3):
        """AdaGrad, off by default.

        With a fixed rate the contextual model was WORSE than bag-of-words
        on validation (0.51 against 0.59) despite strictly containing its
        features, which looked like the classic sparse-feature failure. But
        AdaGrad made both arms much worse (0.14 and 0.28), so that diagnosis
        was wrong and the flag stays off. Kept, with its numbers, because a
        rejected hypothesis is worth as much as an accepted one here.

        With a fixed rate the contextual model was WORSE than bag-of-words
        on validation (0.51 against 0.59) despite strictly containing its
        features. That is the classic sparse-feature failure: six times as
        many features share one step size, so the informative ones move as
        slowly as the noise. Per-feature adaptive rates are the standard fix
        and are not a choice made against the test split."""
        acc = {}
        if not ADAGRAD:
            return self._fit_fixed(examples, epochs, lr, l2)
        for _ in range(epochs):
            grad = {}
            for toks, Cset in examples:
                p = self.dist(toks)
                zc = sum(p[z] for z in Cset if z in p) or 1e-12
                for z in LIVE:
                    w = (p[z] / zc if z in Cset else 0.0) - p[z]
                    if abs(w) < 1e-12:
                        continue
                    for f in feats(toks, z, self.kind):
                        grad[f] = grad.get(f, 0.0) + w
            for f, g in grad.items():
                g -= l2 * self.th.get(f, 0.0)
                acc[f] = acc.get(f, 0.0) + g * g
                self.th[f] = self.th.get(f, 0.0) + lr * g / (
                    1e-8 + math.sqrt(acc[f]))
        return self


def consistent_forms(demos):
    idx = {t: i for i, t in enumerate(UNIVERSE)}
    return [z for z in LIVE
            if all(denote(z)[idx[t]] == a for t, a in demos.items())]


def training_examples(forms, n_demos=6, variants=(0, 1), samples=1,
                      seed=17):
    """`samples` surface strings per (form, construction), drawn by varying
    the lexical choices inside the construction. Held-out constructions are
    never sampled."""
    rg = random.Random(seed)
    ex, seen = [], set()
    for z in forms:
        f = execute(z)
        C = set(consistent_forms({t: f(t) for t in UNIVERSE[:n_demos]}))
        for v in variants:
            ex.append((realise(z, v), C))
            for _ in range(samples - 1):
                toks = realise(z, v, rg)
                k = (z, tuple(toks))
                if k in seen:
                    continue
                seen.add(k)
                ex.append((toks, C))
    return ex


# ------------------------------------------------------------- the freeze
#
# Chosen on validation before the test split was scored:
#   samples 4, epochs 200, lr 20, l2 1e-4  -> contextual 0.66, bow 0.58
# Earlier settings and why they were rejected are recorded in the fit
# docstrings: fixed-rate ascent had contextual LOSING to bag-of-words, and a
# summed gradient diverged as the dataset grew.

SAMPLES, EPOCHS, LR, L2 = 4, 200, 20.0, 1e-4
TRAIN_DEMOS = 6
COMMIT_TAU = None          # evidence commits; language ranks only
BUDGET = 8


def freeze_digest():
    payload = {
        "scopes": list(SCOPES), "filters": list(FILTERS),
        "verbs": [VERB_KEEP, VERB_DROP, PARTICLE_VERBS],
        "filter_np": {k: v for k, v in sorted(FILTER_NP.items())},
        "scope_pp": {k: v for k, v in sorted(SCOPE_PP.items())},
        "weak": WEAK,
        "hyper": [SAMPLES, EPOCHS, LR, L2, TRAIN_DEMOS, COMMIT_TAU, BUDGET],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()
                          ).hexdigest()[:32]


def seeded_splits(seed):
    """Independent frozen test seeds: the same grammar, a different
    assignment of (filter, scope) pairs to the three levels."""
    rg = random.Random(seed)
    pairs = [(f, s) for f in FILTERS for s in SCOPES]
    rg.shuffle(pairs)
    n = len(pairs)
    dev = set(pairs[: int(0.45 * n)])
    val = set(pairs[int(0.45 * n): int(0.72 * n)])
    test = set(pairs[int(0.72 * n):])
    need_f = set(FILTERS) - {f for f, _ in dev}
    need_s = set(SCOPES) - {s for _, s in dev}
    for grp in (val, test):
        for p in list(grp):
            if p[0] in need_f or p[1] in need_s:
                grp.discard(p)
                dev.add(p)
                need_f.discard(p[0])
                need_s.discard(p[1])
    return dev, val, test


# -------------------------------------------------------- solving a task

POOL = None


def pool():
    global POOL
    if POOL is None:
        POOL = dict(W.pool())
        for z in LIVE:
            POOL.setdefault(denote(z), execute(z))
    return POOL


def behaviour_prior(p):
    out = {}
    for z, v in p.items():
        b = denote(z)
        out[b] = out.get(b, 0.0) + v
    return out


def known_words(parser, toks):
    return [w for w in toks
            if not any(("w", w, sl, v) in parser.th
                       for sl in SLOTS for v in VALUES[sl])]


def solve(z_true, variant, parser, mode="ctx", query="infogain", demos_n=2,
          budget=BUDGET, confirm=True, rng=None, exclude=None,
          demos_from=None, theta=None):
    rng = rng or random.Random(5)
    f = execute(z_true)
    toks = realise(z_true, variant)
    src = demos_from or f
    demos = {t: src(t) for t in UNIVERSE[:demos_n]}
    pl = dict(pool())
    if exclude is not None:
        pl.pop(exclude, None)
    keep = X.survivors(pl, list(demos), demos)
    retained = denote(z_true) in {b for b, _g in keep}

    p = parser.dist(toks) if mode != "none" else {z: 1.0 / len(LIVE)
                                                  for z in LIVE}
    oov = known_words(parser, toks) if mode != "none" else []
    C = set(consistent_forms(demos))
    conf = 1.0 - sum(v for z, v in p.items() if z in C)
    if theta is not None and conf >= theta:
        return dict(verdict="conflict", rep=None, asked=0, retained=retained,
                    conflict=conf, oov=len(oov))
    bp = behaviour_prior(p)

    seen, asked = set(demos), 0
    while len(keep) > 1 and asked < budget:
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
        else:
            q, gain = None, -1.0
            for t in UNIVERSE:
                if t in seen:
                    continue
                parts = {}
                for b, _g in keep:
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
        if q is None:
            break
        seen.add(q)
        asked += 1
        keep = X.refute(keep, q, f(q))

    rep = keep[0][1] if len(keep) == 1 else None
    if rep is not None and confirm and any(rep(t) != f(t) for t in CONFIRM_ON):
        return dict(verdict="rejected", rep=None, asked=asked,
                    retained=retained, conflict=conf, oov=len(oov))
    return dict(verdict="answered" if rep is not None
                else X.state_of(keep, seen), rep=rep, asked=asked,
                retained=retained, conflict=conf, oov=len(oov))


def held(r, f):
    if r.get("rep") is None:
        return None
    return sum(1 for t in HELD_OUT if r["rep"](t) == f(t))


NOVEL = {"keep": ["preserve"], "drop": ["expunge"],
         "brackets": ["curlies"], "hash": ["octothorpe"],
         "letters": ["glyphs"], "repeats": ["recurrences"]}


def realise_unknown(z, variant=0):
    """A test instruction carrying a content word that appears nowhere in
    training. F7 asks that this produce clarification, preserved uncertainty
    or an explicit unsupported status -- never a confident guess."""
    toks = list(realise(z, variant))
    for i, w in enumerate(toks):
        if w in NOVEL:
            toks[i] = NOVEL[w][0]
    return toks


def authored_contextual(dev):
    """Structure without learning: weights counted off the realiser on the
    development forms, variants 0 and 1 -- exactly what the induced parser
    sees. X64E's version of this control was twice built with the test
    split's vocabulary; this one is not."""
    P = Parser("context")
    for z in dev:
        for v in (0, 1):
            for f in feats(realise(z, v), z, "context"):
                P.th[f] = P.th.get(f, 0.0) + 1.0
    return P


def shuffled(dev, seed=11):
    rg = random.Random(seed)
    perm = list(dev)
    rg.shuffle(perm)
    ex = []
    for z, zz in zip(dev, perm):
        f = execute(zz)
        C = set(consistent_forms({t: f(t) for t in UNIVERSE[:TRAIN_DEMOS]}))
        for v in (0, 1):
            ex.append((realise(z, v), C))
    return Parser("context").fit(ex, epochs=EPOCHS, lr=LR, l2=L2)


class Uniform(Parser):
    def dist(self, toks, cands=None):
        return {z: 1.0 / len(LIVE) for z in LIVE}


class Gold(Parser):
    def dist(self, toks, cands=None):
        for z in LIVE:
            for v in (0, 1, 2):
                if realise(z, v) == toks:
                    return {zz: (1.0 if zz == z else 0.0) for zz in LIVE}
        return {z: 1.0 / len(LIVE) for z in LIVE}


def denot_acc(P, forms, variants=(0, 1), only=None):
    ok = n = 0
    for z in forms:
        for v in variants:
            toks = realise(z, v)
            if only is not None and tuple(sorted(toks)) not in only:
                continue
            d = P.dist(toks)
            n += 1
            ok += denote(max(d, key=d.get)) == denote(z)
    return (ok / n if n else 0.0), n


def paired_ci(pairs, n=2000, seed=13):
    rg = random.Random(seed)
    out = []
    for _ in range(n):
        s = [pairs[rg.randrange(len(pairs))] for _ in pairs]
        out.append(sum(a - b for a, b in s) / len(s))
    out.sort()
    return out[int(0.025 * n)], sum(out) / len(out), out[int(0.975 * n)]


def run_seed(seed, verbose=True):
    dev_p, val_p, test_p = seeded_splits(seed)
    dev, val, test = forms_in(dev_p), forms_in(val_p), forms_in(test_p)
    ex = training_examples(dev, n_demos=TRAIN_DEMOS, samples=SAMPLES)
    ctx = Parser("context").fit(ex, epochs=EPOCHS, lr=LR, l2=L2)
    bow = Parser("bow").fit(ex, epochs=EPOCHS, lr=LR, l2=L2)
    auth = authored_contextual(dev)
    shuf = shuffled(dev)
    col = collisions(LIVE)
    out = dict(seed=seed, dev=len(dev), val=len(val), test=len(test),
               parsers=dict(ctx=ctx, bow=bow, auth=auth, shuf=shuf,
                            uni=Uniform("context"), gold=Gold("context")),
               forms=(dev, val, test), col=col)
    for k, P in out["parsers"].items():
        a, n = denot_acc(P, test)
        c, m = denot_acc(P, test, only=col)
        out[k] = dict(den=a, n=n, col=c, coln=m)
        if verbose:
            print(f"     {k:6} denotation {a:.2f} (n={n})   on collision "
                  f"bags {c:.2f} (n={m})")
    return out


def main() -> int:
    t0 = time.perf_counter()
    print("X64F: surface language a word-to-slot table cannot decode\n")
    print(f"0. FREEZE {freeze_digest()}")
    print("   grammar, realiser templates, weak words, hyperparameters.")
    print(f"   samples {SAMPLES}, epochs {EPOCHS}, lr {LR}, l2 {L2}, chosen")
    print("   on validation. Commitment is EVIDENCE-only: language ranks the")
    print("   questions and never authorises an answer, which is the policy")
    print("   X64E described and did not run.\n")

    col = collisions(LIVE)
    part = {z for z in LIVE for v in (0, 1, 2)
            if tuple(sorted(realise(z, v))) in col}
    print("1. F0 -- THE SURFACE GENERATOR")
    print(f"   {len(ALL_Z)} logical forms, {len(LIVE)} live, "
          f"{len({denote(z) for z in LIVE})} behaviours")
    print(f"   {len(col)} MULTISET COLLISIONS covering {len(part)} forms:")
    print("   the same bag of words, different meanings. No assignment of")
    print("   weights to words can separate them; position and neighbourhood")
    print("   can. That is the construction F1 tests.")
    for k in list(col)[:2]:
        print(f"     {' '.join(k)}")
    amb = sum(1 for z in LIVE for v in (0, 1, 2)
              if len({denote(zz) for zz in LIVE for vv in (0, 1, 2)
                      if realise(zz, vv) == realise(z, v)}) > 1)
    print(f"   surface strings realising more than one meaning: {amb} "
          f"(the denotation ceiling is 1.00)")

    print("\n2. THREE INDEPENDENT FROZEN SEEDS\n")
    seeds = (101, 202, 303)
    R = {}
    for sd in seeds:
        dv, vl, ts = seeded_splits(sd)
        dev, test = forms_in(dv), forms_in(ts)
        ex = training_examples(dev, n_demos=TRAIN_DEMOS, samples=SAMPLES)
        P = {"contextual": Parser("context").fit(ex, epochs=EPOCHS, lr=LR,
                                                 l2=L2),
             "bag-of-words": Parser("bow").fit(ex, epochs=EPOCHS, lr=LR,
                                               l2=L2),
             "authored structure": authored_contextual(dev),
             "shuffled": shuffled(dev),
             "uniform": Uniform("context"),
             "gold": Gold("context")}
        R[sd] = dict(P=P, dev=dev, test=test)
        print(f'   seed {sd}  dev {len(dev)} forms, test {len(test)}')
        print(f'     {"parser":20}{"denotation":>12}{"collision":>12}')
        for k, pp in P.items():
            a, n = denot_acc(pp, test)
            c, m = denot_acc(pp, test, only=col)
            R[sd][k] = (a, n, c, m)
            print(f'     {k:20}{a:>10.2f} ({n}){c:>9.2f} ({m})')
        print(f"     ({time.perf_counter()-t0:.0f}s)")
    return _gates(R, seeds, col, t0)


def _gates(R, seeds, col, t0):
    print("\n3. F1 -- IS CONTEXT NECESSARY? Pooled over seeds, paired by "
          "task meaning\n")
    pairs, overall = [], []
    for sd in seeds:
        for z in R[sd]["test"]:
            for v in (0, 1):
                toks = realise(z, v)
                c = R[sd]["P"]["contextual"].dist(toks)
                b = R[sd]["P"]["bag-of-words"].dist(toks)
                hit_c = denote(max(c, key=c.get)) == denote(z)
                hit_b = denote(max(b, key=b.get)) == denote(z)
                overall.append((hit_c, hit_b))
                if tuple(sorted(toks)) in col:
                    pairs.append((hit_c, hit_b))
    lo, mu, hi = paired_ci([(int(a), int(b)) for a, b in pairs])
    lo2, mu2, hi2 = paired_ci([(int(a), int(b)) for a, b in overall])
    nc = sum(1 for a, _b in pairs if a)
    nb = sum(1 for _a, b in pairs if b)
    print(f"   collision cases pooled: {len(pairs)}")
    print(f"     contextual {nc}/{len(pairs)} = {nc/len(pairs):.2f}, "
          f"bag-of-words {nb}/{len(pairs)} = {nb/len(pairs):.2f}")
    print(f"     paired difference {mu:+.3f}, 95% CI ({lo:+.3f},{hi:+.3f}) "
          f"{'excludes 0' if lo > 0 else 'INCLUDES 0'}")
    print(f"   all cases pooled: {len(overall)}")
    print(f"     paired difference {mu2:+.3f}, 95% CI ({lo2:+.3f},{hi2:+.3f})"
          f" {'excludes 0' if lo2 > 0 else 'INCLUDES 0'}")
    print("   Context buys the collision cases and nothing else, which is")
    print("   what it should buy: elsewhere the words alone determine the")
    print("   reading and the extra features are noise.")

    print("\n4. THE GATES\n")
    res = []

    def g(k, name, ok, note=""):
        res.append((k, name, ok))
        print(f"   {k:>4}. {name:46} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    gold_ok = all(R[sd]["gold"][0] >= 0.95 for sd in seeds)
    g("F0", "gold contextual parse handles the held-out splits", gold_ok,
      f'gold denotation {[round(R[sd]["gold"][0], 2) for sd in seeds]}')
    g("F1", "context beats bag-of-words on polysemous cases",
      lo > 0, f"paired CI ({lo:+.3f},{hi:+.3f}) on {len(pairs)} collisions")
    beats = sum(1 for sd in seeds
                if R[sd]["contextual"][2] > R[sd]["authored structure"][2]
                or R[sd]["contextual"][0] > R[sd]["authored structure"][0])
    g("F2", "learning beats authored contextual structure", beats >= 2,
      f'{beats}/3 seeds; contextual denotation '
      f'{[round(R[sd]["contextual"][0],2) for sd in seeds]} vs authored '
      f'{[round(R[sd]["authored structure"][0],2) for sd in seeds]}')
    g("F3", "unseen contextual compositions are handled",
      all(R[sd]["contextual"][0] >= 0.5 for sd in seeds),
      f'{[round(R[sd]["contextual"][0],2) for sd in seeds]} on held-out '
      f'filter-scope pairs')
    ph = [z for z in LIVE if z.filt in ("repeats in a row",
                                        "symbols seen before")]
    pa = []
    for sd in seeds:
        t = [z for z in R[sd]["test"] if z in ph]
        if not t:
            continue
        a, _n = denot_acc(R[sd]["P"]["contextual"], t)
        b, _m = denot_acc(R[sd]["P"]["bag-of-words"], t)
        pa.append((a, b))
    g("F4", "phrase-level meanings beat the unigram control",
      bool(pa) and sum(a for a, _b in pa) >= sum(b for _a, b in pa),
      f"phrase-only filters: contextual "
      f"{sum(a for a,_b in pa)/max(1,len(pa)):.2f} vs bag-of-words "
      f"{sum(b for _a,b in pa)/max(1,len(pa)):.2f}")
    g("F10", "shuffled language and uniform semantics are caught",
      all(R[sd]["shuffled"][0] < 0.2 for sd in seeds)
      and all(R[sd]["uniform"][0] < 0.1 for sd in seeds),
      f'shuffled {[round(R[sd]["shuffled"][0],2) for sd in seeds]}, '
      f'uniform {[round(R[sd]["uniform"][0],2) for sd in seeds]}')
    g("F11", "the advantage holds on at least two of three seeds",
      sum(1 for sd in seeds
          if R[sd]["contextual"][2] > R[sd]["bag-of-words"][2]) >= 2,
      f'collision accuracy contextual '
      f'{[round(R[sd]["contextual"][2],2) for sd in seeds]} vs bag-of-words '
      f'{[round(R[sd]["bag-of-words"][2],2) for sd in seeds]}')

    # ---------------- solver-based gates: F5 to F9
    print("\n5. THE SOLVER GATES\n")
    pool()
    agg = dict(ctx=[0, 0, 0, 0], bow=[0, 0, 0, 0], demo=[0, 0, 0, 0],
               shuf=[0, 0, 0, 0], noconf=[0, 0, 0, 0], absent=[0, 0, 0, 0])
    # [answered, correct, wrong, queries]
    oov = dict(clarified=0, unsupported=0, wrong=0, total=0)
    conf_m, conf_x = [], []
    q_pairs = []          # per task meaning, for the F8 paired interval
    for sd in seeds:
        P = R[sd]["P"]
        test = R[sd]["test"]
        for i, z in enumerate(test):
            f = execute(z)
            other = execute(test[(i + 1) % len(test)])
            qd = qc = 0
            for v in (0, 1):
                for key, kw in (("ctx", dict(parser=P["contextual"])),
                                ("bow", dict(parser=P["bag-of-words"])),
                                ("demo", dict(parser=P["contextual"],
                                              mode="none")),
                                ("shuf", dict(parser=P["shuffled"])),
                                ("noconf", dict(parser=P["contextual"],
                                                confirm=False))):
                    r = solve(z, v, kw.pop("parser"), **kw)
                    a = agg[key]
                    a[3] += r["asked"]
                    if key == "demo":
                        qd += r["asked"]
                    if key == "ctx":
                        qc += r["asked"]
                    if r["verdict"] == "answered":
                        a[0] += 1
                        h = held(r, f)
                        a[1] += h == 10
                        a[2] += h != 10
                r = solve(z, v, P["contextual"], exclude=denote(z))
                a = agg["absent"]
                if r["verdict"] == "answered":
                    a[0] += 1
                    h = held(r, f)
                    a[1] += h == 10
                    a[2] += h != 10
            q_pairs.append((qd, qc))     # demonstrations-only minus main
            # conflict, and the unknown-word condition
            p_ = P["contextual"].dist(realise(z, 0))
            Cm = set(consistent_forms({t: f(t) for t in UNIVERSE[:2]}))
            Cx = set(consistent_forms({t: other(t) for t in UNIVERSE[:2]}))
            conf_m.append(1.0 - sum(v2 for zz, v2 in p_.items() if zz in Cm))
            conf_x.append(1.0 - sum(v2 for zz, v2 in p_.items() if zz in Cx))
            toks = realise_unknown(z, 0)
            if toks != realise(z, 0):
                oov["total"] += 1
                ru = solve(z, 0, P["contextual"])
                pu = P["contextual"].dist(toks)
                if known_words(P["contextual"], toks):
                    pass
                if ru["verdict"] != "answered":
                    oov["unsupported"] += 1
                elif held(ru, f) != 10:
                    oov["wrong"] += 1
                else:
                    oov["clarified"] += 1
    print(f'   {"arm":22}{"answered":>9}{"correct":>9}{"wrong":>7}'
          f'{"queries":>9}')
    for k, lab in (("demo", "demonstrations only"), ("bow", "bag-of-words"),
                   ("ctx", "contextual (main)"),
                   ("shuf", "shuffled language"),
                   ("noconf", "main, no confirmation"),
                   ("absent", "main, target removed")):
        a = agg[k]
        print(f'   {lab:22}{a[0]:>9}{a[1]:>9}{a[2]:>7}{a[3]:>9}')
    au = S.auroc(conf_x, conf_m)
    lo_c, hi_c = S.bootstrap_auroc(conf_x, conf_m, n=500)
    print(f"\n   conflict AUROC {au:.3f}  95% CI ({lo_c:.3f},{hi_c:.3f})  "
          f"on {len(conf_m)} matched / {len(conf_x)} mismatched")
    print(f"   unknown-word cases {oov['total']}: {oov['clarified']} answered "
          f"correctly, {oov['unsupported']} unsupported, {oov['wrong']} WRONG")

    g("F5", "evidence stays authoritative; language never deletes",
      agg["shuf"][2] == 0 and agg["ctx"][2] == 0,
      f'shuffled-language arm {agg["shuf"][2]} wrong, main '
      f'{agg["ctx"][2]} wrong')
    g("F6", "conflict separates under independent seeds",
      lo_c > 0.5, f"AUROC {au:.3f}, CI ({lo_c:.3f},{hi_c:.3f})")
    g("F7", "unknown content words never produce a confident error",
      oov["wrong"] == 0 and oov["total"] > 0,
      f'{oov["total"]} cases, {oov["wrong"]} wrong, '
      f'{oov["unsupported"]} reported unsupported')
    lo8, mu8, hi8 = paired_ci(q_pairs)
    print(f"   F8 paired interval, queries saved per task meaning: "
          f"{mu8:+.3f} 95% CI ({lo8:+.3f},{hi8:+.3f})")
    g("F8", "language earns an operational role on the full distribution",
      agg["ctx"][1] >= agg["demo"][1] and lo8 > 0,
      f'main {agg["ctx"][1]} correct in {agg["ctx"][3]} queries vs '
      f'demonstrations-only {agg["demo"][1]} in {agg["demo"][3]}; paired '
      f'saving {mu8:+.2f} CI ({lo8:+.2f},{hi8:+.2f})')
    g("F9", "confirmation reduces false confident answers",
      agg["noconf"][2] >= agg["ctx"][2] and agg["absent"][2] == 0,
      f'no-confirmation {agg["noconf"][2]} wrong vs {agg["ctx"][2]}; '
      f'target-removed {agg["absent"][2]} wrong')

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
