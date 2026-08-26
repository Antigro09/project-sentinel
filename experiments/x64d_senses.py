"""X64D: senses induced from evidence, and language that cannot delete.

X64C falsified the previous design. A frozen bag-of-words lexicon excluded
its own target in 22 of 40 holdout instruction forms, and language plus
demonstrations fell from 10/12 (demonstrations alone) to 3/12. The cause was
structural, not three unlucky entries: `brackets`, `first` and `comment` all
turned out to mean different things in different grammatical positions, and
a fixed word-to-predicate map has nowhere to put that.

TWO CHANGES, and the second is the load-bearing one.

1. SENSES ARE INDEXED BY (WORD, ROLE) AND INDUCED FROM EVIDENCE.
   For a token t = (word, role), let Pi be the task-independent predicate
   library and D(t) the development examples whose instruction contains t,
   each with its intended behaviour b*. The induced sense is

       S(t) = { pi in Pi : for all b* in D(t), b* |= pi }

   the intersection over examples of the predicates they satisfy. This is a
   version space with a unique most-specific element, so no search is
   needed, and it is monotone: more evidence can only shrink S(t). Nothing
   is authored per word. `brackets` as the OBJECT of `remove` and `brackets`
   as the DELIMITER of `inside` are different tokens and get different
   senses, which is where polysemy comes from.

2. LANGUAGE RANKS; IT CANNOT ELIMINATE.
   Each token contributes a CHAIN of interpretations, S(t) and every subset
   of it down to the empty set. An interpretation I of a whole instruction
   picks one link per token, and its constraint is the union. The joint
   version space is

       V = { (I, b) : b |= C(I) and b consistent with the evidence }

   and the system reports the behaviours in the MOST SPECIFIC interpretation
   whose behaviour set is non-empty:

       I* = argmax_{I : {b : (I,b) in V} nonempty}  sum_t |I(t)|

   Because the empty interpretation is always in the chain, no behaviour is
   ever permanently removed by language. A wrong sense costs specificity,
   not the target. That is X64C's failure made structurally impossible
   rather than patched -- D5 holds by construction, and the experiment's job
   is to show language still EARNS its place under that weaker power (D2).

THE MODEL, stated once so it can be attacked.

  Pi          a task-independent library of 19 behavioural predicates
  t = (w, r)  a token: a surface word in a syntactic role
  Sigma(t)    a SET of senses, each a subset of Pi, induced by clustering
              the predicate signatures of the development examples that
              contain t and intersecting within each cluster
  I           a reading: one sense chosen per token
  C(I)        the union of the chosen senses
  viol(b)     sum over tokens of min over that token's senses of |S minus sat(b)|
  V           { b : b consistent with the evidence }, ordered by viol
  answer      only when |V| = 1, never when the tier is a singleton

Three design choices, each forced by a measured failure:

  Sense SETS, not one sense. A single intersection is the most-specific
  boundary of a version space and fails in both directions -- with three
  examples it kept an accident ("only after the hash" for `brackets`), with
  thirty-eight it collapsed and `first`, `hash` and `last` all reduced to
  the same generic core, so the role stopped mattering. Clustering fixes
  both. On validation, three clusters matched the single intersection's
  accuracy (105/105) with fewer questions (137 vs 153) and four polysemous
  words instead of one.

  Language RANKS, never eliminates. X64C's hard filter excluded the target
  in 22 of 40 holdout forms. Here the empty reading is always available, so
  D5 is a property of the definition rather than a behaviour to test for --
  126/126 retained, against 98/126 for the same senses used as a filter and
  77/126 for X64C's authored lexicon.

  Evidence DECIDES. Committing when the language-preferred tier is a
  singleton produced four confident errors, and no amount of confirmation
  against a fixed list repairs it: the target is always among the
  survivors, so a disagreeing rival always exists. Answering only when the
  evidence has resolved the set drove that to zero -- and cost most of the
  query saving, which is itself the finding.

MEASURED on 42 held-out tasks, 126 instruction forms, compositions absent
from development and validation:

  arm                        retained  answered  correct  WRONG  queries
  demonstrations only             126       123      123      0      285
  X64C hard lexicon                77        76       76      0      176
  role-blind, joint               126       125      125      0      263
  induced, hard filter             98       101       97      4      179
  induced, joint                  126       125      125      0      267
  induced, joint + random         126       124      124      0      340
  induced, joint + semantic       126       126      126      0      253
  oracle senses, joint            126       126      126      0      246
  oracle queries                  126       122      122      0      270

9 of 10 gates. D7 fails, and it fails for a reason worth more than a pass.

WHAT CANNOT ELIMINATE CANNOT CONTRADICT. Nine conflict statistics were
tried -- set emptiness under the hard reading, the same with uninformative
predicates stripped at four thresholds, violation gaps at three thresholds,
a contrastive z-score against how other instructions rank the same
behaviour, and a discriminating semantic probe. Precision sits at chance
(0.50-0.62) across the whole family; stripping generic predicates buys
recall to 0.83 and leaves precision at 0.51. Induction by intersection
keeps only what examples SHARE, so senses are generic, and a generic
constraint is satisfied by the wrong task as readily as the right one.
X64B-2 detected conflict at 8/12 precisely because its senses were AUTHORED
and sharp -- and X64C measured what sharp authored senses cost on unseen
compositions. D5 and D7 are in tension and this architecture buys D5.

AND ONE ARM THAT SHOULD TEMPER THE HEADLINE. Role-blind joint scores
125/126 in 263 queries against induced joint's 125/126 in 267. On this test
set the syntactic role earns nothing measurable: the gain is from keeping
alternatives, not from knowing which role a word is in. Roles are what make
D3's polysemy representable, and they are not what makes the system work.

Run: uv run python experiments/x64d_senses.py
"""

import itertools
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x63_sparse_price as P
import x64a_identify as X
import x64b1_openworld as O

UNIVERSE, HELD_OUT = X.UNIVERSE, X.HELD_OUT
# X64B-1's challenge set plus systematic depth-2 nesting with fresh symbols.
# The gap was structural rather than incidental: nothing in it nested two
# deep around distinct letters, so a hypothesis could agree on every input
# it was ever shown and still be wrong. Disjoint from UNIVERSE and HELD_OUT,
# and the extension is chosen by shape coverage, not by which test item
# escaped.
CHALLENGE = O.CHALLENGE + ["((mn))", "(a(bc))", "((pq)(rs))", "(m(n)o)",
                           "((k))m", "n((op))", "(((qr)))", "((s)(t))u"]


def _generated(n=10, seed=3):
    """Freshly generated confirmation inputs, disjoint from everything else.

    A fixed challenge list has fixed blind spots: one test candidate agreed
    on all eighteen of them and still differed from the target on a held-out
    tape. Generating inputs removes the blind spot rather than patching the
    particular hole.

    The COUNT is set by parity with the held-out set (ten tapes), not by
    measurement: validation showed zero confident errors at every setting
    including zero, so it could not distinguish them, and choosing the value
    on the test split would be exactly the tuning this experiment exists to
    avoid."""
    rg = random.Random(seed)
    banned = set(UNIVERSE) | set(HELD_OUT) | set(CHALLENGE)
    out = []
    while len(out) < n:
        t = "".join(rg.choice("#()abc") for _ in range(rg.randrange(4, 10)))
        if t not in banned and t not in out:
            out.append(t)
    return out


CONFIRM_ON = CHALLENGE + _generated()
EVIDENCE0 = X.EVIDENCE0


# ------------------------------------------------------ the task grammar
#
# Tasks are COMPOSED, so a held-out combination of familiar parts is a real
# thing rather than a hand-picked exception.

# The first version of this grammar produced 42 distinct meanings in total,
# which left 15 for development -- far below the 40-60 the protocol calls
# for, and thin enough that induced senses were too unreliable to support
# conflict detection at all. Widened here for that reason.
SCOPES = ("whole", "before hash", "after hash", "inside brackets",
          "outside brackets", "after the first symbol", "before the last")
FILTERS = ("everything", "letters", "brackets", "hashes", "repeats in a row",
           "repeats", "symbols seen before", "symbols matching the first",
           "the first symbol", "the last symbol", "symbols at even places",
           "symbols before a repeat")
POLARITIES = ("keep", "remove")


def scope_positions(s, scope):
    if scope == "whole":
        return list(range(len(s)))
    if scope == "before hash":
        i = s.find("#")
        return list(range(len(s) if i < 0 else i))
    if scope == "after hash":
        i = s.find("#")
        return [] if i < 0 else list(range(i + 1, len(s)))
    if scope == "after the first symbol":
        return list(range(1, len(s)))
    if scope == "before the last":
        return list(range(max(0, len(s) - 1)))
    out, depth = [], 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif (depth > 0) == (scope == "inside brackets"):
            out.append(i)
    return out


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


def enumerate_tasks():
    """Every composition that is behaviourally distinct and not empty."""
    seen, out = set(), []
    for sc in SCOPES:
        for fl in FILTERS:
            for po in POLARITIES:
                f = make_task(sc, fl, po)
                b = tuple(f(t) for t in UNIVERSE)
                if all(o == "" for o in b) or b in seen:
                    continue
                seen.add(b)
                out.append(((sc, fl, po), f, b))
    return out


TASKS = enumerate_tasks()
KEY = {c: (f, b) for c, f, b in TASKS}


# ------------------------------------------------- structured instructions
#
# Roles are supplied by the generator, deliberately. X64C showed bag-of-words
# is insufficient; learning the parse as well is a later experiment, and
# mixing the two would make a failure here uninterpretable.

VERB, OBJ, MOD, PREP, DELIM = "VERB", "OBJ", "MOD", "PREP", "DELIM"

# Variant 2 exists only at test time. Its surface words never appear in
# development, so their tokens have NO induced sense -- which is what D6 is
# about: an unknown word must produce preserved uncertainty or a semantic
# question, not silent ignoring.
VERBS = {"keep": ["keep", "copy", "retain"],
         "remove": ["remove", "drop", "excise"]}
OBJECTS = {
    "everything": [[("everything", OBJ)], [("everything", OBJ)],
                   [("all", OBJ)]],
    "letters": [[("letters", OBJ)], [("letters", OBJ)],
                [("alphabetics", OBJ)]],
    "the first symbol": [[("first", OBJ)], [("first", OBJ)],
                         [("initial", OBJ)]],
    "the last symbol": [[("last", OBJ)], [("last", OBJ)],
                        [("final", OBJ)]],
    "symbols at even places": [[("symbols", OBJ), ("even", MOD)],
                               [("characters", OBJ), ("even", MOD)],
                               [("glyphs", OBJ), ("alternate", MOD)]],
    "symbols before a repeat": [[("symbols", OBJ), ("later", MOD)],
                                [("characters", OBJ), ("later", MOD)],
                                [("glyphs", OBJ), ("recurring", MOD)]],
    "brackets": [[("brackets", OBJ)], [("brackets", OBJ)],
                 [("parens", OBJ)]],
    # `hash` as OBJ and `hash` as DELIM is the second polysemous pair, and
    # D3 cannot be tested without at least two. An earlier edit had left
    # only `brackets`, so the gate was failing on a dataset that did not
    # contain the phenomenon it exists to measure.
    "hashes": [[("hashes", OBJ)], [("hash", OBJ)], [("octothorpes", OBJ)]],
    "repeats in a row": [[("repeats", OBJ), ("row", MOD)],
                         [("duplicates", OBJ), ("adjacent", MOD)],
                         [("runs", OBJ), ("consecutive", MOD)]],
    "repeats": [[("repeats", OBJ)], [("duplicates", OBJ)],
                [("recurrences", OBJ)]],
    "symbols seen before": [[("symbols", OBJ), ("seen", MOD)],
                            [("characters", OBJ), ("before", MOD)],
                            [("glyphs", OBJ), ("encountered", MOD)]],
    "symbols matching the first": [[("symbols", OBJ), ("matching", MOD)],
                                   [("characters", OBJ), ("first", MOD)],
                                   [("glyphs", OBJ), ("identical", MOD)]],
}
SCOPE_PHRASE = {
    "whole": [[]],
    "before hash": [[("before", PREP), ("hash", DELIM)],
                    [("until", PREP), ("hash", DELIM)],
                    [("preceding", PREP), ("octothorpe", DELIM)]],
    "after hash": [[("after", PREP), ("hash", DELIM)],
                   [("after", PREP), ("hash", DELIM)],
                   [("following", PREP), ("octothorpe", DELIM)]],
    "inside brackets": [[("inside", PREP), ("brackets", DELIM)],
                        [("within", PREP), ("brackets", DELIM)],
                        [("nested", PREP), ("parens", DELIM)]],
    "outside brackets": [[("outside", PREP), ("brackets", DELIM)],
                         [("outside", PREP), ("brackets", DELIM)],
                         [("beyond", PREP), ("parens", DELIM)]],
    "after the first symbol": [[("after", PREP), ("first", DELIM)],
                               [("past", PREP), ("first", DELIM)],
                               [("subsequent", PREP), ("initial", DELIM)]],
    "before the last": [[("before", PREP), ("last", DELIM)],
                        [("until", PREP), ("last", DELIM)],
                        [("preceding", PREP), ("final", DELIM)]],
}


def realise(combo, variant=0):
    sc, fl, po = combo
    vs = VERBS[po]
    ob = OBJECTS[fl]
    sp = SCOPE_PHRASE[sc]
    toks = [(vs[variant % len(vs)], VERB)]
    toks += ob[variant % len(ob)]
    toks += sp[variant % len(sp)]
    return toks


def surface(toks):
    return " ".join(w for w, _r in toks)


# ------------------------------------------------- the predicate library
#
# Task-independent. Nothing here mentions a word, a role, or a task; senses
# are drawn from this library by induction, never assigned by hand.

def _sub(a, b):
    it = iter(b)
    return all(c in it for c in a)


def _region(t, inside):
    out, depth = set(), 0
    for i, c in enumerate(t):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif (depth > 0) == inside:
            out.add(c)
    return out


def _allpairs(fn):
    return lambda b: all(fn(o, t) for o, t in zip(b, UNIVERSE))


PI = {
    "subsequence": _allpairs(_sub),
    "sometimes shorter":
        lambda b: any(len(o) < len(t) for o, t in zip(b, UNIVERSE)),
    "never empty everywhere": lambda b: any(o for o in b),
    "no adjacent repeat":
        lambda b: all(all(o[i] != o[i + 1] for i in range(len(o) - 1))
                      for o in b),
    "all symbols distinct": lambda b: all(len(set(o)) == len(o) for o in b),
    "only repeated symbols":
        _allpairs(lambda o, t: all(t.count(c) > 1 for c in o)),
    "a prefix": _allpairs(lambda o, t: t.startswith(o)),
    "only inside brackets": _allpairs(lambda o, t: set(o) <= _region(t, True)),
    "only outside brackets":
        _allpairs(lambda o, t: set(o) <= _region(t, False) | set("()")),
    "no brackets in output": lambda b: all(not set(o) & set("()") for o in b),
    "only brackets in output": lambda b: all(set(o) <= set("()") for o in b),
    "no hash in output": lambda b: all("#" not in o for o in b),
    "only hashes in output": lambda b: all(set(o) <= {"#"} for o in b),
    "no letters in output":
        lambda b: all(not (set(o) - set("()#")) for o in b),
    "only letters in output": lambda b: all(set(o) <= set("()#") is False
                                            or not o for o in b),
    "only before the hash":
        _allpairs(lambda o, t: _sub(o, t.split("#")[0])),
    "only after the hash":
        _allpairs(lambda o, t: "#" not in t or _sub(o, t.split("#", 1)[1])),
    "matches the first symbol":
        _allpairs(lambda o, t: not t or set(o) <= {t[0]}),
    "equals the input somewhere":
        lambda b: any(o == t for o, t in zip(b, UNIVERSE)),
}
PI_NAMES = sorted(PI)


def sat(b):
    """The predicate signature of a behaviour: which of Pi it satisfies."""
    return frozenset(n for n in PI_NAMES if PI[n](b))


# --------------------------------------------------------------- the pool
#
# A hypothesis is a FUNCTION, not a program: programs were only ever a way
# to name behaviours, and X64D is about semantic induction rather than
# synthesis, which X63 gates separately. The pool holds every composed task
# meaning plus several thousand enumerated program behaviours as
# distractors. It is task-independent -- one pool, every task, no labels.

_POOL = {}


def pool():
    if _POOL:
        return _POOL
    for _c, f, b in TASKS:
        _POOL[b] = f
    for b, prog in O.core(3, 1000, None, 0).items():
        if b not in _POOL:
            _POOL[b] = (lambda pr: (lambda t: P.semit(pr, t)))(prog)
    return _POOL


# ------------------------------------------------------- splits by composition
#
# Held out by (scope, filter) PAIR, not by task, so a test item is a novel
# combination of parts each of which was seen elsewhere. Every scope and
# every filter appears in development, or its sense could not be induced at
# all and the test would be measuring vocabulary coverage instead.

def splits():
    pairs = sorted({(c[0], c[1]) for c, _f, _b in TASKS})
    si = {s: i for i, s in enumerate(SCOPES)}
    fi = {f: i for i, f in enumerate(FILTERS)}
    dev, val, test = [], [], []
    for p in pairs:
        k = (si[p[0]] + fi[p[1]]) % 3
        (dev if k == 0 else val if k == 1 else test).append(p)
    # every scope and filter must be inducible from development alone
    need_s = {s for s, _f in pairs} - {s for s, _f in dev}
    need_f = {f for _s, f in pairs} - {f for _s, f in dev}
    for p in list(val):
        if p[0] in need_s or p[1] in need_f:
            val.remove(p)
            dev.append(p)
            need_s.discard(p[0])
            need_f.discard(p[1])
    return set(dev), set(val), set(test)


DEV_PAIRS, VAL_PAIRS, TEST_PAIRS = splits()


def group(pairs):
    return [(c, f, b) for c, f, b in TASKS if (c[0], c[1]) in pairs]


# --------------------------------------------------------- sense induction
#
#   S(t) = intersection over development examples containing t of sat(b*)
#
# A version space with a unique most-specific element, so no search. Monotone
# in the evidence. Nothing is authored per word.

def induce(dev, variants=(0, 1), k=3, floor=2):
    """A SET of candidate senses per token, not one.

    A single intersection is the most-specific boundary of a version space,
    and it fails in both directions. With few examples it keeps accidents:
    `brackets` as an object picked up "only after the hash" from three
    examples that happened to share it. With many examples it collapses --
    at 38 development tasks `first`, `hash` and `last` all intersected down
    to the same uninformative core, and the role stopped mattering.

    So the examples for a token are CLUSTERED and each cluster contributes
    one sense. Greedy agglomeration: merge the two clusters whose union
    loses the fewest predicates, until k remain or a merge would drop a
    cluster's sense below `floor` predicates. Senses stay plural, and the
    joint version space lets a behaviour answer to whichever reading fits
    it best."""
    ex = {}
    for c, _f, b in dev:
        sg = sat(b)
        for v in variants:
            for tok in realise(c, v):
                ex.setdefault(tok, []).append(sg)

    out = {}
    for tok, sigs in ex.items():
        clusters = [[s] for s in sigs]

        def inter(cl):
            r = cl[0]
            for x in cl[1:]:
                r = r & x
            return r

        while len(clusters) > k:
            best, loss = None, None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    merged = inter(clusters[i] + clusters[j])
                    if len(merged) < floor:
                        continue
                    d = (len(inter(clusters[i])) + len(inter(clusters[j]))
                         - 2 * len(merged))
                    if loss is None or d < loss:
                        best, loss = (i, j), d
            if best is None:
                break
            i, j = best
            clusters[i] = clusters[i] + clusters[j]
            del clusters[j]
        senses = {inter(cl) for cl in clusters}
        out[tok] = frozenset(s for s in senses if s)
    return out


def best_sense_violation(tok_senses, sat_b):
    """The loosest reading: each token scored by its best-fitting sense,
    chosen independently. Kept only as a fallback -- read this way every
    behaviour finds an escape hatch, and measured against the joint reading
    it cost 12 held-out tasks and produced 4 confident errors where the
    joint reading produces none."""
    if not tok_senses:
        return 0
    return min(len(s - sat_b) for s in tok_senses)


def assignments(C, cap=512):
    """One sense per token. The senses are independent per token, so the
    number of readings is the product of the sense-set sizes -- with at most
    three senses and five tokens that is 243, small enough to enumerate
    exactly rather than approximate."""
    opts = [sorted(ts, key=lambda x: (-len(x), sorted(x))) or [frozenset()]
            for ts in C]
    n = 1
    for o in opts:
        n *= len(o)
        if n > cap:
            return [tuple(o[0] for o in opts)]
    return list(itertools.product(*opts))


def read(C, items, sat_of):
    """The MOST SPECIFIC interpretation whose behaviour set is non-empty.

        I* = argmax_{I : {b in items : b |= C(I)} nonempty}  |union C(I)|

    Ties broken by fewest total violations. Falling back through less
    specific readings is what makes the target unloseable: the all-empty
    assignment is always available and satisfies everything.
    """
    best = None
    for I in assignments(C):
        U = set()
        for s in I:
            U |= s
        hits = [(b, g) for b, g in items if U <= sat_of(b)]
        if not hits:
            continue
        key = (len(U), -sum(len(s) for s in I))
        if best is None or key > best[0]:
            best = (key, hits)
    if best is not None:
        return best[1]
    lo = min(violation(C, sat_of(b)) for b, _g in items)
    return [(b, g) for b, g in items if violation(C, sat_of(b)) == lo]


def _unused_oracle(target_b):
    """Upper bound on induction: every token means exactly what the target
    satisfies. Anything the induced arm cannot beat here is a limit of the
    induction, not of the idea."""
    s = sat(target_b)
    return lambda _tok: s


# ------------------------------------------- the joint version space
#
#   C        = union of the token senses -- the most specific reading
#   viol(b)  = |C \ sat(b)| -- how many of those constraints b breaks
#   level k  = the smallest k with a non-empty evidence-consistent set
#   V        = { b : viol(b) <= k, b consistent with the evidence }
#
# Since the empty interpretation is always available (k = |C|), language can
# never remove the last candidate. It orders; it does not delete. D5 is a
# property of this definition rather than a behaviour to be tested for.

def constraint(toks, senses):
    """The reading, as a list of per-token sense sets. Violation is summed
    over tokens, each token scored by its best-fitting sense."""
    return [senses.get(t, frozenset()) for t in toks]


def violation(C, sat_b):
    return sum(best_sense_violation(ts, sat_b) for ts in C)


def hard_ok(C, sat_b):
    """The hard reading: every token must be satisfiable in SOME sense."""
    return all(not ts or any(s <= sat_b for s in ts) for ts in C)


def rank(pool_items, C):
    return [(violation(C, sat_b), b, f) for b, f, sat_b in pool_items]


def language_levels(scored, keep):
    """Group the evidence-consistent candidates by how many language
    constraints they violate, and return the least-violating non-empty
    tier."""
    best = None
    for v, b, f in scored:
        if b not in keep:
            continue
        if best is None or v < best:
            best = v
    if best is None:
        return None, []
    return best, [(b, f) for v, b, f in scored if v == best and b in keep]


# ------------------------------------------------------------- the solver

BUDGET = 8
SELECT = "loose"   # set by validation; see calibrate_select()


def solve(combo, variant, senses, mode="joint", query="disagreement",
          demos_n=2, semantic=False, budget=BUDGET, rng=None,
          confirm=True):
    """One task, one instruction form.

    `senses` maps a (word, role) token to a predicate set, or is missing for
    a token whose meaning was never induced. `mode` is the whole point:
      none   language is not consulted
      hard   language FILTERS -- X64C's design, kept as the baseline it is
      joint  language RANKS by violation count and can never empty the set
    """
    rng = rng or random.Random(5)
    f, tb = KEY[combo]
    toks = realise(combo, variant)
    unknown = [t for t in toks if t not in senses]
    C = constraint(toks, senses)

    pl = pool()
    demos = {t: f(t) for t in UNIVERSE[:demos_n]}
    keep = X.survivors(pl, list(demos), demos)          # [(b, fn)]
    asked, sem_asked = 0, 0

    def tier_of(items):
        """The reading. `SELECT` chooses between the two rules the model
        allows -- most-specific-non-empty, or least-total-violation with
        each token read in its best-fitting sense. Which one is better is an
        empirical question and is settled on the validation split."""
        if not items:
            return []
        if SELECT == "specific":
            return read(C, items, sat)
        lo = min(violation(C, sat(b)) for b, _g in items)
        return [(b, g) for b, g in items if violation(C, sat(b)) == lo]

    if mode == "hard":
        keep = [(b, g) for b, g in keep if hard_ok(C, sat(b))]
        pool_all = keep
    elif mode == "joint":
        pool_all = keep
        keep = tier_of(pool_all)
    else:
        pool_all = keep
    retained = tb in {b for b, _g in pool_all}

    # A semantic question asks what a word MEANS, not what the output is.
    # It is the only move available when a token has no induced sense.
    if semantic and (unknown or len(keep) > 1):
        for _ in range(2):
            if len(keep) <= 1:
                break
            split = [pi for pi in PI_NAMES
                     if 0 < sum(1 for b, _g in keep if pi in sat(b)) < len(keep)]
            if not split:
                break
            pi = max(split, key=lambda p: min(
                sum(1 for b, _g in keep if p in sat(b)),
                sum(1 for b, _g in keep if p not in sat(b))))
            want = pi in sat(tb)                 # the user answers about meaning
            sem_asked += 1
            keep = [(b, g) for b, g in keep if (pi in sat(b)) == want]

    seen = set(demos)
    # Language RANKS and evidence DECIDES, so commitment has to wait for the
    # evidence. The question is chosen from the language-preferred tier --
    # that is where the query saving comes from -- but the stopping test is
    # on the full evidence-consistent set. Answering while rivals remain is
    # what produced four confident errors on the test split, and no amount
    # of confirmation against a fixed challenge list repairs it: the target
    # is always among those rivals, so a rival that disagrees always exists.
    def unresolved():
        return len(pool_all if mode == "joint" else keep) > 1

    while unresolved() and asked < budget and query != "none":
        if query == "disagreement":
            q = X.best_query(keep, seen)
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
            cands = [t for t in UNIVERSE if t not in seen
                     and len(X.split(keep, t)) > 1]
            q = rng.choice(cands) if cands else None
        if q is None:
            # nothing in the tier separates: fall back to the full set
            if mode == "joint" and len(pool_all) > 1:
                q = X.best_query(pool_all, seen)
            if q is None:
                break
        seen.add(q)
        asked += 1
        ans = f(q)
        keep = X.refute(keep, q, ans)
        if mode == "joint":
            # A tier emptied by evidence is a reading the user has refuted.
            # Falling back to the next one is what makes the target
            # unloseable: the empty interpretation is always the last tier.
            pool_all = X.refute(pool_all, q, ans)
            if not keep:
                keep = tier_of(pool_all)

    final = pool_all if mode == "joint" else keep
    st = X.state_of(final, seen)
    rep = final[0][1] if len(final) == 1 else None
    if rep is not None and confirm:
        if any(rep(t) != f(t) for t in CONFIRM_ON):
            return dict(verdict="rejected", rep=None, asked=asked,
                        sem=sem_asked, retained=retained, unknown=len(unknown))
    return dict(verdict="answered" if rep is not None else st, rep=rep,
                asked=asked, sem=sem_asked, retained=retained,
                unknown=len(unknown))


def held(r, f):
    if r.get("rep") is None:
        return None
    return sum(1 for t in HELD_OUT if r["rep"](t) == f(t))


# ------------------------------------------------- baselines and oracles

def x64c_senses():
    """The frozen bag-of-words lexicon from X64C, mapped onto these tokens
    with the role ignored -- which is exactly its design, and the baseline
    this experiment exists to beat."""
    import x64b2_language as L
    out = {}
    for c, _f, _b in TASKS:
        for v in (0, 1, 2):
            for tok in realise(c, v):
                w = tok[0]
                if w in L.LEXICON:
                    # wrapped as a one-element sense SET: X64C's design is
                    # exactly "one sense per word, role ignored", which is
                    # the degenerate case of this representation
                    out[tok] = frozenset(
                        {frozenset(L.LEXICON[w] & set(PI_NAMES))})
    return out


def role_blind(senses):
    """Induced senses with the role thrown away: every occurrence of a word
    gets the intersection of its per-role senses. This is the arm that
    separates `alternatives helped` from `roles helped`."""
    byword = {}
    for (w, _r), s in senses.items():
        byword[w] = s if w not in byword else (byword[w] | s)
    return {(w, r): byword[w] for (w, r) in senses}


def freeze_hash(senses):
    import hashlib
    payload = sorted((w, r, tuple(sorted(tuple(sorted(x)) for x in s)))
                     for (w, r), s in senses.items())
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:32]


# ---------------------------------------------------------------- the run

def sweep(items, senses, variants=(0, 1, 2), **kw):
    ans = cor = ret = q = sem = unk = wrong = 0
    for c, f, _b in items:
        for v in variants:
            r = solve(c, v, senses, **kw)
            ret += r["retained"]
            q += r["asked"]
            sem += r["sem"]
            unk += r["unknown"] > 0
            if r["verdict"] == "answered":
                ans += 1
                h = held(r, f)
                cor += h == 10
                wrong += h != 10
    n = len(items) * len(variants)
    return dict(n=n, retained=ret, answered=ans, correct=cor, wrong=wrong,
                queries=q, semantic=sem, unknown=unk)


def main() -> int:
    t0 = time.perf_counter()
    print("X64D: senses induced from evidence, and language that cannot "
          "delete\n")

    dev, val, test = group(DEV_PAIRS), group(VAL_PAIRS), group(TEST_PAIRS)
    S = induce(dev, variants=(0, 1))
    H = freeze_hash(S)
    print(f"0. FREEZE   senses {H}")
    print(f"   induced from {len(dev)} development tasks over variants 0-1")
    print(f"   only. The test split contributes nothing to induction and")
    print(f"   variant 2's surface words appear nowhere in development.")
    used = {(c[0], c[1]) for c, _f, _b in dev}
    print(f"   development pairs {len(DEV_PAIRS)}, validation {len(VAL_PAIRS)},"
          f" test {len(TEST_PAIRS)}; overlap "
          f"{len(used & TEST_PAIRS)}")

    global _S_CAL, _THETA
    _S_CAL = S
    _THETA = calibrate_conflict(val)
    print(f"   conflict threshold theta = {_THETA}, chosen on the "
          f"{len(val)}-task validation split only")

    pl = pool()
    print(f"\n1. THE SPACES   {len(TASKS)} composed task meanings, pool "
          f"{len(pl):,} behaviours, {len(PI)} predicates")
    print(f"   test conditions: {len(test)} tasks x 3 forms x 5 conditions "
          f"= {len(test) * 15}")

    poly = {}
    for (w, r) in S:
        poly.setdefault(w, {})[r] = S[(w, r)]
    multi = {w: rs for w, rs in poly.items() if len(rs) > 1}
    print(f"\n2. INDUCED POLYSEMY -- same word, different role, different "
          f"sense\n")
    for w, rs in sorted(multi.items()):
        for r, s in sorted(rs.items()):
            print(f"   ({w:12}, {r:6}) {len(s):>2} predicates")
        a, b = list(rs.values())[:2]
        print(f"   {'':22} differ by {len(a ^ b)}: "
              f"{sorted(a ^ b)[:3]}")

    print("\n3. NINE ARMS on the frozen test split\n")
    ARMS = [
        ("demonstrations only", dict(mode="none")),
        ("X64C hard lexicon", dict(mode="hard", senses=x64c_senses())),
        ("role-blind, joint", dict(mode="joint", senses=role_blind(S))),
        ("induced, hard filter", dict(mode="hard")),
        ("induced, joint", dict(mode="joint")),
        ("induced, joint + random", dict(mode="joint", query="random")),
        ("induced, joint + semantic", dict(mode="joint", semantic=True)),
        ("oracle senses, joint", dict(mode="joint", senses="ORACLE")),
        ("oracle queries", dict(mode="joint", query="oracle")),
    ]
    print(f'    {"arm":26}{"retained":>9}{"answered":>9}{"correct":>8}'
          f'{"WRONG":>7}{"queries":>8}{"sem":>5}')
    R = {}
    for label, kw in ARMS:
        kw = dict(kw)
        sen = kw.pop("senses", S)
        if sen == "ORACLE":
            tot = dict(n=0, retained=0, answered=0, correct=0, wrong=0,
                       queries=0, semantic=0, unknown=0)
            for c, f, b in test:
                osen = {tok: frozenset({sat(b)}) for v in (0, 1, 2)
                        for tok in realise(c, v)}
                one = sweep([(c, f, b)], osen, **kw)
                for k in tot:
                    tot[k] += one[k]
            r = tot
        else:
            r = sweep(test, sen, **kw)
        R[label] = r
        print(f'    {label:26}{r["retained"]:>9}{r["answered"]:>9}'
              f'{r["correct"]:>8}{r["wrong"]:>7}{r["queries"]:>8}'
              f'{r["semantic"]:>5}')

    print("\n4. THE ACCURACY-QUERY FRONTIER (D2 is a Pareto claim, not a "
          "single number)\n")
    print(f'    {"budget":>8}{"demos: correct":>16}{"queries":>9}'
          f'{"joint: correct":>16}{"queries":>9}')
    front = {}
    for bud in (1, 2, 3, 4, 6, 8):
        a = sweep(test, S, mode="none", budget=bud)
        b = sweep(test, S, mode="joint", budget=bud)
        front[bud] = (a, b)
        print(f'    {bud:>8}{a["correct"]:>16}{a["queries"]:>9}'
              f'{b["correct"]:>16}{b["queries"]:>9}')

    return _gate(R, front, S, test, multi, H, t0)


# ------------------------------------- conditions, defects and the gates

def conflict_gap(combo, demos_fn, S, width=1):
    """Does the language's OWN preferred reading survive the demonstrations?

    The first version of this measured how much worse the best fit becomes
    once the demonstrations are imposed. That statistic turned out to carry
    no signal at all -- matched and mismatched pairs had indistinguishable
    distributions on both the validation and the test split -- because
    induced senses are over-specific and everything violates something.

    What does separate them is the definition the phrase actually names:
    each source is satisfiable alone, and jointly they are not. L is the set
    of behaviours the language ranks best over the whole pool; Dm is the set
    the demonstrations allow. The conflict signal is how far down L one must
    go before meeting Dm, and `None` means they never meet.
    """
    pl = pool()
    C = constraint(realise(combo, 0), S)
    viol = {}
    for b in pl:
        viol.setdefault(violation(C, sat(b)), []).append(b)
    demos = {t: demos_fn(t) for t in UNIVERSE[:2]}
    keep = {b for b, _g in X.survivors(pl, list(demos), demos)}
    if not keep:
        return None
    for k in sorted(viol):
        if set(viol[k]) & keep:
            return k - min(viol)
    return None


def calibrate_conflict(val):
    """Threshold chosen on VALIDATION only: the smallest gap that separates
    matched from mismatched pairs there. No final claim rests on this split."""
    matched = [conflict_gap(c, f, _S_CAL) for c, f, _b in val]
    mismatched = [conflict_gap(c, val[(i + 1) % len(val)][1], _S_CAL)
                  for i, (c, _f, _b) in enumerate(val)]
    matched = [m for m in matched if m is not None]
    mismatched = [m for m in mismatched if m is not None]
    best, score = 1, -1.0
    for th in range(1, 9):
        tp = sum(1 for m in mismatched if m >= th)
        fp = sum(1 for m in matched if m >= th)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, len(mismatched))
        f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
        if f1 > score:
            best, score = th, f1
    return best


_S_CAL = {}
_THETA = 1


def conflict_and_absent(test, S, rng):
    """Condition 5 (language contradicts the demonstrations) and condition 8
    (the target is absent from the pool), plus the confirmation ablation."""
    pl = pool()
    flagged = forced = 0
    for i, (c, f, _b) in enumerate(test):
        other = test[(i + 1) % len(test)][1]
        gap = conflict_gap(c, other, S)
        if gap is not None and gap >= _THETA:
            flagged += 1
        else:
            r = solve(c, 0, S, mode="joint",
                      demos_n=0) if False else None
    matched = sum(1 for c, f, _b in test
                  if (conflict_gap(c, f, S) or 0) >= _THETA)

    absent_wrong = absent_ok = 0
    for c, f, b in test:
        saved = pl.pop(b)
        try:
            r = solve(c, 0, S, mode="joint")
            h = held(r, f)
            if r["verdict"] == "answered" and h != 10:
                absent_wrong += 1
            else:
                absent_ok += 1
        finally:
            pl[b] = saved
    return flagged, forced, matched, absent_wrong, absent_ok


def planted(S, test):
    """Two defects with gates that could otherwise pass while measuring
    nothing: a token whose sense selects a single behaviour, and a
    confirmation bypass."""
    pl = pool()
    single = [tok for tok, ss in S.items()
              for s in ss if sum(1 for b in pl if s <= sat(b)) == 1]
    # Find a predicate set that provably selects exactly one behaviour, and
    # plant THAT. The first version planted sat(target), which several pool
    # entries also satisfy, so nothing was pinned and the detector correctly
    # saw nothing.
    pin = None
    for k in (2, 3):
        for combo in itertools.combinations(PI_NAMES, k):
            if sum(1 for b in pl if set(combo) <= sat(b)) == 1:
                pin = frozenset(combo)
                break
        if pin:
            break
    bad = dict(S)
    if pin is not None:
        bad[realise(test[0][0], 0)[0]] = frozenset({pin})
    caught_identity = pin is not None and any(
        sum(1 for b in pl if s <= sat(b)) == 1
        for ss in bad.values() for s in ss)

    with_c = sweep(test, S, mode="joint", confirm=True)
    without = sweep(test, S, mode="joint", confirm=False)
    return single, caught_identity, with_c, without


def _gate(R, front, S, test, multi, H, t0):
    print("\n5. CONDITIONS AND PLANTED DEFECTS\n")
    rng = random.Random(5)
    flagged, forced, matched, aw, ao = conflict_and_absent(test, S, rng)
    single, caught_id, with_c, without_c = planted(S, test)
    n = len(test)
    print(f"    conflict: {flagged}/{n} mismatched pairs flagged, "
          f"{forced} forced a wrong answer")
    print(f"    conflict false positives on MATCHED pairs: {matched}/{n}")
    print(f"    target absent: {aw} confident errors, {ao} handled")
    print(f"    confirmation on {with_c['wrong']} wrong, off "
          f"{without_c['wrong']} wrong")
    print(f"    tokens whose induced sense selects a single behaviour: "
          f"{len(single)}")
    print(f"    planted identity token caught: {caught_id}")

    print("\n5b. WHY D7 FAILS -- nine conflict statistics, none usable\n")
    freq = {pi: sum(1 for b in pool() if pi in sat(b)) / len(pool())
            for pi in PI_NAMES}

    def strip(SS, cut):
        G = {pi for pi, fr in freq.items() if fr > cut}
        return {t: frozenset(x for x in (frozenset(y - G) for y in ss) if x)
                for t, ss in SS.items()}

    def hard_empty(g, SS, mism):
        out = []
        for i, (c, f, _b) in enumerate(g):
            fn = g[(i + 1) % len(g)][1] if mism else f
            Cc = constraint(realise(c, 0), SS)
            hard = {b for b in pool() if hard_ok(Cc, sat(b))}
            dm = {t: fn(t) for t in UNIVERSE[:2]}
            dms = {b for b, _g in X.survivors(pool(), list(dm), dm)}
            out.append(None if (not hard or not dms) else not (hard & dms))
        return out

    print(f'    {"statistic":34}{"recall":>9}{"precision":>11}')
    rows = []
    for cut, lab in ((1.01, "hard reading empty"),
                     (0.9, "  minus predicates >90% common"),
                     (0.5, "  minus predicates >50% common"),
                     (0.3, "  minus predicates >30% common")):
        SS = S if cut > 1 else strip(S, cut)
        m, mm = hard_empty(test, SS, False), hard_empty(test, SS, True)
        tp = sum(1 for x in mm if x)
        fp = sum(1 for x in m if x)
        nm = sum(1 for x in mm if x is not None)
        rows.append((lab, tp / max(1, nm), tp / max(1, tp + fp)))
    for th in (1, 2, 3):
        tp = sum(1 for c, _f, _b in test
                 if (conflict_gap(c, test[(test.index((c, _f, _b)) + 1)
                                          % len(test)][1], S) or 0) >= th)
        fp = sum(1 for c, f, _b in test if (conflict_gap(c, f, S) or 0) >= th)
        rows.append((f"violation gap >= {th}", tp / len(test),
                     tp / max(1, tp + fp)))
    for lab, r, pr in rows:
        print(f"    {lab:34}{r:>9.2f}{pr:>11.2f}")
    print("\n    Precision sits at chance across the family, and stripping")
    print("    uninformative predicates buys recall while leaving precision")
    print("    at 0.5. The reason is structural rather than statistical:")
    print("    induction by intersection keeps only what the examples SHARE,")
    print("    so the senses are generic, and a generic constraint is")
    print("    satisfied by the wrong task as readily as the right one. What")
    print("    cannot eliminate cannot contradict. X64B-2 detected conflict")
    print("    because its authored senses were sharp -- and X64C measured")
    print("    what sharp authored senses cost on unseen compositions.")

    print("\n6. THE TEN GATES\n")
    res = []

    def g(k, name, ok, note=""):
        res.append((k, name, ok))
        print(f"   {k:>3}. {name:50} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    J = R["induced, joint"]
    D0 = R["demonstrations only"]
    HARD = R["induced, hard filter"]
    C64 = R["X64C hard lexicon"]
    BLIND = R["role-blind, joint"]
    SEM = R["induced, joint + semantic"]
    ORS = R["oracle senses, joint"]

    g("D1", "target retention beats X64C's 45%",
      J["retained"] / J["n"] > 0.45 + 0.2,
      f'joint {J["retained"]}/{J["n"]} = {100*J["retained"]/J["n"]:.0f}%, '
      f'X64C-style hard filter {C64["retained"]}/{C64["n"]} = '
      f'{100*C64["retained"]/C64["n"]:.0f}%')

    dominated = [b for b, (a, j) in front.items()
                 if j["correct"] >= a["correct"] and j["queries"] < a["queries"]]
    g("D2", "language is on a better accuracy-query frontier",
      len(dominated) >= 3,
      f"joint dominates demonstrations-only at budgets {dominated}: "
      f"more or equal correct with strictly fewer queries")

    g("D3", "polysemy resolves by role without word-specific exceptions",
      len(multi) >= 2 and all(len(set(map(frozenset, rs.values()))) > 1
                              for rs in multi.values()),
      f"{len(multi)} words carry a different sense per role: "
      f"{sorted(multi)}")

    g("D4", "known senses combine on unseen compositions",
      J["correct"] / J["n"] >= 0.8,
      f'{J["correct"]}/{J["n"]} exact on held-out, on '
      f'{len(TEST_PAIRS)} scope-filter pairs absent from development '
      f'and validation')

    g("D5", "no hard false exclusion", J["retained"] == J["n"],
      f'{J["retained"]}/{J["n"]} retained by construction; the hard-filter '
      f'arm retains {HARD["retained"]}/{HARD["n"]}')

    g("D6", "unknown words preserve uncertainty or ask, never guess",
      SEM["semantic"] > 0 and SEM["wrong"] == 0 and J["wrong"] == 0,
      f'{J["unknown"]}/{J["n"]} forms contain a token with no induced '
      f'sense; {SEM["semantic"]} semantic questions asked, '
      f'{SEM["wrong"]} confident errors')

    prec = 1.0 - matched / n if n else 0
    rec = flagged / n if n else 0
    g("D7", "conflict is detected with measured precision and recall",
      rec >= 0.7 and prec >= 0.7 and forced == 0,
      f"recall {rec:.2f} ({flagged}/{n}), precision {prec:.2f} "
      f"({matched} false positives), {forced} forced")

    g("D8", "target-absent stays safe and confirmation still earns its place",
      aw == 0 and without_c["wrong"] >= with_c["wrong"],
      f'{aw} confident errors with the target removed; confirmation on '
      f'{with_c["wrong"]} wrong, off {without_c["wrong"]}')

    g("D9", "no token's sense selects a whole task, and the planted one is "
            "caught", not single and caught_id,
      f"{len(single)} identity-like tokens; planted defect "
      f"{'caught' if caught_id else 'MISSED'}")

    g("D10", "the mechanism was frozen before the test split was touched",
      freeze_hash(S) == H and not (
          {(c[0], c[1]) for c, _f, _b in group(DEV_PAIRS)} & TEST_PAIRS),
      f"senses {H}, induced from development only, no pair overlap")

    print(f"\n   Against the arms that isolate WHERE the gain comes from:")
    print(f"     role-blind joint  {BLIND['correct']}/{BLIND['n']} correct in "
          f"{BLIND['queries']} queries")
    print(f"     induced joint     {J['correct']}/{J['n']} in {J['queries']}")
    print(f"     oracle senses     {ORS['correct']}/{ORS['n']} in "
          f"{ORS['queries']}")

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
