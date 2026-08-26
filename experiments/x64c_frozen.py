"""X64C: the same lexicon, frozen, against tasks it has never seen.

X64B-2 reported 24 of 24 paraphrases landing in the canonical class. That
number is DEVELOPMENT-SET PERFORMANCE and should never have been offered as
generalisation: the lexicon was edited three times in response to failures
on those exact paraphrases -- `within` overconstrained, `first` ambiguous,
`comment` encoding almost a whole task. What 24/24 shows is that the final
lexicon covers the examples used to debug it.

So this experiment freezes the lexicon FIRST, mechanically, and then meets
tasks and instructions it has never been exposed to.

THE FREEZE IS ENFORCED, NOT PROMISED. The lexicon and the predicate set are
hashed below. If either is edited after this line was written, the hash
check fails and the experiment refuses to run. There is no path where a
holdout failure is quietly repaired by touching the lexicon -- a failure
here is a finding, and it gets reported as one.

THREE DISJOINT LEVELS:
  development   X64B-2's eleven tasks and their paraphrases. The lexicon was
                authored against these. Reported for reference only; no
                generalisation claim rests on them.
  compositional NEW task behaviours built from primitives the lexicon knows,
                in combinations never used while authoring it.
  language      NEW instruction forms -- unseen word orders, unseen
                combinations, and words the lexicon does not contain at all.

FIVE CONDITIONS, because four of them hide different failures:
  1 clear and realisable
  2 ambiguous but realisable
  3 the reference program is absent but an empirically adequate candidate
    exists
  4 no adequate candidate exists at any expansion rung
  5 the instruction contradicts the demonstrations

THREE THINGS THAT ARE NOT THE SAME, kept apart because they came apart in
X64B-1 and one of them was briefly reported as another:
  reference recovery   the same behaviour as the hidden target over U
  empirical adequacy   right on every input actually tested
  global correctness   right on the intended domain -- NOT measured here

FIVE PLANTED DEFECTS. Every gate that could pass while measuring nothing is
run against a deliberately broken system that it must catch:
  a word mapped to a task identity
  a semantically ambiguous word overconstrained
  a confirmation bypass
  a target absent from every pool
  a conflicting instruction/demonstration pair

PRE-REGISTERED FALSIFIERS -- if any of these holds, X64B's generalisation
claim is DOWNGRADED in the README rather than defended:
  the query advantage disappears on frozen tasks
  the lexicon excludes intended targets on unseen compositions
  target-absent cases still produce confident singleton answers
  conflict detection works only on the authored examples
  new paraphrases would require lexicon edits
  gains come only from the development families

RESULT: 10 of 12 gates. THE TWO THAT FAIL ARE THE POINT, and both fire
pre-registered falsifiers, so X64B's generalisation claim is downgraded
rather than defended.

C1 -- THE QUERY ADVANTAGE DOES NOT SURVIVE. It reverses.

    arm                       answered  correct  WRONG  queries  held-out
    demos + disagreement            10       10      0       23       100
    language + disagreement          3        3      0       38        30
    language + demos + dis.          3        3      0        5        30
    random clarification             3        3      0       14        30
    no confirmation                  4        3      1        4        35
    reckless                         4        3      1        4        35
    paranoid                         0        0      0        4         0
    oracle                           3        3      0        8        30

On X64B-2's development tasks language cut questions from 14 to 8. Here
demonstrations alone answer 10 of 12 and language cuts that to 3. Language
is not merely unhelpful on unseen compositions; it is actively harmful.

C6 -- THE GAINS ARE CONFINED TO THE DEVELOPMENT FAMILIES. 3 of 12
compositional-holdout tasks solved exactly, against 10 of 11 on the tasks
the lexicon was authored against.

THE CAUSE IS MEASURED, not guessed. 22 of 40 holdout instruction forms
EXCLUDE THEIR OWN TARGET, against 0 of 30 on the development set. `brackets`
carries "only inside brackets" -- authored for "copy what is inside the
brackets" and simply wrong for "remove the brackets". That is the same
polysemy that `first` and `comment` had, and the same fix would work, and
the fix is exactly what the freeze forbids. Fitting the lexicon to the
evaluation suite is what X64B-2 did, and this is what it cost.

WHAT DOES SURVIVE, and it is not nothing:

  the failure mode is SAFE.   22 false exclusions, 0 confident errors. A
     lexicon that excludes the truth makes the system say CONFLICT, not
     guess. C2, C3, C7 pass.
  conflict detection transfers. 10 of 12 unseen mismatched pairs flagged,
     0 forced. It was not memorising the eight authored examples.
  confirmation still earns its place. With it, 0 confident errors; without
     it, 2.
  unseen language FORM is not the problem. Where the canonical instruction
     answered at all, 6 of 6 unseen forms -- new word orders, 44
     out-of-vocabulary words -- landed in the same behavioural class. What
     fails is unseen COMPOSITION, not unseen phrasing. Note the small n: an
     earlier draft of this measurement scored two failures as an agreement
     when both returned nothing, and reported 17 of 24.

ALL FIVE PLANTED DEFECTS ARE CAUGHT. The first injection was too weak and
the detector missed it -- an exhaustive search over predicate subsets found
three that pin a single behaviour, so the defect was injectable and the miss
belonged to the injection. Two other gates had broken premises: condition 4
treated a de-seeded target as hopeless when blind enumeration still reaches
it, and the paraphrase score counted None == None as agreement.

Run: uv run python experiments/x64c_frozen.py
"""

import hashlib
import json
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x63_sparse_price as P
import x63b_cegis_store as B
import x64a_identify as X
import x64b1_openworld as O
import x64b2_language as L

UNIVERSE, HELD_OUT, CHALLENGE = X.UNIVERSE, X.HELD_OUT, O.CHALLENGE
EVIDENCE0, FEW = X.EVIDENCE0, L.FEW

# ------------------------------------------------------------- THE FREEZE
LEXICON_SHA = "e295cb6c1e9c5ee6e8290f598ef9ef80"
PREDS_SHA = "f89db1fa0dc5ecad49139be394107972"


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()
                          ).hexdigest()[:32]


def check_freeze():
    lex = _sha({k: sorted(v) for k, v in sorted(L.LEXICON.items())})
    pre = _sha(sorted(L.PREDS))
    return lex == LEXICON_SHA and pre == PREDS_SHA, lex, pre


# ------------------------------------------------ compositional holdout
#
# Behaviours built from primitives the frozen lexicon knows -- brackets,
# the hash, adjacency, uniqueness, having-been-seen, position -- in
# combinations it was never authored against. Written in one pass, before
# any of them was run.

def strip_brackets(s):
    return "".join(c for c in s if c not in "()")


def only_brackets(s):
    return "".join(c for c in s if c in "()")


def strip_hash_chars(s):
    return "".join(c for c in s if c != "#")


def letters_only(s):
    return "".join(c for c in s if c not in "()#")


def drop_first(s):
    return s[1:]


def unique_before_hash(s):
    seen, out = set(), []
    for c in s.split("#")[0]:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "".join(out)


def dedupe_before_hash(s):
    out = []
    for c in s.split("#")[0]:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def matching_first_before_hash(s):
    h = s.split("#")[0]
    return "".join(c for c in h[1:] if h and c == h[0])


def unique_no_brackets(s):
    seen, out = set(), []
    for c in s:
        if c in "()":
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "".join(out)


def dedupe_no_brackets(s):
    out = []
    for c in s:
        if c in "()":
            continue
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def seen_before_no_hash(s):
    seen, out = set(), []
    for c in s:
        if c == "#":
            continue
        if c in seen:
            out.append(c)
        seen.add(c)
    return "".join(out)


def two_behind_no_brackets(s):
    t = strip_brackets(s)
    return t[:-2] if len(t) > 2 else ""


NEW_TASKS = {
    "strip brackets": strip_brackets,
    "only brackets": only_brackets,
    "strip hashes": strip_hash_chars,
    "letters only": letters_only,
    "drop first": drop_first,
    "unique before hash": unique_before_hash,
    "dedupe before hash": dedupe_before_hash,
    "matching first before hash": matching_first_before_hash,
    "unique no brackets": unique_no_brackets,
    "dedupe no brackets": dedupe_no_brackets,
    "seen before no hash": seen_before_no_hash,
    "two behind no brackets": two_behind_no_brackets,
}

NEW_FAMILY = {
    "strip brackets": "streaming", "only brackets": "streaming",
    "strip hashes": "streaming", "letters only": "streaming",
    "drop first": "sequence", "unique before hash": "set",
    "dedupe before hash": "register",
    "matching first before hash": "register",
    "unique no brackets": "set", "dedupe no brackets": "register",
    "seen before no hash": "set", "two behind no brackets": "sequence",
}

# Witnesses for the ones the machine can express, so that "the reference is
# absent" is a controlled condition rather than an accident of enumeration.
E, AD = "EMIT", "ADV"
EA = B.seq(E, AD)
OPEN, CLOSE, HASH = ("AT", 0, "("), ("AT", 0, ")"), ("AT", 0, "#")
HAS, M0 = ("HAS",), ("MATCH", 0)

NEW_WITNESS = {
    "strip brackets": ("LOOP", ("IF", OPEN, AD, ("IF", CLOSE, AD, EA))),
    "only brackets": ("LOOP", ("IF", OPEN, EA, ("IF", CLOSE, EA, AD))),
    "strip hashes": ("LOOP", ("IF", HASH, AD, EA)),
    "letters only": ("LOOP", ("IF", OPEN, AD, ("IF", CLOSE, AD,
                                               ("IF", HASH, AD, EA)))),
    "drop first": ("SEQ", AD, ("LOOP", EA)),
    "unique before hash": ("LOOP", ("IF", HASH, "HALT",
                                    B.seq("LOAD", ("IF", HAS, AD,
                                                   B.seq(E, "PUT", AD))))),
    "dedupe before hash": ("LOOP", ("IF", HASH, "HALT",
                                    ("IF", M0, AD,
                                     B.seq(E, "LOAD", AD)))),
    "unique no brackets": ("LOOP", ("IF", OPEN, AD,
                                    ("IF", CLOSE, AD,
                                     B.seq("LOAD", ("IF", HAS, AD,
                                                    B.seq(E, "PUT", AD)))))),
    "dedupe no brackets": ("LOOP", ("IF", OPEN, AD,
                                    ("IF", CLOSE, AD,
                                     ("IF", M0, AD,
                                      B.seq(E, "LOAD", AD))))),
    "seen before no hash": ("LOOP", ("IF", HASH, AD,
                                     B.seq("LOAD",
                                           ("IF", HAS, B.seq(E, "PUT", AD),
                                            B.seq("PUT", AD))))),
    # No witness written for these two: the first needs the head to look
    # backwards past a hash, the second needs a two-step delay over a
    # filtered stream. They are the condition-4 cases by construction.
    "matching first before hash": None,
    "two behind no brackets": None,
}


# ---------------------------------------------------------- language holdout
#
# Instructions written from the FROZEN vocabulary. `canon` is a plain
# request; `seen_form` reuses the phrasing style the lexicon was authored
# against; `unseen_form` uses word orders, combinations and out-of-vocabulary
# words it has never met. Everything below was written in one pass, before
# any of it was run, and committed before the first result existed.

NEW_INSTRUCTIONS = {
    "strip brackets": (
        "remove the brackets",
        ["delete the parentheses"],
        ["the brackets, discard them", "please drop every parenthesis"]),
    "only brackets": (
        "keep the brackets",
        ["take the parentheses"],
        ["only the brackets, keep them", "retain solely the parentheses"]),
    "strip hashes": (
        "remove the hash",
        ["delete the hash"],
        ["the hash, drop it", "kindly strip out the hash symbols"]),
    "letters only": (
        "remove the brackets and the hash",
        ["delete the parentheses and the hash"],
        ["the brackets and the hash, discard them both",
         "eliminate punctuation, namely parentheses and hash"]),
    "drop first": (
        "drop the first",
        ["remove the beginning"],
        ["the beginning, drop it", "omit the initial character"]),
    "unique before hash": (
        "keep each symbol once until the comment",
        ["take every character once before the comment"],
        ["until the comment, keep each symbol once",
         "retain each distinct glyph a single time prior to the comment"]),
    "dedupe before hash": (
        "remove repeats in a row until the comment",
        ["delete consecutive duplicates before the comment"],
        ["until the comment, remove adjacent repeats",
         "collapse runs of identical glyphs ahead of the comment"]),
    "matching first before hash": (
        "keep the symbols matching the first until the comment",
        ["take what is same as the first before the comment"],
        ["until the comment, keep the symbols matching the first",
         "retain glyphs identical to the initial one, pre-comment"]),
    "unique no brackets": (
        "keep each symbol once and remove the brackets",
        ["take every character once and delete the parentheses"],
        ["and remove the brackets, keep each symbol once",
         "deduplicate globally while excising parentheses"]),
    "dedupe no brackets": (
        "remove repeats in a row and the brackets",
        ["delete adjacent duplicates and the parentheses"],
        ["and the brackets, remove repeats in a row",
         "collapse runs and excise parentheses"]),
    "seen before no hash": (
        "keep the symbols seen before and remove the hash",
        ["take what was seen again and delete the hash"],
        ["and remove the hash, keep the symbols seen before",
         "retain glyphs encountered previously, minus the hash"]),
    "two behind no brackets": (
        "copy two behind and remove the brackets",
        ["echo two behind and delete the parentheses"],
        ["and remove the brackets, copy two behind",
         "reproduce with a lag of two, parentheses excised"]),
}

# Deliberately vague instructions for condition 2 -- each is satisfied by
# more than one of the tasks above.
NEW_AMBIGUOUS = {
    "remove the punctuation": ["strip brackets", "strip hashes",
                               "letters only"],
    "keep each symbol once": ["unique before hash", "unique no brackets"],
    "remove repeats": ["dedupe before hash", "dedupe no brackets"],
}


ALL_TASKS = dict(L.TASKS)
ALL_TASKS.update(NEW_TASKS)
ALL_FAMILY = dict(L.FAMILY)
ALL_FAMILY.update(NEW_FAMILY)
ALL_WITNESS = dict(B.WITNESS)
ALL_WITNESS.update(NEW_WITNESS)


def build(level, exclude=(), gen=0, seed=1000, defect_none=()):
    """O.build, widened to seed the holdout witnesses too. `defect_none`
    removes a target from every pool -- one of the planted defects."""
    pool = dict(O.core(level, seed, None, gen))
    for n, w in ALL_WITNESS.items():
        if w is not None and n not in exclude and n not in defect_none:
            O._insert(pool, w)
    return pool


# ------------------------------------------------------------- the protocol

def solve(instruction, f, rng, exclude=(), use_lang=True, use_demos=True,
          query="disagreement", confirmations=True, max_rung=4, demos=None,
          defect_none=(), report="honest"):
    """X64B-2's pipeline over the widened pool, with hooks for the planted
    defects: `confirmations=False` is the confirmation bypass, `defect_none`
    removes a target from every rung, and `report` forces a reckless or
    paranoid verdict."""
    preds, _unknown = L.meaning(instruction) if use_lang else (set(), [])
    demos = demos if demos is not None else {t: f(t) for t in FEW}
    extra, asked_total = {}, 0
    for level in range(max_rung + 1):
        pool = build(level, exclude=exclude, gen=1 if level == 4 else 0,
                     defect_none=defect_none)
        pool = O.keep_consistent(pool, extra)
        lang = L.narrow(pool, preds) if use_lang else pool
        ev = dict(demos) if use_demos else {}
        both = X.survivors(lang, list(ev), ev) if ev else list(lang.items())
        if use_lang and use_demos and ev and not both:
            if lang and X.survivors(pool, list(ev), ev):
                return dict(verdict="conflict", rep=None, asked=asked_total,
                            rung=O.RUNGS[level])
        surv, asked, used = both, set(ev), 0
        while len(surv) > 1 and used < X.BUDGET and query != "none":
            if query == "disagreement":
                q = X.best_query(surv, asked)
            elif query == "oracle":
                best, keep = None, len(surv) + 1
                for t in UNIVERSE:
                    if t in asked:
                        continue
                    k = len(X.split(surv, t).get(f(t), []))
                    if k < keep:
                        best, keep = t, k
                q = best
            else:
                cands = [t for t in UNIVERSE if t not in asked
                         and len(X.split(surv, t)) > 1]
                q = rng.choice(cands) if cands else None
            if q is None:
                break
            asked.add(q)
            used += 1
            asked_total += 1
            surv = X.refute(surv, q, f(q))
        st = X.state_of(surv, asked)
        if report == "paranoid":
            return dict(verdict="unresolved_within_budget", rep=None,
                        asked=asked_total, rung=O.RUNGS[level])
        if st != "identified_on_U":
            if st == "inconsistent":
                continue
            if report == "reckless" and surv:
                rep = min(surv, key=lambda bp: X._size(bp[1]))[1]
                return dict(verdict="answered", rep=rep, asked=asked_total,
                            rung=O.RUNGS[level], scope="unconfirmed")
            return dict(verdict=st, rep=None, asked=asked_total,
                        rung=O.RUNGS[level])
        rep = surv[0][1]
        if not confirmations:
            return dict(verdict="answered", rep=rep, asked=asked_total,
                        rung=O.RUNGS[level], scope="unconfirmed")
        ok, ce = O.confirm(rep, f, rng)
        if ok:
            return dict(verdict="answered", rep=rep, asked=asked_total,
                        rung=O.RUNGS[level], scope="confirmed_on_challenge")
        extra[ce] = f(ce)
    return dict(verdict="none_of_the_above", rep=None, asked=asked_total,
                rung=None)


def held(r, f):
    if r.get("rep") is None:
        return None
    return sum(1 for t in HELD_OUT if P.semit(r["rep"], t) == f(t))


DEV = list(L.TASKS)
NEW = list(NEW_TASKS)


def instr_of(n):
    if n in NEW_INSTRUCTIONS:
        return NEW_INSTRUCTIONS[n]
    canon, paras = L.INTENTS[n]
    return canon, paras[:1], paras[1:2] or paras[:1]


def main() -> int:
    t0 = time.perf_counter()
    ok, lex, pre = check_freeze()
    print("X64C: the same lexicon, frozen, against tasks it has never seen\n")
    print(f"0. THE FREEZE   lexicon {lex}  ({'INTACT' if ok else 'BROKEN'})")
    if not ok:
        print("   The lexicon or predicate set changed after the hash was")
        print("   pinned. That invalidates every generalisation claim below,")
        print("   so the experiment stops rather than reporting them.")
        return 2
    print(f"   {len(L.LEXICON)} words, {len(L.PREDS)} predicates, unchanged "
          f"since the holdout was written and committed.\n")

    pool = build(3)
    print(f"1. SPLITS   development {len(DEV)} tasks, compositional holdout "
          f"{len(NEW)}, pool {len(pool):,} behaviours")

    # ---- M1 candidate recall, M2 false lexical exclusion
    def recall_and_exclusion(names):
        present, excluded, forms = 0, [], 0
        for n in names:
            f = ALL_TASKS[n]
            tb = tuple(f(t) for t in UNIVERSE)
            if tb not in pool:
                continue
            present += 1
            canon, seen, unseen = instr_of(n)
            for i in [canon] + list(seen) + list(unseen):
                forms += 1
                if tb not in L.narrow(pool, L.meaning(i)[0]):
                    excluded.append((n, i))
        return present, excluded, forms

    dp, dex, dfo = recall_and_exclusion(DEV)
    np_, nex, nfo = recall_and_exclusion(NEW)
    print(f"\n   M1 candidate recall before clarification: "
          f"development {dp}/{len(DEV)}, compositional {np_}/{len(NEW)}")
    print(f"   M2 FALSE LEXICAL EXCLUSION -- target in the pool, excluded by")
    print(f"      its own instruction:")
    print(f"        development   {len(dex):>3}/{dfo} instruction forms")
    print(f"        compositional {len(nex):>3}/{nfo} instruction forms")
    worst = sorted({n for n, _i in nex})
    print(f"      affected holdout tasks: {worst}")
    print("      `brackets` carries \"only inside brackets\", authored for")
    print("      \"copy what is inside the brackets\" and wrong for \"remove")
    print("      the brackets\". The same polysemy as `first`. The lexicon is")
    print("      frozen, so this is reported, not repaired.")

    # ---- conditions
    print("\n2. FIVE CONDITIONS on the compositional holdout\n")
    adequate = {}
    for n in NEW:
        f = ALL_TASKS[n]
        tb = tuple(f(t) for t in UNIVERSE)
        if tb in pool:
            adequate[n] = True
            continue
        found = False
        for lvl in range(len(O.RUNGS)):
            pl = build(lvl, exclude=(n,), gen=1 if lvl == 4 else 0)
            if any(all(P.semit(pr, t) == f(t) for t in HELD_OUT)
                   for pr in pl.values()):
                found = True
                break
        adequate[n] = found

    hdr = (f'{"task":28}{"1 clear":>16}{"2 vague":>14}{"3 absent":>16}'
           f'{"4 hopeless":>15}{"5 conflict":>14}')
    print("   " + hdr + "\n   " + "-" * len(hdr))

    def cell(r, f):
        h = held(r, f)
        if r["verdict"] == "answered":
            return (f"{h}/10" if h == 10 else f"{h}/10 WRONG") + f" q{r['asked']}"
        return {"none_of_the_above": "none-above",
                "unresolved_within_budget": "unresolved",
                "underspecified_on_U": "underspec",
                "inconsistent": "inconsistent",
                "conflict": "CONFLICT"}[r["verdict"]] + f" q{r['asked']}"

    C = {}
    for i, n in enumerate(NEW):
        f = ALL_TASKS[n]
        canon, _seen, _unseen = instr_of(n)
        vague = next((a for a, w in NEW_AMBIGUOUS.items() if n in w), canon)
        other = ALL_TASKS[NEW[(i + 1) % len(NEW)]]
        row = {
            "clear": solve(canon, f, random.Random(5)),
            "vague": solve(vague, f, random.Random(5),
                           demos={t: f(t) for t in FEW[:1]}),
            "absent": solve(canon, f, random.Random(5), exclude=(n,)),
            "hopeless": solve(canon, f, random.Random(5), defect_none=(n,),
                              exclude=(n,)),
            "conflict": solve(canon, f, random.Random(5),
                              demos={t: other(t) for t in FEW}),
        }
        C[n] = row
        print(f"   {n:28}" + f'{cell(row["clear"], f):>16}'
              + f'{cell(row["vague"], f):>14}{cell(row["absent"], f):>16}'
              + f'{cell(row["hopeless"], f):>15}'
              + f'{cell(row["conflict"], f):>14}')

    # ---- arms
    print("\n3. EIGHT ARMS on the compositional holdout, clear condition\n")
    ARMS = [("demos + disagreement", dict(use_lang=False)),
            ("language + disagreement", dict(use_demos=False)),
            ("language + demos + dis.", dict()),
            ("random clarification", dict(query="random")),
            ("no confirmation", dict(confirmations=False)),
            ("reckless", dict(report="reckless", confirmations=False)),
            ("paranoid", dict(report="paranoid")),
            ("oracle", dict(query="oracle"))]
    print(f'    {"arm":26}{"answered":>10}{"correct":>9}{"WRONG":>7}'
          f'{"queries":>9}{"held-out":>10}')
    stats = {}
    for label, kw in ARMS:
        ans = cor = wr = q = h = 0
        for n in NEW:
            f = ALL_TASKS[n]
            canon, _s, _u = instr_of(n)
            r = solve(canon, f, random.Random(5), **kw)
            q += r["asked"]
            if r["verdict"] == "answered":
                ans += 1
                hh = held(r, f) or 0
                h += hh
                cor += hh == 10
                wr += hh != 10
        stats[label] = (ans, cor, wr, q, h)
        print(f"    {label:26}{ans:>10}{cor:>9}{wr:>7}{q:>9}{h:>10}")

    # ---- language forms
    print("\n4. SEEN VERSUS UNSEEN LANGUAGE FORM\n")
    form = {"canonical": [0, 0], "seen style": [0, 0], "unseen form": [0, 0]}
    drift = []
    for n in NEW:
        f = ALL_TASKS[n]
        canon, seen, unseen = instr_of(n)
        base = solve(canon, f, random.Random(5))
        bb = X.behaviour(base["rep"]) if base.get("rep") is not None else None
        if bb is None:
            # The canonical instruction produced no answer, so "lands in the
            # canonical class" is not a question that can be asked. Counting
            # a paraphrase that also failed as agreement would score two
            # failures as a success, which is how the first draft of this
            # measurement reached 17/24.
            continue
        for label, group in (("canonical", [canon]), ("seen style", seen),
                             ("unseen form", unseen)):
            for i in group:
                r = solve(i, f, random.Random(5))
                rb = (X.behaviour(r["rep"]) if r.get("rep") is not None
                      else None)
                form[label][1] += 1
                if rb == bb:
                    form[label][0] += 1
                elif label != "canonical":
                    drift.append((n, i))
    print("    Scored only where the canonical instruction itself produced")
    print("    an answer, so two failures cannot count as an agreement.\n")
    print(f'    {"form":16}{"lands in the canonical class":>32}')
    for k, (a, b) in form.items():
        print(f"    {k:16}{f'{a}/{b}':>32}")
    oov = sorted({w for n in NEW for i in instr_of(n)[2]
                  for w in i.lower().split() if w not in L.LEXICON})
    print(f"\n    out-of-vocabulary words in the unseen forms: {len(oov)}")
    print(f"    {oov}")

    return _gate(C, stats, form, dex, nex, dfo, nfo, adequate, pool, t0)


def _defects(pool):
    """Five planted defects. Every gate that could pass while measuring
    nothing is run against a system deliberately broken in the way that gate
    exists to catch. A gate that misses its own defect is reported VACUOUS,
    which is worse than reported failing."""
    caught = {}

    # d1: a word mapped to a task identity. The first version of this
    # injection assigned a four-predicate conjunction and the detector
    # MISSED it -- because that conjunction did not actually pin anything.
    # An exhaustive search over predicate subsets found three that select a
    # single behaviour, so the defect is injectable and the miss was the
    # injection's fault, not the detector's.
    saved = dict(L.LEXICON)
    L.LEXICON["comment"] = {"a prefix", "only inside brackets"}
    single = [w for w, ps in L.LEXICON.items()
              if ps and len(L.narrow(pool, ps)) == 1]
    L.LEXICON.clear()
    L.LEXICON.update(saved)
    caught["word encodes a task identity"] = bool(single)

    # d2: a semantically ambiguous word overconstrained.
    f = ALL_TASKS["strip hashes"]
    base = solve(NEW_INSTRUCTIONS["strip hashes"][0], f, random.Random(5))
    L.LEXICON["hash"] = {"only inside brackets"}
    broke = solve(NEW_INSTRUCTIONS["strip hashes"][1][0], f, random.Random(5))
    L.LEXICON.clear()
    L.LEXICON.update(saved)
    bb = X.behaviour(base["rep"]) if base.get("rep") is not None else None
    rb = X.behaviour(broke["rep"]) if broke.get("rep") is not None else None
    caught["ambiguous word overconstrained"] = bb != rb

    # d3: a confirmation bypass.
    wrong_with, wrong_without = 0, 0
    for n in NEW:
        g = ALL_TASKS[n]
        canon, _s, _u = instr_of(n)
        a = solve(canon, g, random.Random(5), exclude=(n,),
                  confirmations=False)
        b = solve(canon, g, random.Random(5), exclude=(n,))
        if a["verdict"] == "answered" and (held(a, g) or 0) != 10:
            wrong_without += 1
        if b["verdict"] == "answered" and (held(b, g) or 0) != 10:
            wrong_with += 1
    caught["confirmation bypass"] = wrong_without > wrong_with

    # d4: a target absent from every pool.
    n = "strip hashes"
    g = ALL_TASKS[n]
    r = solve(NEW_INSTRUCTIONS[n][0], g, random.Random(5), exclude=(n,),
              defect_none=(n,))
    caught["target absent from every pool"] = not (
        r["verdict"] == "answered" and (held(r, g) or 0) != 10)

    # d5: a conflicting instruction and demonstration pair.
    g, other = ALL_TASKS["strip hashes"], ALL_TASKS["only brackets"]
    r = solve(NEW_INSTRUCTIONS["strip hashes"][0], g, random.Random(5),
              demos={t: other(t) for t in FEW})
    caught["conflicting instruction and demos"] = r["verdict"] != "answered"
    return caught, wrong_with, wrong_without


def _gate(C, stats, form, dex, nex, dfo, nfo, adequate, pool, t0):
    print("\n5. FIVE PLANTED DEFECTS -- each gate against the break it exists")
    print("   to catch\n")
    caught, wrong_with, wrong_without = _defects(pool)
    for k, v in caught.items():
        print(f"    {k:36} {'CAUGHT' if v else 'MISSED':>8}")
    print(f"\n    confirmation on: {wrong_with} confident errors; "
          f"off: {wrong_without}")

    print("\n6. THE GATES\n")
    res = []

    def g(n, name, okv, note=""):
        res.append((n, name, okv))
        print(f"   {n:>3}. {name:50} {('PASS' if okv else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    A = stats["language + demos + dis."]
    D = stats["demos + disagreement"]
    R = stats["random clarification"]
    top = max(v[0] for k, v in stats.items() if k not in ("oracle",))
    peers = {k: v for k, v in stats.items()
             if v[0] == top and k not in ("language + demos + dis.", "oracle")}
    g("C1", "the query advantage survives on frozen tasks",
      A[0] == top and all(A[3] < v[3] for v in peers.values()) if peers
      else A[0] == top,
      f"language+demos {A[0]} answered in {A[3]} queries; "
      f"demos-only {D[0]}/{D[3]}, random {R[0]}/{R[3]}")

    g("C2", "false lexical exclusion never yields a confident error",
      all(not (C[n]["clear"]["verdict"] == "answered"
               and (held(C[n]["clear"], ALL_TASKS[n]) or 0) != 10)
          for n in NEW),
      f"{len(nex)}/{nfo} holdout forms exclude their own target; "
      f"0 became confident errors")

    wrong_abs = [n for n in NEW if C[n]["absent"]["verdict"] == "answered"
                 and (held(C[n]["absent"], ALL_TASKS[n]) or 0) != 10]
    g("C3", "target-absent cases produce no confident singleton answer",
      not wrong_abs, f"{len(wrong_abs)} confident errors: {wrong_abs or 'none'}")

    forced = [n for n in NEW if C[n]["conflict"]["verdict"] == "answered"
              and (held(C[n]["conflict"], ALL_TASKS[n]) or 0) != 10]
    flagged = [n for n in NEW if C[n]["conflict"]["verdict"] == "conflict"]
    fp = [n for n in NEW if C[n]["clear"]["verdict"] == "conflict"
          and adequate.get(n)]
    g("C4", "conflict detection works on unseen pairs, not just authored ones",
      not forced and len(flagged) > 0,
      f"{len(flagged)}/{len(NEW)} flagged on unseen mismatched pairs, "
      f"{len(forced)} forced; {len(fp)} matched pairs also flagged")

    sc, st_ = form["unseen form"]
    g("C5", "unseen language forms need no lexicon edit",
      sc == st_, f"{sc}/{st_} unseen forms land in the canonical class")

    devq = sum(1 for n in NEW if C[n]["clear"]["verdict"] == "answered"
               and held(C[n]["clear"], ALL_TASKS[n]) == 10)
    g("C6", "gains are not confined to the development families",
      devq >= 6, f"{devq}/{len(NEW)} compositional-holdout tasks solved "
                 f"exactly")

    # Removing the witness does not make a task hopeless: blind enumeration
    # may still reach it, and for `strip hashes` and `drop first` it does.
    # The first draft of this gate treated de-seeded as hopeless and failed
    # on two tasks that were correctly answered. Condition 4 is the subset
    # with no empirically adequate candidate at ANY rung.
    truly = [n for n in NEW if not adequate.get(n)]
    answered = [n for n in truly if C[n]["hopeless"]["verdict"] == "answered"]
    g("C7", "with no adequate candidate anywhere, nothing is answered",
      not answered and len(truly) > 0,
      f"{len(truly)} genuinely hopeless {truly}, {len(answered)} answered; "
      f"de-seeded but still reachable: "
      f"{[n for n in NEW if adequate.get(n) and C[n]['hopeless']['verdict'] == 'answered']}")

    for i, (k, v) in enumerate(caught.items()):
        g(f"C{8 + i}", f"planted defect caught: {k}", v)

    okn = [n for n, _m, p in res if p]
    print(f"\n   VERDICT: {len(okn)}/{len(res)} gates pass")
    bad = [(n, m) for n, m, p in res if not p]
    if bad:
        print("\n   FAILING:")
        for n, m in bad:
            print(f"     {n}. {m}")
        print("\n   The lexicon is frozen. These are findings, not a to-do")
        print("   list, and the README claim is downgraded to match.")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
