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
