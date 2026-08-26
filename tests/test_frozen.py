"""Pins X64C. The most important test here is the freeze itself: if the
lexicon can be edited without anything complaining, every generalisation
number in this project's language work is unfalsifiable."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x63_sparse_price as P
import x64a_identify as X
import x64b2_language as L
import x64c_frozen as C


def test_the_freeze_is_intact():
    ok, lex, pre = C.check_freeze()
    assert ok, f"lexicon/predicates changed after the freeze: {lex} {pre}"


def test_editing_the_lexicon_breaks_the_freeze():
    """The freeze is only a guarantee if violating it is detected. Without
    this, `frozen` is a comment rather than a mechanism."""
    saved = dict(L.LEXICON)
    L.LEXICON["brackets"] = {"subsequence"}
    try:
        ok, _l, _p = C.check_freeze()
    finally:
        L.LEXICON.clear()
        L.LEXICON.update(saved)
    assert not ok, "the lexicon can be edited without the freeze noticing"
    assert C.check_freeze()[0], "restoring the lexicon did not restore it"


def test_the_holdout_witnesses_are_correct():
    """A wrong witness would make a task look unreachable and quietly turn a
    language failure into a machine failure."""
    for n, w in C.NEW_WITNESS.items():
        if w is None:
            continue
        f = C.NEW_TASKS[n]
        bad = [t for t in C.UNIVERSE + C.HELD_OUT + C.CHALLENGE
               if P.semit(w, t) != f(t)]
        assert not bad, f"{n} witness wrong on {bad[:2]}"


def test_the_frozen_lexicon_excludes_targets_on_unseen_compositions():
    """The finding, pinned. If this ever stops failing, the lexicon changed
    or the holdout got easier, and either way X64C's conclusion needs
    re-reading rather than quoting."""
    pool = C.build(3)
    excluded = []
    for n in C.NEW_TASKS:
        f = C.ALL_TASKS[n]
        tb = tuple(f(t) for t in C.UNIVERSE)
        if tb not in pool:
            continue
        canon = C.NEW_INSTRUCTIONS[n][0]
        if tb not in L.narrow(pool, L.meaning(canon)[0]):
            excluded.append(n)
    assert excluded, "the frozen lexicon no longer excludes any holdout target"


def test_a_false_exclusion_never_becomes_a_confident_error():
    """The failure mode is safe, and that is what makes the audit reportable
    rather than alarming: a lexicon that excludes the truth makes the system
    say CONFLICT, not guess."""
    for n in C.NEW_TASKS:
        f = C.ALL_TASKS[n]
        r = C.solve(C.NEW_INSTRUCTIONS[n][0], f, random.Random(5))
        if r["verdict"] == "answered":
            assert C.held(r, f) == 10, f"{n} answered confidently and wrong"


def test_some_predicate_subset_can_pin_a_single_behaviour():
    """Calibration for the planted defect. The first injection assigned a
    conjunction that pinned nothing, the detector correctly saw nothing, and
    the gate was recorded as MISSED. The defect has to be injectable."""
    import itertools
    pool = C.build(3)
    names = sorted(L.PREDS)
    found = [combo for k in (1, 2)
             for combo in itertools.combinations(names, k)
             if len(L.narrow(pool, set(combo))) == 1]
    assert found, "no predicate subset pins a behaviour; the defect is not " \
                  "injectable and its gate would be vacuous"
