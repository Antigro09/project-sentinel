"""Pins X64E. The claims worth pinning are the structural ones -- the freeze,
the split discipline, that language cannot delete, and that conflict is
measured as posterior mass rather than set emptiness."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x64e_semantics as E


@pytest.fixture(scope="module")
def parser():
    return E.Parser().fit(E.training_examples(E.forms_in(E.DEV_PAIRS),
                                              n_demos=E.TRAIN_DEMOS),
                          epochs=E.EPOCHS, lr=E.LR, l2=E.L2)


def test_mutating_the_frozen_configuration_changes_the_digest():
    """A freeze that nothing checks is a comment."""
    before = E.freeze_digest()
    saved = E.ROLE_SLOT[E.W.VERB]
    E.ROLE_SLOT[E.W.VERB] = ("filt",)
    try:
        after = E.freeze_digest()
    finally:
        E.ROLE_SLOT[E.W.VERB] = saved
    assert before != after
    assert E.freeze_digest() == before


def test_the_splits_share_no_filter_scope_pair():
    assert not (E.DEV_PAIRS & E.TEST_PAIRS)
    assert not (E.VAL_PAIRS & E.TEST_PAIRS)
    dev_f = {f for f, _s in E.DEV_PAIRS}
    dev_s = {s for _f, s in E.DEV_PAIRS}
    assert {f for f, _s in E.TEST_PAIRS} <= dev_f
    assert {s for _f, s in E.TEST_PAIRS} <= dev_s


def test_exact_form_is_partly_unidentifiable_from_behaviour():
    """28 behaviours have more than one logical form, so exact-form accuracy
    below 1.00 is not by itself evidence the parser is wrong."""
    byb = E.forms_by_behaviour()
    multi = [v for v in byb.values() if len(v) > 1]
    assert multi, "every behaviour now has a unique form; re-read E0.3"
    assert max(len(v) for v in byb.values()) > 2


def test_the_gold_representation_can_separate_conflict():
    """E0's stop condition. If this fails, no amount of learning helps."""
    test = E.forms_in(E.TEST_PAIRS)
    m, mm = [], []
    for i, z in enumerate(test):
        f, other = E.execute(z), E.execute(test[(i + 1) % len(test)])
        p = E.gold_parser(z)
        m.append(E.conflict_score(p, {t: f(t) for t in E.UNIVERSE[:2]}))
        mm.append(E.conflict_score(p, {t: other(t) for t in E.UNIVERSE[:2]}))
    assert E.auroc(mm, m) >= 0.9


def test_posterior_conflict_separates_where_set_statistics_did_not(parser):
    """X64D's twelve set-based statistics all sat at chance. Mass does not."""
    test = E.forms_in(E.TEST_PAIRS)
    m, mm = [], []
    for i, z in enumerate(test):
        f, other = E.execute(z), E.execute(test[(i + 1) % len(test)])
        p = parser.dist(E.instr(z, 0))
        m.append(E.conflict_score(p, {t: f(t) for t in E.UNIVERSE[:2]}))
        mm.append(E.conflict_score(p, {t: other(t) for t in E.UNIVERSE[:2]}))
    lo, _hi = E.bootstrap_auroc(mm, m, n=200)
    assert lo > 0.5, "the bootstrap CI no longer excludes chance"
    assert E.auroc(mm, m) > 0.9


def test_language_never_deletes_the_target(parser):
    """Every logical form keeps non-zero probability, so the behaviour the
    demonstrations support is always still in the pool."""
    E.pool()
    for z in E.forms_in(E.TEST_PAIRS)[:10]:
        for v in (0, 1, 2):
            r = E.solve(z, v, parser, budget=0, commit=None)
            assert r["retained"], f"{z} v{v}: language removed the target"


def test_shuffled_instructions_destroy_the_parser(parser):
    """The calibration arm for `the parser reads the language at all`."""
    dev = E.forms_in(E.DEV_PAIRS)
    sh = E.shuffled_parser(dev)
    test = E.forms_in(E.TEST_PAIRS)
    good = sum(1 for z in test
               if E.denote(max(parser.dist(E.instr(z, 0)).items(),
                               key=lambda kv: kv[1])[0]) == E.denote(z))
    bad = sum(1 for z in test
              if E.denote(max(sh.dist(E.instr(z, 0)).items(),
                              key=lambda kv: kv[1])[0]) == E.denote(z))
    assert good > 0.8 * len(test)
    assert bad < 0.2 * len(test)


def test_a_meaning_outside_the_grammar_leaves_no_consistent_form():
    """Condition 9: out-of-grammar is a different state from ambiguous."""
    demos = {t: t[::-1] for t in E.UNIVERSE[:2]}
    assert E.consistent_forms(demos) == []


def test_the_authored_control_never_sees_the_test_vocabulary():
    """The control was twice built with the test split's surface words. If
    it ever scores above chance on variant 2 again, it is cheating."""
    dev = E.forms_in(E.DEV_PAIRS)
    auth = E.authored_parser(dev)
    test = E.forms_in(E.TEST_PAIRS)
    hits = sum(1 for z in test
               if max(auth.dist(E.instr(z, 2)).items(),
                      key=lambda kv: kv[1])[0] == z)
    assert hits <= 0.1 * len(test), "the authored control sees test words"
