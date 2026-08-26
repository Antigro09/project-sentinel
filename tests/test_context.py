"""Pins X64F. The property the whole experiment rests on is that the surface
language contains constructions a bag of words provably cannot decode, and
that the contextual parser decodes them."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x64f_context as F


@pytest.fixture(scope="module")
def trained():
    dev_p, _v, test_p = F.seeded_splits(101)
    dev, test = F.forms_in(dev_p), F.forms_in(test_p)
    ex = F.training_examples(dev, n_demos=F.TRAIN_DEMOS, samples=2)
    return dict(dev=dev, test=test,
                ctx=F.Parser("context").fit(ex, epochs=60, lr=F.LR, l2=F.L2),
                bow=F.Parser("bow").fit(ex, epochs=60, lr=F.LR, l2=F.L2))


def test_the_realiser_is_not_one_to_one():
    """X64E's realiser let an authored word-to-slot table reach 1.00
    exact-form, which is why X64 was not closed. If this ever stops failing,
    the generator has regressed to a serialisation of the logical form."""
    col = F.collisions(F.LIVE)
    assert len(col) >= 20, f"only {len(col)} multiset collisions"
    part = {z for z in F.LIVE for v in (0, 1, 2)
            if tuple(sorted(F.realise(z, v))) in col}
    assert len(part) >= 40, f"only {len(part)} forms participate"


def test_collisions_are_genuine_same_bag_different_meaning():
    col = F.collisions(F.LIVE)
    bag, meanings = next(iter(col.items()))
    assert len(meanings) > 1
    hits = [z for z in F.LIVE for v in (0, 1, 2)
            if tuple(sorted(F.realise(z, v))) == bag]
    assert len({F.denote(z) for z in hits}) > 1


def test_no_surface_string_is_ambiguous():
    """The collisions are of BAGS, not of strings: word order resolves them,
    so the denotation ceiling is 1.00 and a shortfall is a learning failure
    rather than a data ceiling."""
    strings = {}
    for z in F.LIVE:
        for v in (0, 1, 2):
            strings.setdefault((v, tuple(F.realise(z, v))),
                               set()).add(F.denote(z))
    assert all(len(v) == 1 for v in strings.values())


def test_context_beats_bag_of_words_on_collisions(trained):
    """F1, at reduced training cost. The margin is smaller than the full run
    but the ordering has to hold."""
    col = F.collisions(F.LIVE)
    c, n = F.denot_acc(trained["ctx"], trained["test"], only=col)
    b, _m = F.denot_acc(trained["bow"], trained["test"], only=col)
    assert n > 0, "no collision cases in this test split"
    assert c >= b, f"contextual {c:.2f} did not beat bag-of-words {b:.2f}"


def test_the_gradient_is_averaged_not_summed():
    """A summed gradient made the effective step scale with the dataset:
    504 examples gave 0.58 on validation and 637 gave 0.04. That was
    divergence, and the fix must not be undone."""
    import inspect
    src = inspect.getsource(F.Parser._fit_fixed)
    assert "g / m" in src, "the gradient is no longer averaged"


def test_the_authored_control_collapses_on_this_realiser():
    """The point of X64F: a hand-authored word-to-slot table works when the
    realiser is one-to-one and does not work here."""
    dev_p, _v, test_p = F.seeded_splits(101)
    dev, test = F.forms_in(dev_p), F.forms_in(test_p)
    a, _n = F.denot_acc(F.authored_contextual(dev), test)
    assert a < 0.4, f"authored structure reached {a:.2f}; realiser too easy"


def test_freeze_digest_changes_when_the_realiser_changes():
    before = F.freeze_digest()
    saved = F.WEAK[:]
    F.WEAK.append("verily")
    try:
        assert F.freeze_digest() != before
    finally:
        F.WEAK[:] = saved
    assert F.freeze_digest() == before
