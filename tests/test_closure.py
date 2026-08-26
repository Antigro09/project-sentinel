"""Pins X64G. The finding is negative and it is about the testbed: a
self-generated controlled language cannot show that induction beats
authoring, because whoever writes the generator can write its inverse."""

import sys

import pytest

sys.path.insert(0, "experiments")

import x64f_context as F
import x64g_closure as G


def test_the_authored_rule_parser_is_exact():
    """The whole verdict rests on this. If a hand-written inverse of the
    realiser is perfect, no induced parser can beat it, and the experiment
    is measuring the generator rather than the mechanism."""
    ok = n = 0
    for z in F.LIVE:
        for v in (0, 1, 2):
            n += 1
            ok += F.denote(G.rule_parse(F.realise(z, v))) == F.denote(z)
    assert ok / n >= 0.99, f"authored rules reach only {ok/n:.2f}"


def test_it_is_also_exact_on_the_collisions():
    """Including the construction bag-of-words provably cannot decode."""
    col = F.collisions(F.LIVE)
    ok = n = 0
    for z in F.LIVE:
        for v in (0, 1, 2):
            toks = F.realise(z, v)
            if tuple(sorted(toks)) not in col:
                continue
            n += 1
            ok += F.denote(G.rule_parse(toks)) == F.denote(z)
    assert n > 0
    assert ok / n >= 0.99


def test_the_x64f_authored_control_was_weaker_than_a_real_one():
    """X64F reported the authored control at 0.02-0.18 and concluded that
    learning beat authoring. That control counted the induced parser's own
    features; it was a tally, not a parser."""
    dev_p, _v, _t = F.seeded_splits(401)
    dev = F.forms_in(dev_p)
    tally, _n = F.denot_acc(F.authored_contextual(dev), F.forms_in(_t))
    rules = sum(1 for z in F.forms_in(_t)
                for v in (0, 1)
                if F.denote(G.rule_parse(F.realise(z, v))) == F.denote(z))
    total = len(F.forms_in(_t)) * 2
    assert tally < 0.5 < rules / total, \
        f"tally {tally:.2f}, rules {rules/total:.2f}"


def test_the_fresh_seeds_are_disjoint_from_the_exposed_ones():
    assert not (set(G.FRESH_SEEDS) & set(G.TAINTED))
    assert len(G.FRESH_SEEDS) >= 4
