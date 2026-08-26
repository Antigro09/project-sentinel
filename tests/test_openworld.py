"""Pins X64B-1. The property that matters is negative: a singleton version
space must not be executed as though it were knowledge."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x63_sparse_price as P
import x64a_identify as X
import x64b1_openworld as K


def test_challenge_inputs_leak_neither_the_universe_nor_the_held_out_set():
    """Confirming on a held-out tape would launder the score it is graded
    against; confirming on a universe tape would just be another query."""
    assert not (set(K.CHALLENGE) & set(K.UNIVERSE))
    assert not (set(K.CHALLENGE) & set(K.HELD_OUT))


def test_each_rung_adds_exactly_one_thing():
    """The diagnosis is 'which rung recovered it', so the rungs have to be
    nested and differ by one ingredient, or the answer means nothing."""
    alpha = sorted(set("".join(K.UNIVERSE)))
    prev_t, prev_b = None, None
    for lvl in range(len(K.RUNGS)):
        t, b = K.vocab(lvl, alpha)
        if prev_t is not None:
            assert set(prev_t) <= set(t), f"rung {lvl} dropped a test"
            assert set(prev_b) <= set(b), f"rung {lvl} dropped a body"
        prev_t, prev_b = t, b
    assert len(K.shapes(0)) == 1 and len(K.shapes(2)) == 3


def test_confirmation_catches_a_singleton_that_is_wrong():
    """The central claim of X64B-1, and it needs a case where the naive arm
    actually errs -- otherwise confirmation is being credited for nothing."""
    caught = []
    for n, f in K.TASKS.items():
        naive = K.solve_pinned(f, random.Random(5), 4, (n,), False)
        if naive["verdict"] != "answered":
            continue
        if all(P.semit(naive["rep"], t) == f(t) for t in K.HELD_OUT):
            continue
        conf = K.solve_pinned(f, random.Random(5), 4, (n,), True)
        assert conf["verdict"] != "answered", \
            f"{n}: confirmation executed a wrong singleton"
        caught.append(n)
    assert caught, "no wrong singleton occurred; the test proves nothing"


def test_abstention_tracks_adequacy_not_exact_universe_equivalence():
    """`balanced prefix` has no exact U-behaviour match in any pool and is
    still answered correctly on every held-out tape. Gating abstention on
    U-exactness therefore demands a wrong abstention, which is what the
    first draft of B4 did."""
    f = K.TASKS["balanced prefix"]
    tb = tuple(f(t) for t in K.UNIVERSE)
    pools = [K.build(lvl, exclude=("balanced prefix",),
                     gen=1 if lvl == 4 else 0)
             for lvl in range(len(K.RUNGS))]
    assert all(tb not in p for p in pools), "it became U-exact; re-read B4"
    r = K.solve(f, random.Random(5), exclude=("balanced prefix",))
    assert r["verdict"] == "answered"
    assert all(P.semit(r["rep"], t) == f(t) for t in K.HELD_OUT)


def test_expansion_keeps_the_counterexample_across_rungs():
    """A rejection that is forgotten on the next rung lets the same wrong
    hypothesis be re-selected, and the ladder never terminates usefully."""
    f = K.TASKS["reverse"]                    # nothing expressible fits
    r = K.solve(f, random.Random(5), exclude=("reverse",))
    assert r["verdict"] == "none_of_the_above"
    assert len(r["trail"]) >= len(K.RUNGS), r["trail"]


def test_a_target_that_is_present_is_still_answered():
    """Criticism must not turn into refusal: the point is to stop confident
    error, not to stop answering."""
    ok = 0
    for n, f in K.TASKS.items():
        r = K.solve(f, random.Random(5))
        if r["verdict"] == "answered" and all(
                P.semit(r["rep"], t) == f(t) for t in K.HELD_OUT):
            ok += 1
    assert ok >= 9, f"only {ok}/11 answered correctly with the target present"
