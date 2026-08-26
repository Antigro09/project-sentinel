"""Pins X64A. The gates that matter here are the ones that could pass while
measuring nothing, so each of those is pinned together with the known-bad
input it has to catch."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x63_sparse_price as P
import x64a_identify as K


@pytest.fixture(scope="module")
def pool():
    # Smaller sample than the experiment's; enough for every property here.
    p, _built, _t, _b = K.build_pool(random.Random(20260825),
                                     depth_sample=8_000, verbose=False)
    return p


def test_queries_can_never_reach_the_held_out_set():
    """A clarification query that could ask about a held-out tape would make
    the final score meaningless."""
    assert not (set(K.UNIVERSE) & set(K.HELD_OUT))
    assert set(K.EVIDENCE0) <= set(K.UNIVERSE)


def test_the_solver_only_ever_sees_the_target_through_answers(pool):
    """The target function is a synthetic user, not an oracle the search can
    read. Every call it receives must be an input the solver was entitled to
    ask about, and there must be at most BUDGET of them beyond the demos."""
    f = K.TASKS["dedupe adjacent"]
    seen = []

    def watched(t):
        seen.append(t)
        return f(t)

    K.run_arm("disagreement", pool, watched, random.Random(7))
    assert set(seen) <= set(K.UNIVERSE), "the target was asked about a tape " \
                                         "outside the queryable universe"
    assert len(set(seen)) <= len(K.EVIDENCE0) + K.BUDGET


def test_the_pool_carries_no_information_about_which_task_is_asked(pool):
    """Task-independence is what makes seeding the witnesses legitimate: one
    pool, every task, no labels."""
    again, _b, _t, _bb = K.build_pool(random.Random(20260825),
                                      depth_sample=8_000, verbose=False)
    assert set(pool) == set(again), "pool construction consulted something " \
                                    "other than its seed"


def test_g1_calibration_the_check_catches_a_system_with_no_ambiguity_state(pool):
    """`reckless` is X63: answer the simplest survivor from the
    demonstrations alone. G1 is worthless unless it flags that, so this test
    fails if reckless ever stops being catchable."""
    caught, wrong = [], []
    for n, f in K.TASKS.items():
        st, _q, rep, _tr, surv = K.run_arm("simplest", pool, f,
                                           random.Random(7))
        if K.reported("reckless", st) == "identified" and len(surv) > 1:
            caught.append(n)
            # The error is committing to a hypothesis the evidence did not
            # determine, whether or not held-out happens to catch it. Scoring
            # by held-out alone makes this test depend on pool size: with a
            # smaller pool the simplest survivor is more often a witness and
            # reckless gets away with it.
            if K.behaviour(rep) != tuple(f(t) for t in K.UNIVERSE):
                wrong.append(n)
    assert caught, "G1 is VACUOUS: reckless is never caught answering early"
    assert wrong, "G1 is WEAK: reckless commits early but always to the truth"


def test_g2_calibration_the_check_catches_an_always_ambiguous_system(pool):
    ident = 0
    for n, f in K.TASKS.items():
        st, _q, _r, _tr, surv = K.run_arm("disagreement", pool, f,
                                          random.Random(7))
        if len(surv) == 1:
            ident += 1
            assert K.reported("paranoid", st) != "identified"
    assert ident, "G2 is VACUOUS: nothing was ever identified"


def test_refutation_never_discards_a_target_that_is_in_the_pool(pool):
    """Exact replay may only remove hypotheses that actually disagree."""
    for n, f in K.TASKS.items():
        tb = tuple(f(t) for t in K.UNIVERSE)
        if tb not in pool:
            continue
        _st, _q, _r, _tr, surv = K.run_arm("disagreement", pool, f,
                                           random.Random(7))
        assert tb in {b for b, _p in surv}, f"{n}: the truth was refuted"


def test_all_three_states_actually_occur(pool):
    """0 / 1 / many is only a distinction if the run produces all three."""
    seen = set()
    for n, f in K.TASKS.items():
        answers = {t: f(t) for t in K.EVIDENCE0}
        seen.add(K.state_of(K.survivors(pool, K.EVIDENCE0, answers)))
        st, _q, _r, _tr, _s = K.run_arm("disagreement", pool, f,
                                        random.Random(7))
        seen.add(st)
    assert seen == {"identified", "underspecified", "inconsistent"}, seen


def test_disagreement_queries_beat_random_ones(pool):
    """The calibration arm for the whole active-query claim. Fewer seeds than
    the experiment runs, but the effect has to survive them."""
    def total(arm, sd):
        q = h = 0
        for _n, f in K.TASKS.items():
            st, qq, rep, _tr, _s = K.run_arm(arm, pool, f, random.Random(sd))
            q += qq
            h += K.held_out(rep, f) if st == "identified" else 0
        return q, h

    dq, dh = total("disagreement", 0)
    rs = [total("random", sd) for sd in range(6)]
    rq = sum(v[0] for v in rs) / len(rs)
    rh = sum(v[1] for v in rs) / len(rs)
    assert dq < rq and dh >= rh, f"disagreement {dq}/{dh} vs random {rq}/{rh}"
