"""X65A-S2 regressions: the impossibility result, the two tiers, the cause
model, active criticism, and the detectability partition.

The load-bearing tests are that a provisional event cannot write confirmed
state, that the old S1 rule still fails on the population it failed on, and
that the indistinguishable class is treated as empirical adequacy rather
than knowledge.
"""

import random
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, "experiments")

from x64h import family as F
from x65a import provisional as P
from x65a import restart_s2 as R2
from x65a import s2_suite as S2
from x65a.semantic_mem import GroundedObservation, surviving_mask


@pytest.fixture(scope="module")
def fam():
    return F.Family(F.FamilySpec(overlap="shared"))


@pytest.fixture(scope="module")
def cases(fam):
    return S2.build_suite(fam, 400, 40, 40)


# ------------------------------------------------- S2.0 the impossibility

def test_no_one_shot_policy_is_correct_in_both_worlds(fam):
    r = P.impossibility_microcase(fam)
    assert r["constructed"]
    assert r["correct_action"]["world_A"] == P.REJECT
    assert r["correct_action"]["world_B"] == P.PROMOTE
    assert r["min_deterministic_errors"] == 1
    assert r["randomised_total_error_always_one"]


def test_both_worlds_present_the_same_observation(fam):
    r = P.impossibility_microcase(fam)
    a, b, z = r["phi_A"], r["phi_B"], r["meaning"]
    assert fam.u3[a, z] != fam.u3[b, z]
    assert r["event"] == (z, int(fam.u3[b, z]))
    assert set(r["identical_observation"]["posterior_support"]) == {a, b}


# --------------------------------------------------- S2.1 the two tiers

def test_a_provisional_event_never_writes_confirmed_state(fam, cases):
    rng = random.Random(1)
    for c in cases[:60]:
        outcome, conf, branch, _u = P.resolve(
            fam, c.confirmed, c.event, c.phi_true, "main",
            list(range(fam.m)), rng)
        if outcome != P.PROMOTE:
            assert conf is c.confirmed
            assert conf.grounded == c.confirmed.grounded
            assert conf.version == c.confirmed.version


def test_a_branch_is_opened_before_any_decision(fam, cases):
    rng = random.Random(1)
    opened = 0
    for c in cases[:40]:
        _o, _cf, branch, _u = P.resolve(fam, c.confirmed, c.event,
                                        c.phi_true, "main",
                                        list(range(fam.m)), rng)
        opened += branch is not None
    assert opened == 40


# ------------------------------------------------------ the cause model

def test_the_cause_posterior_is_exact_and_normalised(fam):
    m = np.ones(fam.n, dtype=bool)
    cp = P.cause_posterior(fam, m, (0, int(fam.u3[0, 0])))
    assert all(isinstance(v, Fraction) for v in cp.values())
    assert sum(cp.values()) == 1


def test_the_other_likelihood_is_family_defined_not_event_defined(fam):
    """It may not become a sink: it is the marginal over the frozen family
    and does not depend on the partner's convention."""
    z = 3
    for u in np.unique(fam.u3[:, z])[:4]:
        rho = P.other_likelihood(fam, (z, int(u)))
        want = Fraction(int((fam.u3[:, z] == u).sum()), fam.n)
        assert rho == want
        assert 0 < rho < 1


def test_the_cause_model_alone_does_not_fix_the_s1_failure(fam, cases):
    """The impossibility result says it cannot, and it does not."""
    rng = random.Random(1)
    und = [c for c in cases
           if c.kind == "alien" and c.record_class == "underdetermined"]
    bad = 0
    for c in und:
        _o, conf, _b, _u = P.resolve(fam, c.confirmed, c.event, c.phi_true,
                                     "cause_mixture_no_query",
                                     list(range(fam.m)), rng)
        bad += not bool(surviving_mask(fam, conf.grounded)[c.phi_true])
    assert bad == len(und)


def test_the_old_quarantine_rule_still_fails_where_it_failed(fam, cases):
    rng = random.Random(1)
    und = [c for c in cases
           if c.kind == "alien" and c.record_class == "underdetermined"]
    bad = 0
    for c in und:
        _o, conf, _b, _u = P.resolve(fam, c.confirmed, c.event, c.phi_true,
                                     "old_quarantine", list(range(fam.m)),
                                     rng)
        bad += not bool(surviving_mask(fam, conf.grounded)[c.phi_true])
    assert bad == len(und)


# ------------------------------------------- S2.2 / S2.3 safety and plasticity

def test_main_corrupts_nothing_and_promotes_everything_legitimate(fam,
                                                                  cases):
    rng = random.Random(1)
    corrupt = promoted = n_alien = n_legit = 0
    for c in cases:
        if c.kind == "legit_after_corruption":
            continue
        outcome, conf, _b, _u = P.resolve(fam, c.confirmed, c.event,
                                          c.phi_true, "main",
                                          list(range(fam.m)), rng)
        ok = bool(surviving_mask(fam, conf.grounded)[c.phi_true])
        if c.kind == "alien":
            n_alien += 1
            corrupt += not ok
        else:
            n_legit += 1
            promoted += outcome == P.PROMOTE
    assert corrupt == 0
    assert promoted == n_legit


def test_always_quarantine_is_safe_but_has_no_plasticity(fam, cases):
    rng = random.Random(1)
    legit = [c for c in cases if c.kind == "legit"]
    got = [P.resolve(fam, c.confirmed, c.event, c.phi_true,
                     "always_quarantine", list(range(fam.m)), rng)[0]
           for c in legit]
    assert all(o == P.REJECT for o in got)


@pytest.mark.parametrize("arm", ["always_accept", "no_other",
                                 "confirmation_bypass"])
def test_the_planted_unsafe_arms_corrupt(fam, cases, arm):
    rng = random.Random(1)
    alien = [c for c in cases if c.kind == "alien"]
    bad = 0
    for c in alien:
        _o, conf, _b, _u = P.resolve(fam, c.confirmed, c.event, c.phi_true,
                                     arm, list(range(fam.m)), rng)
        bad += not bool(surviving_mask(fam, conf.grounded)[c.phi_true])
    assert bad == len(alien)


# --------------------------------------------------- S2.5 active criticism

def test_information_gain_needs_no_more_questions_than_random(fam, cases):
    q = {}
    rej = {}
    for arm in ("main", "provisional_random"):
        rng = random.Random(1)
        used = 0
        got = 0
        alien = [c for c in cases if c.kind == "alien"]
        for c in alien:
            o, _cf, _b, u = P.resolve(fam, c.confirmed, c.event, c.phi_true,
                                      arm, list(range(fam.m)), rng)
            used += u
            got += o == P.REJECT
        q[arm] = used / len(alien)
        rej[arm] = got / len(alien)
    assert q["main"] <= q["provisional_random"]
    assert rej["main"] >= rej["provisional_random"]


def test_a_challenge_with_no_discrimination_has_zero_gain(fam, cases):
    c = next(x for x in cases if x.kind == "alien")
    mask = surviving_mask(fam, c.confirmed.grounded)
    grounded_z = {g.z for g in c.confirmed.grounded}
    zq = next(iter(grounded_z))
    assert P.challenge_gain(fam, mask, c.event, zq) == 0.0


# ---------------------------------------------- S2.6 the detectability map

def test_class_c_is_empty_under_the_full_challenge_set(fam, cases):
    assert not any(c.detect_class == "indistinguishable" for c in cases)


def test_class_c_appears_only_when_the_challenge_set_is_restricted(fam):
    r = S2.build_suite(fam, 400, 20, 20, legal=range(4))
    assert any(c.detect_class == "indistinguishable" for c in r)


def test_class_b_is_not_constructible_here(fam):
    """Measured, not assumed: no alien needed more than one question at any
    challenge-set size."""
    for k in (2, 4, 8, 32):
        r = S2.build_suite(fam, 401, 15, 15, legal=range(k))
        assert not any(c.detect_class.startswith("multi_query") for c in r)


def test_an_indistinguishable_promotion_is_adequate_inside_and_wrong_outside(
        fam):
    """The honest boundary of S2.6: adequacy inside the tested query set is
    not knowledge of the partner."""
    legal = list(range(4))
    rng = random.Random(1)
    r = [c for c in S2.build_suite(fam, 400, 20, 20, legal=legal)
         if c.detect_class == "indistinguishable"]
    assert r
    inside = outside = 0
    for c in r:
        _o, conf, _b, _u = P.resolve(fam, c.confirmed, c.event, c.phi_true,
                                     "main", legal, rng)
        m = surviving_mask(fam, conf.grounded)
        idx = np.where(m)[0]
        inside += all((fam.u3[idx, z]
                       == int(fam.u3[c.phi_true, z])).all() for z in legal)
        outside += all((fam.u3[idx, z]
                        == int(fam.u3[c.phi_true, z])).all()
                       for z in range(fam.m) if z not in set(legal))
    assert inside == len(r)
    assert outside < len(r)


# ------------------------------------------------------------ S2.7 / S2.8

def test_missing_representation_on_an_already_wrong_record(fam, cases):
    rng = random.Random(1)
    post = [c for c in cases if c.kind == "legit_after_corruption"]
    assert post
    missing = untouched = 0
    for c in post:
        outcome, conf, _b, _u = P.resolve(fam, c.confirmed, c.event,
                                          c.phi_true, "main",
                                          list(range(fam.m)), rng)
        missing += outcome == P.MISSING
        untouched += conf.grounded == c.confirmed.grounded
    assert missing >= 0.9 * len(post)
    assert untouched == len(post)


def test_both_tiers_survive_a_genuine_restart(tmp_path):
    r = R2.cycle(tmp_path / "s2.json", "shared", 400)
    assert r["ok"], r
    assert r["parent_pid_gone"] and r["parent_pid"] != r["child_pid"]
    assert r["confirmed_identical"] and r["provisional_identical"]
    assert r["outcomes_identical"] and r["provisional_branches"] > 0
    assert r["forbidden_channel_closed"] and r["env_size"] <= 8
