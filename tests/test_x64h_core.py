"""X64H pins 3-6, 11-15, 20: the mathematics."""

import math
import random
import sys

import pytest

sys.path.insert(0, "experiments")

from x64h import (convention as C, grammar as G, layer0 as L0,
                  microcase as M, posterior as PO, semantic as S, types as T)

FAM = tuple(C.sample_convention(s) for s in range(900, 906))
FORMS = tuple(S.x64h_forms())


def test_03_inside_equals_brute_force_derivation_sum():
    """Exact means exact: the dynamic program must equal an explicit
    enumeration of derivations, not approximate it."""
    rng = random.Random(11)
    checked = 0
    for phi in FAM[:3]:
        for z in FORMS[:8]:
            for _ in range(3):
                u = G.generate(phi, z, rng)
                a = G.inside(phi, z, u)
                b = G.brute_force_likelihood(phi, z, u)
                assert abs(a - b) < 1e-12, (phi.digest(), z, u, a, b)
                checked += 1
    assert checked >= 50


def test_04_every_utterance_likelihood_is_normalised():
    for phi in FAM[:3]:
        for z in FORMS[:8]:
            total = sum(G.support(phi, z).values())
            assert abs(total - 1.0) < 1e-12, (phi.digest(), z, total)


def test_05_behavioral_likelihood_uses_the_trusted_executor():
    """A fast path that disagrees with the trusted executor is an evaluator
    bug, so there is no fast path: the likelihood calls execute()."""
    z = FORMS[4]
    f = S.execute(z)
    demos = tuple((t, f(t)) for t in S.UNIVERSE[:4])
    assert S.behavioral_loglik(z, demos) == 0.0
    wrong = tuple((t, f(t) + "!") for t in S.UNIVERSE[:1])
    assert S.behavioral_loglik(z, wrong) == -math.inf


def test_06_joint_posterior_normalises_and_matches_hand_enumeration():
    u = ("w1", "w2")
    post, phi_marg, z_marg = M.exact_posterior(u)
    assert abs(sum(post.values()) - 1.0) < 1e-12
    fam = M.micro_family()
    manual = [(i, z) for i, p in enumerate(fam) for z in M.MICRO_Z
              if M.realise(p, z) == u]
    assert sorted(manual) == sorted(post.keys())
    # 8 conventions x 4 meanings; each utterance is realised by exactly 8
    # pairs under uniform priors, so every survivor carries 1/8
    assert len(post) == 8
    assert all(abs(v - 0.125) < 1e-12 for v in post.values())
    assert all(abs(v - 0.25) < 1e-12 for v in z_marg.values())


def test_11_k4_query_microcase_reproduces_the_reference():
    q = L0.query_statistics(4)
    assert q["largest_answer_alphabet"] == 6
    assert abs(q["posterior_entropy_bits"] - math.log2(24)) < 1e-12
    assert abs(q["entropy_lower_bound_questions"] - 1.7737056144690833) < 1e-9
    assert q["optimal_expected_questions"] == pytest.approx(2.0)
    assert q["greedy_information_gain_expected_questions"] == pytest.approx(2.0)
    assert q["random_disagreement_expected_questions"] == pytest.approx(
        2.857142857142857)


def test_11b_separating_probabilities_match_the_closed_form():
    ref = [0.0, 0.09375, 0.41015625, 0.66650390625]
    for m, want in enumerate(ref, start=1):
        assert L0.exact_separating_probability(4, m) == pytest.approx(want)
        assert L0.closed_form_separating_probability(4, m) == pytest.approx(want)


def test_12_conflict_posterior_matches_a_hand_computed_bayes_factor():
    """Matched evidence puts the conflict posterior near the prior floor;
    evidence from a different meaning drives it to one."""
    phi, z = FAM[2], FORMS[9]
    f = S.execute(z)
    u = G.generate(phi, z, random.Random(3))
    lp = [-math.log(len(FAM))] * len(FAM)
    cfg = PO.Config()
    matched = PO.joint(T.Evidence(u, tuple((t, f(t))
                                           for t in S.UNIVERSE[:3])),
                       FAM, lp, FORMS, cfg)
    other = S.execute(FORMS[30])
    mismatched = PO.joint(T.Evidence(u, tuple((t, other(t))
                                              for t in S.UNIVERSE[:3])),
                          FAM, lp, FORMS, cfg)
    assert matched.p_conflict < 0.05
    assert mismatched.p_conflict > 0.95


def test_13_ambiguity_conflict_and_other_are_separate_numbers():
    phi, z = FAM[1], FORMS[6]
    f = S.execute(z)
    u = G.generate(phi, z, random.Random(4))
    p = PO.joint(T.Evidence(u, tuple((t, f(t)) for t in S.UNIVERSE[:2])),
                 FAM, [-math.log(len(FAM))] * len(FAM), FORMS, PO.Config())
    assert set(p.p_other) == {"IN", "UNKNOWN_REALIZATION",
                              "UNKNOWN_MEANING", "UNKNOWN_PROGRAM"}
    assert abs(sum(p.p_other.values()) - 1.0) < 1e-9
    assert isinstance(p.ambiguity, float) and isinstance(p.p_conflict, float)
    assert p.ambiguity != p.p_conflict or p.ambiguity == 0.0


def test_14_dominant_other_prevents_confident_singleton_execution():
    """A single in-class survivor must not be executed when the episode is
    probably outside the model. This is the regression the specification
    asks for by name."""
    from x64h import decision as DE
    beh = {("x",): 1.0}
    post = PO.Posterior({(0, FORMS[0]): 0.0}, beh, 0.0,
                        {"IN": 0.05, "UNKNOWN_REALIZATION": 0.90,
                         "UNKNOWN_MEANING": 0.03, "UNKNOWN_PROGRAM": 0.02},
                        0.0, False, 0.0)
    dec, _d = DE.decide(post, DE.Costs(), DE.Gates(), budget_left=4)
    assert dec is not T.Decision.EXECUTE
    dec2, _d2 = DE.decide(post, DE.Costs(), DE.Gates(), budget_left=0)
    assert dec2 is not T.Decision.EXECUTE


def test_15_probe_equivalence_is_not_called_program_equivalence():
    """Forms sharing an output on the frozen universe may still differ, so
    the grouping is named probe-relative and the classes are kept."""
    classes = S.equivalence_classes(FORMS)
    assert any(len(v) > 1 for v in classes.values())
    assert sum(len(v) for v in classes.values()) == len(FORMS)


def test_20_a_truncated_candidate_set_reports_incomplete():
    from x64h import arms as A, decision as DE
    ctx = A.Context(FAM, FORMS, PO.Config(), DE.Costs(), DE.Gates(),
                    query_universe=tuple(S.UNIVERSE[:12]))
    phi, z = FAM[3], FORMS[11]
    f = S.execute(z)
    ev = T.Evidence(G.generate(phi, z, random.Random(5)),
                    tuple((t, f(t)) for t in S.UNIVERSE[:2]))
    st = T.PosteriorState(tuple([-math.log(len(FAM))] * len(FAM)), "h")
    v, _s = A.run_arm("amortized_convention", T.Episode(ev, 0, "a"), st, ctx,
                      A.Oracle(phi, z), random.Random(1))
    assert v.incomplete_candidates is True
    v2, _s2 = A.run_arm("exact_bayesian_convention", T.Episode(ev, 0, "b"),
                        st, ctx, A.Oracle(phi, z), random.Random(1))
    assert v2.incomplete_candidates is False
