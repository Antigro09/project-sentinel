"""X64H-0B regressions: the family accounting, the exact symmetry that makes
the testbed valid, the separating calibration sets, and the calibration /
transfer split.

Every check that can pass vacuously is paired with a planted defect it has
to catch.
"""

import math
import sys

import numpy as np
import pytest

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x64h import semantic as S


@pytest.fixture(scope="module")
def shared():
    return F.Family(F.FamilySpec(overlap="shared"))


@pytest.fixture(scope="module")
def disjoint():
    return F.Family(F.FamilySpec(overlap="disjoint_op"))


@pytest.fixture(scope="module")
def beh(shared):
    return EP.behaviour_table(shared.forms)


# ------------------------------------------------------- V0 accounting

def test_raw_counts_match_the_factorisation(shared, disjoint):
    assert disjoint.accounting()["raw_parameter_assignments"] == 2 * 24 * 24 * 2
    assert disjoint.n == 2304
    assert shared.accounting()["raw_parameter_assignments"] == 12 * 24 * 24 * 2
    assert shared.n == 13824


@pytest.mark.parametrize("name", ["shared", "disjoint"])
def test_raw_executable_and_observational_counts_agree(name, request):
    fam = request.getfixturevalue(name)
    a = fam.accounting()
    assert (a["raw_parameter_assignments"]
            == a["unique_executable_conventions"]
            == a["observational_equivalence_classes"] == fam.n)
    assert a["class_size_histogram"] == {1: fam.n}


def test_pinning_the_operator_map_is_a_restriction_not_a_symmetry(disjoint):
    """X64H-0's 1152 was `fix_op=True`. If that were a two-to-one symmetry
    the excluded half would duplicate the kept half. It does not."""
    pin = F.pin_no_operator_symmetry(disjoint)
    assert pin["kept_by_pinning"] == 1152
    assert pin["excluded_by_pinning"] == 1152
    assert pin["excluded_that_duplicate_a_kept_convention"] == 0
    assert pin["pinning_is_a_symmetry"] is False


def test_uniform_prior_entropy_is_log2_of_the_class_count(shared):
    assert len(shared.classes()) == shared.n
    assert math.log2(len(shared.classes())) == pytest.approx(13.7548875, abs=1e-6)


# --------------------------------------------------- the exact symmetry

@pytest.mark.parametrize("name", ["shared", "disjoint"])
def test_an_utterance_says_nothing_about_meaning_without_the_convention(
        name, request):
    fam = request.getfixturevalue(name)
    a = F.symmetry_audit(fam)
    assert a["max_spread_over_meanings"] == 0.0
    assert a["utterance_is_uninformative_without_convention"]


def test_the_symmetry_audit_is_not_vacuous(shared):
    """CALIBRATION ARM. A sub-family is not closed under relabelling, so the
    audit must report a spread there or it is measuring nothing."""
    broken = F.symmetry_audit(F.subset(shared, range(0, shared.n, 7)))
    assert broken["max_spread_over_meanings"] > 0.0
    assert not broken["utterance_is_uninformative_without_convention"]


# --------------------------------------------------- separating families

def test_no_two_calibration_meanings_can_separate_the_family(shared):
    """Two grounded meanings expose two filter values and leave pi_F
    ambiguous between the two unassigned words, so k >= 3."""
    import itertools
    assert not any(shared.separates(c)
                   for c in itertools.combinations(range(shared.m), 2))


def test_three_calibration_meanings_can(shared):
    ms = shared.minimal_separating_size()
    assert ms["k"] == 3
    assert shared.separates(ms["example_idx"])


def test_greedy_finds_a_separating_set(shared):
    gr = shared.greedy_separating(order=range(shared.m))
    assert gr["separating"]
    assert gr["residual_conventions"] == shared.n


# --------------------------------------------------------- leak audits

def test_one_transfer_utterance_leaves_the_whole_family(shared):
    la = shared.one_utterance_audit()
    assert not la["identifies_convention"]
    assert la["fraction_of_family_left"] == 1.0
    assert la["bits_leaked_worst_case"] == 0.0
    assert la["words_not_used_by_every_convention"] == 0


def test_the_disjoint_operator_alphabet_carries_an_order_artifact(disjoint):
    """Reported rather than hidden: an operator codeword in a two-token
    utterance fixes the order bit by position, one bit of 11.17."""
    la = disjoint.one_utterance_audit()
    assert la["fraction_of_family_left"] == 0.5
    assert la["bits_leaked_worst_case"] == pytest.approx(1.0)
    assert not la["identifies_convention"]


def test_a_planted_private_codeword_is_caught(shared):
    pl = F.plant_private_codeword(shared)
    assert pl.one_utterance_audit()["words_not_used_by_every_convention"] > 0


# ---------------------------------------------------- forms and demos

def test_every_typed_form_has_its_own_behaviour(shared):
    assert len(shared.forms) == 32
    assert len({S.denote(z) for z in shared.forms}) == 32


def test_calibration_demonstrations_identify_the_meaning_without_language(
        shared, beh):
    ep = EP.build_episode(shared, beh, EP.Config(), 400)
    for i in ep.cal_idx:
        assert len(ep.tasks[i].live) == 1
        assert ep.tasks[i].live[0] == ep.tasks[i].z


def test_transfer_demonstrations_leave_the_intended_band(shared, beh):
    cfg = EP.Config()
    for seed in (400, 401, 402):
        ep = EP.build_episode(shared, beh, cfg, seed)
        for i in ep.tr_idx:
            assert cfg.ambiguity[0] <= len(ep.tasks[i].live) <= cfg.ambiguity[1]
            assert ep.tasks[i].z in ep.tasks[i].live


def test_the_oracle_identifies_every_accepted_transfer_task(shared, beh):
    """The oracle ceiling is CONSTRUCTED: this pins that construction so it
    is a property of the generator rather than of a lucky seed."""
    ep = EP.build_episode(shared, beh, EP.Config(), 403)
    for i in ep.tr_idx:
        t = ep.tasks[i]
        surv = [k for k in t.live
                if any(shared.realise(ep.phi, k, q) == t.u for q in t.pool)]
        assert surv == [t.z]


def test_calibration_and_transfer_meanings_are_disjoint(shared, beh):
    for seed in (400, 405, 410):
        ep = EP.build_episode(shared, beh, EP.Config(), seed)
        cal = {ep.tasks[i].z for i in ep.cal_idx}
        tr = {ep.tasks[i].z for i in ep.tr_idx}
        assert not (cal & tr)
        assert len(tr) == len(ep.tr_idx)


def test_calibration_covers_every_slot_value(shared, beh):
    for seed in (400, 405, 410):
        ep = EP.build_episode(shared, beh, EP.Config(n_cal=4), seed)
        assert len(ep.coverage["op"]) == len(F.OPS)
        assert len(ep.coverage["filt"]) == len(F.FILTERS_0B)
        assert len(ep.coverage["scope"]) == len(F.SCOPES_0B)


def test_three_calibration_tasks_do_not_cover_and_that_is_reported(
        shared, beh):
    ep = EP.build_episode(shared, beh, EP.Config(n_cal=3), 400)
    assert len(ep.coverage["filt"]) < len(F.FILTERS_0B)


# ----------------------------------------------------------- inference

def test_exact_inference_matches_brute_force(shared, beh):
    """The vectorised joint against an independent enumeration, on a small
    sub-family so the reference can be written the slow obvious way."""
    sub = F.subset(shared, range(0, shared.n, 331))
    ep = EP.build_episode(shared, beh, EP.Config(), 400)
    t = ep.tasks[ep.tr_idx[0]]
    p = np.full(sub.n, 1.0 / sub.n)
    b, conv, _best = EP.infer(sub, p, t.u, t.pool, t.live, t.tie)
    ref = {}
    for i in range(sub.n):
        for j in t.live:
            c = sum(1 for q in t.pool if sub.realise(i, j, q) == t.u)
            if c:
                ref[(i, j)] = (1.0 / sub.n) * (c / len(t.pool))
    tot = sum(ref.values())
    for j in t.live:
        want = sum(v for (i, jj), v in ref.items() if jj == j) / tot
        assert b[j] == pytest.approx(want, abs=1e-12)
    for i in range(sub.n):
        want = sum(v for (ii, j), v in ref.items() if ii == i) / tot
        assert conv[i] == pytest.approx(want, abs=1e-12)


def test_posteriors_stay_normalised_at_every_step(shared, beh):
    ep = EP.build_episode(shared, beh, EP.Config(), 400)
    r = EP.run_arm(shared, beh, ep, "persist", EP.Config(), 400)
    assert r["max_normalisation_error"] < 1e-9


def test_the_static_arm_never_reads_history(shared, beh):
    ep = EP.build_episode(shared, beh, EP.Config(), 400)
    h0 = math.log2(shared.n)
    r = EP.run_arm(shared, beh, ep, "static", EP.Config(), 400)
    assert all(abs(x - h0) < 1e-9 for x in r["prior_H"])
    # planted: an arm that DOES read history must move, or the check above
    # would pass for a leaky implementation too
    rp = EP.run_arm(shared, beh, ep, "persist", EP.Config(), 400)
    assert any(abs(x - h0) > 1e-9 for x in rp["prior_H"])


def test_the_convention_posterior_concentrates_on_the_truth(shared, beh):
    ep = EP.build_episode(shared, beh, EP.Config(), 400)
    r = EP.run_arm(shared, beh, ep, "persist", EP.Config(), 400)
    assert r["entropy"][0] > r["entropy"][-1]
    assert r["mass"][-1] > 0.8
    assert r["mass"][-1] > r["mass"][0]


def test_history_causality(shared, beh):
    cfg = EP.Config()
    ep = EP.build_episode(shared, beh, cfg, 400)
    lp = EP.run_arm(shared, beh, ep, "persist", cfg, 400)["transfer"]
    h = len(lp) // 2
    good = sum(lp[h:]) / len(lp[h:])
    for arm in ("reset", "shuffled", "wrong_pairing"):
        c = EP.run_arm(shared, beh, ep, arm, cfg, 400)["transfer"]
        assert sum(c[h:]) / len(c[h:]) < good


def test_episodes_are_deterministic(shared, beh):
    a = EP.build_episode(shared, beh, EP.Config(), 407)
    b = EP.build_episode(shared, beh, EP.Config(), 407)
    assert [t.z for t in a.tasks] == [t.z for t in b.tasks]
    assert [t.u for t in a.tasks] == [t.u for t in b.tasks]
    assert a.phi == b.phi


def test_token_decoding_round_trips(shared):
    for u in range(shared.A ** 2):
        assert F._tokens(u, 2, shared.A) == [u // shared.A, u % shared.A]
    for u in range(shared.A ** 3):
        assert F._tokens(u, 3, shared.A) == [
            u // shared.A ** 2, (u // shared.A) % shared.A, u % shared.A]


# ---------------------------------------------------- X64H-0C additions

def test_zero_meaning_leakage_does_not_imply_zero_convention_information(
        shared, disjoint):
    """X64H-0B said `0.0 bits leaked`. That was about SUPPORT. The two
    quantities are different and only one of them is zero."""
    from x64h import audit0c as A
    for fam in (shared, disjoint):
        i = A.information_audit(fam)
        assert abs(i["I_Z_U"]) < 1e-12
        assert i["I_Phi_U"] > 0.1
        assert i["one_utterance_task_meaning_accuracy"] == pytest.approx(
            i["chance_task_meaning_accuracy"], abs=1e-12)
        assert (i["one_utterance_convention_accuracy"]
                > i["chance_convention_accuracy"])


def test_full_support_does_not_imply_uniform_posterior(shared):
    """Every convention keeps non-zero mass after one utterance AND the
    posterior is measurably non-uniform. Both at once."""
    from x64h import audit0c as A
    i = A.information_audit(shared)
    assert i["support_over_Phi_after_U_min"] == shared.n
    assert i["one_utterance_convention_accuracy"] > 1.0 / shared.n


def test_selection_weights_match_the_generator(shared, beh):
    """The correctly specified likelihood against a brute-force reference
    written the slow obvious way."""
    from x64h import audit0c as A
    ep = EP.build_episode(shared, beh, EP.Config(), 400)
    t = ep.tasks[ep.tr_idx[0]]
    L = list(t.live)
    W = A.selection_weights(shared, L, t.u, t.pool)
    for i in (0, 7, 1234, shared.n - 1):
        for a, z in enumerate(L):
            good = []
            for pat in t.pool:
                uu = shared.realise(i, z, pat)
                surv = [k for k in L
                        if any(shared.realise(i, k, q) == uu for q in t.pool)]
                if surv == [z]:
                    good.append(uu)
            want = (good.count(t.u) / len(good)) if good else 0.0
            assert W[i, a] == pytest.approx(want, abs=1e-12)


def test_the_true_convention_can_always_generate_an_accepted_task(
        shared, beh):
    """Selection acceptance is not vacuous: the generator only emits tasks
    the true convention could have produced, so the aware likelihood never
    assigns the truth zero probability."""
    from x64h import audit0c as A
    ep = EP.build_episode(shared, beh, EP.Config(), 402)
    for i in ep.tr_idx:
        t = ep.tasks[i]
        W = A.selection_weights(shared, list(t.live), t.u, t.pool)
        assert W[ep.phi, list(t.live).index(t.z)] > 0


def test_the_correct_likelihood_does_not_only_hurt(shared, beh):
    """X64H-0B asserted misspecification `can only cost the treatment arms`.
    Measured, the correct likelihood HELPS it."""
    cfg = EP.Config()
    acc = {}
    for lk in ("naive", "aware"):
        tot = []
        for s in (400, 401, 402, 403):
            ep = EP.build_episode(shared, beh, cfg, s)
            tr = EP.run_arm(shared, beh, ep, "persist", cfg, s,
                            likelihood=lk)["transfer"]
            tot.append(sum(tr) / len(tr))
        acc[lk] = sum(tot) / len(tot)
    assert acc["aware"] > acc["naive"]


def test_the_changed_episode_is_reselected_not_merely_rerealised(
        shared, beh):
    """A random re-realisation would have probability zero under the correct
    likelihood, so the diagnostic would measure the splice."""
    from x64h import audit0c as A
    cfg = EP.Config()
    ep = EP.change_episode(shared, beh, cfg, 400)
    after = [i for i in ep.tr_idx if i >= ep.boundary]
    ok = 0
    for i in after:
        t = ep.tasks[i]
        W = A.selection_weights(shared, list(t.live), t.u, t.pool)
        ok += W[ep.phi_alt, list(t.live).index(t.z)] > 0
    assert ok == len(after)


def test_unconditional_episodes_keep_the_acceptance_flag(shared, beh):
    ep = EP.build_episode(shared, beh, EP.Config(select=False), 400)
    flags = [t.accepted for t in ep.tasks if t.kind == "transfer"]
    assert 0.0 < sum(flags) / len(flags) < 1.0
    sel = EP.build_episode(shared, beh, EP.Config(select=True), 400)
    assert all(t.accepted for t in sel.tasks if t.kind == "transfer")


def test_paired_bootstrap_is_paired(shared):
    from x64h import audit0c as A
    a = [0.9, 0.8, 1.0, 0.95, 0.85, 0.9, 1.0, 0.9]
    b = [0.3, 0.2, 0.4, 0.35, 0.25, 0.3, 0.4, 0.3]
    bs = A.paired_bootstrap(a, b, reps=2000)
    assert bs["lo"] > 0 and bs["excludes_zero"]
    assert A.paired_bootstrap(a, a, reps=2000)["delta"] == 0.0


def test_auroc_matches_a_known_case():
    from x64h import audit0c as A
    assert A.auroc([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)
    assert A.auroc([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)
    assert A.auroc([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.5)
