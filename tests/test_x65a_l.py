"""X65A-L regressions: the exact sufficient sketch, latent identity,
provisional assignment, and the construction bugs that had to be fixed."""

import random
import sys

import numpy as np
import pytest

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x65a import l_suite as LS
from x65a import latent_id as LI
from x65a import provisional as P
from x65a import semantic_mem as SM
from x65a.semantic_mem import surviving_mask


@pytest.fixture(scope="module")
def fam():
    return F.Family(F.FamilySpec(overlap="shared"))


@pytest.fixture(scope="module")
def beh(fam):
    return EP.behaviour_table(fam.forms)


@pytest.fixture(scope="module")
def setup(fam, beh):
    cfg = EP.Config(overlap="shared")
    ids = LS.build_identities(fam, 400)
    probes = LS.build_probes(fam, beh, cfg, ids, 400)
    sk = [LI.sketch_of(type("R", (), {"grounded": i.grounded})())
          for i in ids]
    return ids, probes, sk


# ------------------------------------------------------------ the sketch

def test_the_sketch_is_exactly_sufficient(fam, setup):
    ids, probes, sk = setup
    for s, i in zip(sk, ids):
        assert np.array_equal(s.mask(fam), surviving_mask(fam, i.grounded))
    for pr in probes[:12]:
        if not pr.task.live:
            continue
        W = LI.task_weights(fam, pr.task)
        for s, i in zip(sk, ids):
            assert (LI.record_likelihood(fam, s.mask(fam), W)
                    == LI.record_likelihood(
                        fam, surviving_mask(fam, i.grounded), W))


def test_the_sketch_is_far_smaller_than_a_full_record(fam, setup):
    ids, _p, sk = setup
    full = SM.SemanticRecord("id", ids[0].grounded).bytes()
    assert sk[0].bytes() * 4 < full
    assert sum(s.bytes() for s in sk) <= LI.RETRIEVAL_BYTES


def test_the_sketch_carries_no_label_convention_or_provenance(setup):
    _i, _p, sk = setup
    blob = str(sk[0].canon())
    assert set(sk[0].canon()) == {"p"}
    assert "id" not in blob and "phi" not in blob and "ev:" not in blob


# ------------------------------------------------ identity construction

def test_the_order_pair_does_not_collide_with_the_base_pair(fam):
    """order_apart is an involution, so a second application returned the
    original convention and slot 5 silently equalled slot 0."""
    ids = LS.build_identities(fam, 400)
    assert ids[0].phi == ids[1].phi                 # the intended pair
    assert ids[4].phi != ids[0].phi
    assert ids[5].phi != ids[0].phi
    assert len({i.phi for i in ids}) == 7


def test_the_planted_relations_hold(fam):
    ids = LS.build_identities(fam, 400)
    assert fam.ORD[ids[4].phi] != fam.ORD[ids[5].phi]
    d = (int((fam.PO[ids[2].phi] != fam.PO[ids[3].phi]).any())
         + int((fam.PF[ids[2].phi] != fam.PF[ids[3].phi]).any())
         + int((fam.PS[ids[2].phi] != fam.PS[ids[3].phi]).any())
         + int(fam.ORD[ids[2].phi] != fam.ORD[ids[3].phi]))
    assert d >= 1


def test_equivalence_groups_the_same_convention_only(fam):
    ids = LS.build_identities(fam, 400)
    assert set(LS.equivalence_of(ids, 0)) == {0, 1}
    assert LS.equivalence_of(ids, 4) == (4,)


def test_out_of_family_probe_construction_terminates(fam, beh):
    """The first version looped forever whenever every two-token code had an
    in-family reading, which is the common case in the shared alphabet."""
    cfg = EP.Config(overlap="shared")
    ids = LS.build_identities(fam, 400)
    probes = LS.build_probes(fam, beh, cfg, ids, 400)
    assert probes
    assert any(p.kind == "unknown_meaning" for p in probes)


def test_the_shared_alphabet_has_no_out_of_family_two_token_utterance(fam,
                                                                      beh):
    """Reported as a property, not patched around."""
    cfg = EP.Config(overlap="shared")
    ids = LS.build_identities(fam, 400)
    probes = LS.build_probes(fam, beh, cfg, ids, 400)
    assert not any(p.kind == "out_of_family" for p in probes)


def test_the_disjoint_alphabet_does_have_them(beh):
    fam = F.Family(F.FamilySpec(overlap="disjoint_op"))
    b = EP.behaviour_table(fam.forms)
    cfg = EP.Config(overlap="disjoint_op")
    ids = LS.build_identities(fam, 400)
    probes = LS.build_probes(fam, b, cfg, ids, 400)
    assert any(p.kind == "out_of_family" for p in probes)


# ------------------------------------------------------ latent inference

def test_the_identity_posterior_normalises_and_carries_new_and_out(fam,
                                                                   setup):
    ids, probes, sk = setup
    pr = next(p for p in probes if p.task.live)
    masks = [s.mask(fam) for s in sk]
    post = LI.identity_posterior(fam, masks, pr.task)
    assert abs(sum(post.values()) - 1.0) < 1e-9
    assert LI.NEW_IDENTITY in post and LI.OUT_OF_FAMILY in post


def test_removing_a_component_removes_it_from_the_posterior(fam, setup):
    ids, probes, sk = setup
    pr = next(p for p in probes if p.task.live)
    masks = [s.mask(fam) for s in sk]
    a = LI.identity_posterior(fam, masks, pr.task, with_new=False)
    b = LI.identity_posterior(fam, masks, pr.task, with_out=False)
    assert LI.NEW_IDENTITY not in a
    assert LI.OUT_OF_FAMILY not in b


def test_retrieval_respects_the_node_and_byte_budget(fam, setup):
    ids, probes, sk = setup
    pr = next(p for p in probes if p.task.live)
    keep, _m, stats = LI.retrieve(fam, sk, pr.task)
    assert len(keep) <= LI.SHORTLIST
    assert stats["bytes_retrieved"] <= LI.RETRIEVAL_BYTES
    assert stats["nodes_retrieved"] <= 4
    assert stats["incomplete_retrieval"] is (len(sk) > LI.SHORTLIST)


def test_a_new_identity_needs_grounded_evidence(fam, setup):
    """One ambiguous utterance may not create an identity."""
    ids, probes, sk = setup
    rng = random.Random(1)
    new = [p for p in probes if p.kind == "new" and p.task.live]
    assert new
    outcome, _b, _best, _s = LI.resolve_identity(
        fam, sk, new[0].task, new[0].phi_true, "main", list(range(fam.m)),
        rng, budget=0)
    assert outcome != LI.CREATE_NEW


def test_ambiguous_identity_never_assigns(fam, setup):
    ids, probes, sk = setup
    rng = random.Random(1)
    for pr in probes[:30]:
        if not pr.task.live:
            continue
        outcome, branch, _b, _s = LI.resolve_identity(
            fam, sk, pr.task, pr.phi_true if pr.phi_true >= 0 else 0, "main",
            list(range(fam.m)), rng)
        if outcome == LI.UNRESOLVED_IDENTITY:
            assert branch.status == LI.UNRESOLVED_IDENTITY


def test_an_unknown_meaning_yields_missing_representation(fam, setup):
    ids, probes, sk = setup
    rng = random.Random(1)
    unk = [p for p in probes if p.kind == "unknown_meaning"]
    assert unk
    for pr in unk:
        assert not pr.task.live


def test_the_forced_assimilation_calibration_arm_fires(fam, setup):
    """Removing NEW_IDENTITY alone pushes cases to UNRESOLVED and tests
    nothing; the calibration arm must force a decision."""
    ids, probes, sk = setup
    new = [p for p in probes if p.kind == "new" and p.task.live]
    assert new
    forced = main_forced = 0
    rng = random.Random(1)
    for pr in new:
        o, _b, _p, _s = LI.resolve_identity(fam, sk, pr.task, pr.phi_true,
                                            "no_new_forced",
                                            list(range(fam.m)), rng)
        forced += o == LI.ASSIGN_EXISTING
        o2, _b2, _p2, _s2 = LI.resolve_identity(
            fam, sk, pr.task, pr.phi_true, "main", list(range(fam.m)), rng)
        main_forced += o2 == LI.ASSIGN_EXISTING
    # not every probe can be forced: when no stored record explains the
    # evidence at all the outcome is MISSING before any assignment
    assert forced > len(new) // 2
    assert main_forced == 0


def test_main_beats_random_and_recency_retrieval(fam, setup):
    ids, probes, sk = setup
    got = {}
    for arm in ("main", "random_record", "most_recent"):
        rng = random.Random(1)
        ok = n = 0
        for pr in probes:
            if pr.kind != "returning" or not pr.task.live:
                continue
            _o, _b, best, _s = LI.resolve_identity(
                fam, sk, pr.task, pr.phi_true, arm, list(range(fam.m)), rng,
                budget=1)
            ok += best == pr.task.z
            n += 1
        got[arm] = ok / n
    assert got["main"] > got["random_record"] + 0.2
    assert got["main"] > got["most_recent"] + 0.2
