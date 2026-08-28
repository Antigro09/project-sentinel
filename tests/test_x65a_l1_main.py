"""Regressions for the one production open-world L1 inference adapter."""

from __future__ import annotations

import sys
from fractions import Fraction

import pytest

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x65a import l1_main as M
from x65a import l_suite as LS
from x65a import semantic_mem as SM
from x65a.latent_id import NEW_IDENTITY, OUT_OF_FAMILY


def _fixture(overlap: str):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    ids = LS.build_identities(fam, 6400)
    probes = LS.build_probes(
        fam, beh, EP.Config(overlap=overlap), ids, 6400)
    probe = next(p for p in probes if p.slot >= 0 and p.task.live)
    masks = [SM.surviving_mask(fam, i.grounded) for i in ids]
    return fam, ids, probe, masks


@pytest.mark.parametrize("overlap", ("shared", "disjoint_op"))
def test_latent_state_really_contains_new_and_out(overlap):
    fam, _ids, probe, masks = _fixture(overlap)
    state = M.latent_state(fam, probe.task, masks)
    post = state.identity_posterior()
    assert NEW_IDENTITY in post and post[NEW_IDENTITY] >= 0
    assert OUT_OF_FAMILY in post and post[OUT_OF_FAMILY] > 0
    assert sum(post.values(), Fraction(0)) == 1


@pytest.mark.parametrize("overlap", ("shared", "disjoint_op"))
def test_stable_risk_never_exceeds_matched_open_world_latent_action(overlap):
    fam, _ids, probe, masks = _fixture(overlap)
    stable = M.stable_state(fam, probe.task, masks[probe.slot], probe.slot)
    latent = M.latent_state(fam, probe.task, masks)
    got = M.matched_risk_audit(stable, latent, probe.phi_true, range(8))
    assert got["all_pass"], got
    for key in ("q0", "q1", "oracle_query"):
        assert got[key]["latent_has_NEW"]
        assert got[key]["latent_has_OUT"]
        assert got[key]["matched_history"]
        assert got[key]["stable_risk"] <= \
            got[key]["latent_action_risk_under_stable"]


@pytest.mark.parametrize("overlap", ("shared", "disjoint_op"))
def test_main_policy_applies_every_counted_answer(overlap):
    fam, _ids, probe, masks = _fixture(overlap)
    initial = M.latent_state(fam, probe.task, masks)
    run = M.run_policy(initial, M.INFORMATION_GAIN, 2, probe.phi_true,
                       probe.task.z, tuple(range(8)), 6400)
    assert run.queries_asked == len(run.state.history)
    assert run.queries_asked <= 2
    assert run.action in probe.task.live
