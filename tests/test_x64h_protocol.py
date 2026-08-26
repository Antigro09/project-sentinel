"""X64H pins 1, 2, 7-10, 16-19: freeze, leakage, controls and schema."""

import json
import math
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, "experiments")

from x64h import (arms as A, convention as C, decision as DE, grammar as G,
                  layer0 as L0, metrics as MT, microcase as M,
                  persistence as PS, posterior as PO, protocol as PR,
                  queries as Q, semantic as S, types as T)

FAM = tuple(C.sample_convention(s) for s in range(900, 906))
FORMS = tuple(S.x64h_forms())


def _ctx():
    return A.Context(FAM, FORMS, PO.Config(), DE.Costs(), DE.Gates(),
                     query_universe=tuple(S.UNIVERSE[:12]))


def _episode(phi, z, n_demo=2, seed=5):
    f = S.execute(z)
    return T.Evidence(G.generate(phi, z, random.Random(seed)),
                      tuple((t, f(t)) for t in S.UNIVERSE[:n_demo]))


def test_01_every_frozen_field_changes_the_digest():
    base = PR.freeze_digest()
    assert PR.freeze_digest(costs=DE.Costs(wrong_execution=21.0)) != base
    assert PR.freeze_digest(gates=DE.Gates(max_conflict=0.26)) != base
    assert PR.freeze_digest(cfg=PO.Config(prior_conflict=0.11)) != base
    assert PR.freeze_digest(
        cfg=PO.Config(other=PO.OtherModel(prior_in=0.93))) != base
    assert PR.freeze_digest(budget=7) != base
    assert PR.freeze_digest(universe_size=17) != base
    assert PR.freeze_digest() == base


def test_02_final_seeds_require_a_committed_manifest():
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "none.json"
        with pytest.raises(T.TaintError):
            PR.release_final_seeds(path=missing)
        stale = Path(td) / "stale.json"
        stale.write_text(json.dumps({"digest": "0" * 64, "commit": "x",
                                     "created": "x", "python": "x"}))
        with pytest.raises(T.TaintError):
            PR.release_final_seeds(path=stale)


def test_07_convention_posterior_improves_on_a_separating_family():
    """Repeated tasks under one convention must concentrate the convention
    posterior. If it does not, the persistent state is decorative."""
    phi_idx, phi = 2, FAM[2]
    lp = [-math.log(len(FAM))] * len(FAM)
    trace = []
    for k, z in enumerate(FORMS[3:9]):
        ev = _episode(phi, z, n_demo=1, seed=20 + k)
        post = PO.joint(ev, FAM, lp, FORMS, PO.Config())
        cp = PO.convention_posterior(post.log_joint, len(FAM))
        tot = PO.logsumexp(cp)
        trace.append(math.exp(cp[phi_idx] - tot))
        lp = [c - tot for c in cp]
    assert trace[-1] >= trace[0]
    assert trace[-1] > 0.9, trace


def test_08_duplicate_signatures_keep_a_non_singleton_class():
    ex = L0.indistinguishable_example()
    assert ex["observationally_identical"] is True
    assert ex["class_size_at_least"] >= 2
    fam = M.micro_family_functional()
    observed = [z for z in M.MICRO_Z if z[0] == "a1"]   # a2 never observed
    cls = M.observational_class_given(fam[0], observed, fam)
    assert len(cls) > 1, "a non-separating observation set collapsed to one"
    # and a separating observation set does not
    full = M.observational_class_given(fam[0], list(M.MICRO_Z), fam)
    assert len(full) < len(cls)


def test_09_no_memory_and_shuffled_history_remove_persistent_evidence():
    ctx, phi, z = _ctx(), FAM[3], FORMS[11]
    ev = _episode(phi, z)
    informed = T.PosteriorState(
        tuple(0.0 if i == 3 else -50.0 for i in range(len(FAM))), "h")
    rng = random.Random(1)
    base = A._prior("exact_bayesian_convention", informed, ctx, rng)
    none = A._prior("no_convention_memory", informed, ctx, rng)
    shuf = A._prior("shuffled_convention_history", informed, ctx,
                    random.Random(2))
    assert base == list(informed.log_p_phi)
    assert len(set(none)) == 1, "no-memory kept a non-uniform prior"
    assert sorted(shuf) == sorted(informed.log_p_phi)
    assert shuf != list(informed.log_p_phi), "shuffle was a no-op"


def test_10_infogain_and_random_share_pool_and_stopping():
    phi = FAM[1]
    ev = _episode(phi, FORMS[6], n_demo=0, seed=31)
    post = PO.joint(ev, FAM, [-math.log(len(FAM))] * len(FAM), FORMS,
                    PO.Config())
    pool = (Q.behavioral_pool(S.UNIVERSE[:10], set(), 1.0)
            + Q.semantic_pool(FORMS, set(), 1.0))
    live = [q for q in pool
            if len(Q.answer_distribution(q, post.log_joint, FORMS, FAM)) > 1]
    ig = Q.choose("infogain", pool, post.log_joint, FORMS, FAM,
                  random.Random(1))
    rd = Q.choose("random", pool, post.log_joint, FORMS, FAM,
                  random.Random(1))
    assert (ig is None) == (rd is None) == (not live), \
        "the policies disagree about whether any question is live"
    if live:
        assert ig in live and rd in live, "a policy left the shared pool"
        best = max(Q.mutual_information(q, post.log_joint, FORMS, FAM)
                   for q in live)
        assert Q.mutual_information(ig, post.log_joint, FORMS,
                                    FAM) == pytest.approx(best)


def test_16_restart_restores_only_observed_evidence():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.json"
        st = T.PosteriorState((-0.1, -2.0, -3.0), "hash-A", ("obs1",))
        PS.save(path, st)
        back = PS.load(path, 3, "hash-A")
        # save normalises: a persisted prior must be a probability
        # distribution. Writing the raw marginal let accumulated evidence
        # leak into the prior's scale, which made the persistent arm strictly
        # worse than having no memory at all.
        assert abs(math.log(sum(math.exp(x) for x in back.log_p_phi))) < 1e-9
        order_in = sorted(range(3), key=lambda i: st.log_p_phi[i])
        order_out = sorted(range(3), key=lambda i: back.log_p_phi[i])
        assert order_in == order_out, "normalisation reordered the prior"
        assert back.observation_hashes == ("obs1",)
        payload = json.loads(path.read_text())
        assert set(payload) == {"log_p_phi", "model_hash",
                                "observation_hashes"}
        with pytest.raises(T.TaintError):
            PS.load(path, 3, "hash-B")


def test_17_taint_rejects_target_oracle_and_future_fields():
    for t in (T.Taint.TARGET_ONLY, T.Taint.ORACLE_ONLY, T.Taint.FUTURE):
        with pytest.raises(T.TaintError):
            T.Tainted("x", t).read(T.PERSISTABLE)
    assert T.Tainted("x", T.Taint.PUBLIC).read(T.PERSISTABLE) == "x"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        with pytest.raises(T.TaintError):
            PS.save(p, T.PosteriorState((0.0,), "h"),
                    {"convention": T.Taint.ORACLE_ONLY})
    ep = T.Episode(_episode(FAM[0], FORMS[0]), 0, "arm")
    assert PR.taint_audit(ep) == []
    assert PR.taint_audit({"ok": 1, "phi": "leak"}) == ["root.phi"]


def test_18_every_arm_emits_the_same_complete_schema():
    ctx, phi, z = _ctx(), FAM[3], FORMS[11]
    ev = _episode(phi, z)
    st = T.PosteriorState(tuple([-math.log(len(FAM))] * len(FAM)), "h")
    orc = A.Oracle(phi, z)
    keys = None
    for arm in A.ARMS:
        v, _s = A.run_arm(arm, T.Episode(ev, 0, arm), st, ctx, orc,
                          random.Random(1))
        acc = MT.blank(arm)
        MT.accumulate(acc, v, v.program == S.denote(z))
        acc = MT.finish(acc)
        assert set(acc) == set(MT.SCHEMA), arm
        keys = keys or set(acc)
        assert set(acc) == keys, arm
    assert len(A.ARMS) == 14


def test_19_the_oracle_convention_arm_succeeds_on_well_formed_items():
    """A failure here is a grammar or executor error, not an inference
    result -- which is what makes it a useful diagnostic."""
    ctx = _ctx()
    st = T.PosteriorState(tuple([-math.log(len(FAM))] * len(FAM)), "h")
    ok = 0
    for k, z in enumerate(FORMS[:10]):
        phi = FAM[k % len(FAM)]
        ev = _episode(phi, z, n_demo=2, seed=40 + k)
        v, _s = A.run_arm("oracle_convention", T.Episode(ev, 0, "o"), st,
                          ctx, A.Oracle(phi, z), random.Random(1))
        ok += (v.decision is T.Decision.EXECUTE
               and v.program == S.denote(z))
    assert ok >= 9, f"oracle convention solved only {ok}/10"


def test_02b_the_oracle_channel_refuses_every_other_arm():
    orc = A.Oracle(FAM[0], FORMS[0])
    for arm in A.ARMS:
        if arm != "oracle_convention":
            with pytest.raises(T.TaintError):
                orc.convention(arm)
        if arm != "oracle_task_meaning":
            with pytest.raises(T.TaintError):
                orc.meaning(arm)
