"""X65A-L1 regressions for matched exact inference and clarification."""

from __future__ import annotations

import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, "experiments")

from x64h import audit0c as A0
from x64h import episode as EP
from x64h import family as F
from x65a import l1_inference as L1
from x65a import l_suite as LS
from x65a.semantic_mem import surviving_mask


def test_selection_cache_cannot_alias_recycled_family_id():
    """A stale integer-id entry must not be a valid cache key in L1."""
    L1._SELECTION_CACHE.clear()
    shared = F.Family(F.FamilySpec(overlap="shared"))
    disjoint = F.Family(F.FamilySpec(overlap="disjoint_op"))
    live = (0, 1, 2)
    pool = F.P2
    u = int(disjoint.realise(0, 0, pool[0]))

    shared_weights = L1.exact_selection_weights(shared, live, 0, pool)
    stale_id_key = (id(disjoint), live, u, tuple(pool))
    L1._SELECTION_CACHE[stale_id_key] = shared_weights

    got = L1.exact_selection_weights(disjoint, live, u, pool)
    assert got.num.shape == (disjoint.n, len(live))
    assert got is not shared_weights


def _fixture(overlap: str):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    ids = LS.build_identities(fam, 400)
    probes = LS.build_probes(
        fam, beh, EP.Config(overlap=overlap), ids, 400)
    pr = next(p for p in probes
              if p.kind == "returning" and len(p.task.live) > 1)
    masks = [surviving_mask(fam, i.grounded) for i in ids]
    return fam, beh, ids, pr, masks


@pytest.fixture(scope="module")
def shared():
    return _fixture("shared")


@pytest.fixture(scope="module")
def disjoint():
    return _fixture("disjoint_op")


def _contains_float(value) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(k) or _contains_float(v)
                   for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(v) for v in value)
    return False


def test_sha_seed_is_process_independent_and_input_sensitive():
    a = L1.stable_seed("arm", 400, (1, 2, 3))
    assert a == L1.stable_seed("arm", 400, (1, 2, 3))
    assert a != L1.stable_seed("arm", 401, (1, 2, 3))
    # Pin the exact digest-derived value so replacing this with hash() is red.
    assert a == 5320853045005307777


def test_selection_aware_weights_are_exact_and_float_free(shared):
    fam, _beh, _ids, pr, _masks = shared
    exact = L1.exact_selection_weights(
        fam, pr.task.live, pr.task.u, pr.task.pool)
    trusted = A0.selection_weights(
        fam, list(pr.task.live), pr.task.u, pr.task.pool)
    reconstructed = np.asarray([
        [float(exact.fraction(phi, z)) for z in pr.task.live]
        for phi in range(fam.n)
    ])
    assert np.array_equal(reconstructed, trusted)
    assert exact.scale in (1, 2, 3, 6)
    assert not _contains_float(exact.canon())
    assert all(isinstance(exact.fraction(0, z), Fraction)
               for z in pr.task.live)


def test_common_joint_applies_semantic_and_behavioral_answers(shared):
    fam, beh, _ids, pr, _masks = shared
    state = L1.make_memoryless_state(fam, beh, pr.task)
    before = state.normalizer()

    semantic = L1.Query(L1.SEMANTIC, 0)
    s_event = state.truthful_event(semantic, pr.phi_true, pr.task.z)
    after_semantic = state.condition(s_event)
    assert after_semantic.normalizer() > 0
    assert after_semantic.normalizer() <= before
    conv = after_semantic.convention_posterior()
    assert conv
    assert all(int(fam.u3[phi, 0]) == s_event.answer
               for phi, p in conv.items() if p > 0)

    candidates = [k for k in range(len(EP.UNIVERSE))
                  if k not in set(pr.task.demos)
                  and len({beh[z][k] for z in pr.task.live}) > 1]
    assert candidates
    behavioral = L1.Query(L1.BEHAVIORAL, candidates[0])
    b_event = after_semantic.truthful_event(
        behavioral, pr.phi_true, pr.task.z)
    final = after_semantic.condition(b_event)
    task_post = final.task_posterior()
    assert final.normalizer() > 0
    assert all(beh[z][behavioral.item] == b_event.answer
               for z, p in task_post.items() if p > 0)
    assert all(isinstance(p, Fraction) for p in task_post.values())
    assert not _contains_float(final.canon())


def test_matched_q0_q1_and_oracle_query_use_one_risk_measure(shared):
    fam, beh, ids, pr, masks = shared
    stable = L1.make_stable_state(
        fam, beh, pr.task, masks[pr.slot])
    latent = L1.make_latent_state(fam, beh, pr.task, masks)
    audit = L1.matched_retrieval_audit(
        stable, latent, pr.phi_true, pr.task.z,
        legal_behavioral=tuple(range(8)),
        legal_semantic=tuple(range(6)),
        decision_rule=L1.DecisionRule(Fraction(1)),
    )
    assert audit["all_pass"]
    for key in ("q0", "q1", "oracle_query"):
        row = audit[key]
        assert row.matched and row.passed
        assert row.stable_risk <= row.latent_action_risk_under_stable
    assert audit["shared_q1_question"] is not None
    assert audit["oracle_question"] is not None


def test_old_unmatched_and_counted_but_unapplied_semantics_are_red(shared):
    fam, beh, _ids, pr, masks = shared
    stable = L1.make_stable_state(fam, beh, pr.task, masks[pr.slot])
    latent = L1.make_latent_state(fam, beh, pr.task, masks)
    query = L1.Query(L1.SEMANTIC, 0)
    event = stable.truthful_event(query, pr.phi_true, pr.task.z)
    calibration = L1.old_semantics_calibration(stable, latent, event)
    assert calibration["fires"]
    assert calibration["old_unmatched_rejected"]
    assert calibration["answer_not_applied_rejected"]
    assert any("query budget" in x or "history" in x
               for x in calibration["unmatched_reasons"])
    assert "counted answer was not applied to posterior" in \
        calibration["unapplied_reasons"]


def test_all_seven_memoryless_controls_have_q0_to_q4_accounting(disjoint):
    fam, beh, _ids, pr, _masks = disjoint
    curves = L1.memoryless_policy_curves(
        fam, beh, pr.task, pr.phi_true, pr.task.z,
        legal_behavioral=tuple(range(8)),
        legal_semantic=tuple(range(5)),
        budgets=(0, 1, 2, 3, 4), seed=400,
    )
    assert tuple(curves) == L1.MEMORYLESS_POLICIES
    for policy, rows in curves.items():
        assert tuple(rows) == (0, 1, 2, 3, 4)
        previous_history = ()
        for budget, row in rows.items():
            assert row.query_budget == budget
            assert row.queries_asked <= budget
            assert row.answers_applied
            assert sum(dict(row.query_types).values()) == row.queries_asked
            assert len(row.resolution_effects) == row.queries_asked
            assert tuple(e.event for e in row.resolution_effects) == \
                row.state.history
            assert all(not e.cause_changed for e in row.resolution_effects)
            assert len(row.offered_each) == budget
            assert row.queries_offered >= row.queries_asked
            assert row.state.history[:len(previous_history)] == previous_history
            previous_history = row.state.history
            assert isinstance(row.task_risk, Fraction)
            assert row.candidate_classes_before >= row.candidate_classes_after
            assert not _contains_float(row.canon())
            assert row.latent_quantity == L1.LATENT_QUANTITY[policy]

    # Knowing a stable label with no stored posterior is informationally the
    # same as a fresh family-prior learner; this is a planted equality check.
    fresh = curves[L1.FRESH_FAMILY_PRIOR]
    stable_fresh = curves[L1.STABLE_ID_FRESH]
    for q in range(5):
        assert fresh[q].state.history == stable_fresh[q].state.history
        assert fresh[q].task_risk == stable_fresh[q].task_risk
        assert fresh[q].correct == stable_fresh[q].correct


def test_a_counted_query_cannot_pass_accounting_without_an_answer(disjoint):
    fam, beh, _ids, pr, _masks = disjoint
    state = L1.make_memoryless_state(fam, beh, pr.task)
    event = state.truthful_event(
        L1.Query(L1.SEMANTIC, 0), pr.phi_true, pr.task.z)
    bad = L1.ArmEvaluation(
        state, 1, "bad_baseline", L1.DecisionRule(), "all_tasks", 1,
        (event,))
    assert bad.validation_errors() == (
        "counted answer was not applied to posterior",)
