"""X65A-L1 negative-transfer conditions, metrics, and red calibrations."""

from __future__ import annotations

import sys
from fractions import Fraction

import pytest

sys.path.insert(0, "experiments")

from x65a import l1_negative as N
from x65a import l1_main as M
from x65a.latent_id import NEW_IDENTITY, OUT_OF_FAMILY
from x65a.types import decode, encode


@pytest.fixture(scope="module", params=("shared", "disjoint_op"))
def audit(request):
    return N.audit_stratum(request.param)


def _contains_float(value) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(k) or _contains_float(v)
                   for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(v) for v in value)
    return False


def test_all_eight_required_conditions_are_present_and_asserted(audit):
    assert tuple(r.condition.name for r in audit.conditions) == N.CONDITIONS
    for result in audit.conditions:
        assert result.condition.invariants
        assert all(ok for _name, ok in result.condition.invariants)
    assert audit.gates["all_eight_conditions_constructed"]
    assert audit.gates["all_construction_invariants_hold"]


def test_shared_oof_transfer_is_unavailable_but_grounded_test_is_live(audit):
    row = next(r for r in audit.conditions
               if r.condition.name == N.OUT_OF_FAMILY)
    if audit.overlap == "shared":
        assert not row.condition.transfer_constructible
        assert row.condition.task is None
        assert "unconstructible" in row.condition.tested_via
        assert row.main.task_accuracy is None
    else:
        assert row.condition.transfer_constructible
        assert row.condition.task is not None
        assert row.main.task_accuracy == Fraction(0)
        assert row.main.action is None


def test_main_and_calibrated_no_memory_share_budget_loss_and_denominator(audit):
    for result in audit.conditions:
        assert result.matched_protocol
        assert result.main.queries_asked <= audit.query_budget
        assert result.no_memory.queries_asked <= audit.query_budget
        assert result.main.excess_questions == (
            result.main.queries_asked - result.no_memory.queries_asked)
        assert result.no_memory.excess_questions == 0
        assert result.main.query_policy == M.INFORMATION_GAIN
        assert result.no_memory.query_policy == N.INF.TASK_INFORMATION_GAIN
        assert result.main.has_new_component
        assert result.main.has_out_component
        contract = result.protocol_contract
        assert contract["same_current_task"]
        assert contract["same_truthful_answer_channel"]
        assert contract["same_query_budget"]
        assert contract["same_zero_one_task_loss"]
        assert contract["same_budget_exhaustion_stopping_rule"]
        assert contract["same_metric_denominator"]
        assert contract["answers_applied_to_current_posterior"]
        assert contract["different_query_universes_explicit"]
        assert contract["main_legal_query_types"] == ("semantic",)
        assert contract["no_memory_legal_query_types"] == (
            "behavioral", "semantic")
    assert audit.gates["matched_evidence_and_query_budget"]


def test_main_is_noninferior_at_the_frozen_minus_one_twentieth_margin(audit):
    assert audit.accuracy_margin == Fraction(-1, 20)
    for result in audit.conditions:
        if result.accuracy_delta is not None:
            assert isinstance(result.main.task_accuracy, Fraction)
            assert isinstance(result.no_memory.task_accuracy, Fraction)
            assert result.accuracy_delta >= Fraction(-1, 20)
        assert result.noninferior == (
            result.accuracy_delta is None
            or result.accuracy_delta >= Fraction(-1, 20))
    assert audit.gates["main_noninferior_at_frozen_margin"] == all(
        result.noninferior for result in audit.conditions)


def test_main_metrics_cover_every_required_negative_transfer_quantity(audit):
    for result in audit.conditions:
        row = result.main.canon()
        assert set(row) == {
            "task_accuracy", "queries_offered", "queries_asked",
            "excess_questions", "false_confident_actions",
            "established_record_corruption", "provisional_branches",
            "unresolved_outcomes", "action", "query_policy",
            "identity_decision", "has_new_component", "has_out_component"}
        assert row["established_record_corruption"] == 0
    assert any(r.main.provisional_branches > 0 for r in audit.conditions)
    assert any(r.main.unresolved_outcomes > 0 for r in audit.conditions)
    assert audit.gates["main_never_corrupts_established_records"]
    assert audit.gates["main_safety_predicate_holds"]


def test_bad_arms_really_corrupt_force_and_act_falsely(audit):
    c = audit.calibrations
    assert c["immediate_map_write"]["owner_survived_before"]
    assert not c["immediate_map_write"]["owner_survived_after"]
    assert c["immediate_map_write"]["established_record_corruption"] == 1
    assert c["forced_new_assimilation"]["forced_decisions"] == 1
    assert c["forced_new_assimilation"]["established_record_corruption"] == 1
    assert c["no_confirmation_contamination"][
        "established_record_corruption"] == 1
    assert c["forced_stale_action"]["forced_actions"] == 1
    assert c["forced_stale_action"]["false_confident_actions"] == 1
    assert c["false_global_promotion"]["false_global_promotions"] == 1
    assert c["all_fire"]
    assert all(row["rejected_by_main_safety_predicate"]
               for name, row in c.items()
               if name not in ("all_fire", "same_predicate_rejections"))
    assert c["same_predicate_rejections"]
    assert audit.gates["bad_arms_fire"]


def test_authoritative_audit_is_exact_and_float_free(audit):
    payload = decode(encode(audit))
    assert not _contains_float(payload)
    assert payload["accuracy_margin"] == Fraction(-1, 20)
    for row in payload["conditions"]:
        accuracy = row["main"]["task_accuracy"]
        assert accuracy is None or isinstance(accuracy, Fraction)


@pytest.mark.parametrize("overlap,seed", [
    ("shared", 6402),
    ("shared", 7400),
    ("disjoint_op", 7403),
])
def test_stale_calibration_search_fires_beyond_the_first_owner(overlap, seed):
    audit = N.audit_stratum(overlap, seed)
    assert audit.calibrations["forced_stale_action"]["fires"]
    assert audit.gates["bad_arms_fire"]
