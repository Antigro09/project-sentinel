"""Stream-complete genuine restart replication for actual L1 MAIN states."""

from __future__ import annotations

import sys
from fractions import Fraction

import pytest

sys.path.insert(0, "experiments")

from x65a import l1_restart_audit as A
from x65a import restart_l1 as R
from x65a.types import decode, encode


def _assert_no_float(value):
    assert not isinstance(value, float)
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_float(key)
            _assert_no_float(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_float(item)


@pytest.fixture(scope="module")
def matrix_audit(tmp_path_factory):
    contract = A.RestartMatrixContract(
        development_seeds=(6400,), validation_seeds=(7400,))
    cases = tuple(A.make_actual_main_case(overlap, split, seed)
                  for overlap in A.OVERLAPS
                  for split, seed in (("development", 6400),
                                      ("validation", 7400)))
    output = tmp_path_factory.mktemp("main-restart-matrix")
    audit = A.audit_main_restart_matrix(cases, output, contract)
    return contract, cases, audit


def test_every_actual_development_validation_main_stream_restarts(matrix_audit):
    contract, _cases, audit = matrix_audit
    assert audit.schema == A.SCHEMA
    assert audit.validation.passed, audit.validation.errors
    assert len(audit.cases) == len(contract.expected_keys()) == 4
    assert {case.key for case in audit.cases} == set(contract.expected_keys())
    for case in audit.cases:
        state = case.checkpoint_state
        cycle = case.cycle_result
        assert case.post_query_history
        assert len(state.identity_posterior) == 6  # top four + NEW + OUT
        assert sum(dict(state.identity_posterior).values(), Fraction(0)) == 1
        assert dict(state.identity_posterior)[R.NEW_IDENTITY] == state.new_mass
        assert dict(state.identity_posterior)[R.OUT_OF_FAMILY] == state.out_mass
        assert len(state.record_convention_posteriors) == 4
        assert len(state.confirmed_records) == 8
        assert state.provisional_branches
        assert 0 < len(state.retrieval_shortlist) <= 4
        assert {key for key, _support in state.post_query_record_supports} \
            == set(state.retrieval_shortlist)
        assert state.retrieval_accounting.total_retrieval_node_equivalents == 8
        assert state.retrieval_accounting.incomplete_retrieval
        assert not state.retrieval_accounting.four_node_claim
        assert set(dict(state.serialized_hashes)) == set(R.HASHED_FIELDS) | {
            "metadata"}
        assert cycle["ok"] and cycle["parent_pid_gone"]
        assert cycle["parent_pid"] != cycle["child_pid"]
        assert cycle["child_loaded_parent_state"]
        assert (cycle["uninterrupted_final_sha256"]
                == cycle["restarted_final_sha256"])
        assert cycle["final_hashes_identical"]
        assert cycle["final_state_identical"]
        assert cycle["real_main_continuation"]
        assert cycle["continuation_policy"] == "information_gain"
        assert len(cycle["continuation_queries"]) == 1
        assert len(cycle["continuation_answers"]) == 1
        assert cycle["forbidden_channel_closed"]
        assert cycle["child_env_size"] <= 5


def test_required_state_fields_and_calibrations_are_reported_exactly(
        matrix_audit):
    contract, _cases, audit = matrix_audit
    payload = decode(encode(audit))
    _assert_no_float(payload)
    required = {
        "identity_posterior", "record_convention_posteriors", "new_mass",
        "out_mass", "confirmed_records", "provisional_branches",
        "retrieval_shortlist", "retrieval_accounting", "task_evidence",
        "post_query_record_supports", "post_query_new_support",
        "inference_priors", "selection_weights", "query_policy_state",
        "serialized_hashes", "cycle",
        "corrupt_checkpoint_calibrations",
    }
    assert all(required.issubset(case) for case in payload["cases"])
    calibrations = {
        key: value
        for case in audit.cases
        for key, value in case.corrupt_checkpoint_calibrations.items()}
    assert set(calibrations) == set(contract.required_calibration_keys())
    assert all(row["rejected"] and row["returncode"] != 0
               for row in calibrations.values())
    assert all(row["same_child_validator"]
               == "x65a.restart_l1 child/state_from_payload"
               for row in calibrations.values())


def test_same_matrix_validator_rejects_an_accepted_corruption(matrix_audit):
    contract, _cases, audit = matrix_audit
    planted = A.planted_accepted_corruption(audit.cases)
    rejected = A.validate_restart_matrix(planted, contract)
    assert not rejected.passed
    assert not rejected.checks["all_case_rows_valid"]
    assert any("all_scheduled_corrupt_checkpoints_rejected" in error
               for error in rejected.errors)


def test_contract_rejects_an_omitted_validation_stream_before_process_launch(
        matrix_audit, tmp_path):
    contract, cases, _audit = matrix_audit
    with pytest.raises(R.RestartIntegrityError, match="every development"):
        A.audit_main_restart_matrix(cases[:-1], tmp_path, contract)
