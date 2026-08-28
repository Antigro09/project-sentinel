"""X65A-L1 genuine latent-state restart and calibration regressions."""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction

import pytest

sys.path.insert(0, "experiments")

from x65a import restart_l1 as R
from x65a import l1_main as M
from x65a import l1_retrieval as RET
from x65a import l_suite as LS
from x65a import semantic_mem as SM
from x65a.types import decode, encode
from x64h import episode as EP
from x64h import family as F


def _assert_no_float(obj):
    assert not isinstance(obj, float)
    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_no_float(key)
            _assert_no_float(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _assert_no_float(value)


@pytest.mark.parametrize("overlap", ("shared", "disjoint_op"))
def test_parent_dies_and_scrubbed_child_loads_its_exact_state(tmp_path,
                                                               overlap):
    result = R.cycle(tmp_path / f"{overlap}.json", overlap=overlap,
                     run_calibrations=True)
    assert result["ok"], result
    assert result["parent_pid_gone"]
    assert result["parent_pid"] != result["child_pid"]
    assert result["child_loaded_parent_state"]
    assert result["final_hashes_identical"]
    assert (result["uninterrupted_final_sha256"]
            == result["restarted_final_sha256"])
    assert result["loaded_step"] == 0 and result["final_step"] == 1
    assert result["real_main_continuation"]
    assert result["continuation_policy"] == M.INFORMATION_GAIN
    assert len(result["continuation_queries"]) == 1
    assert len(result["continuation_answers"]) == 1
    assert result["forbidden_channel_closed"]
    assert result["child_env_size"] <= 5


@pytest.mark.parametrize("overlap", ("shared", "disjoint_op"))
def test_fixture_payload_is_exact_canonical_and_complete(overlap):
    state = R.fixture_state(overlap)
    blob = encode(state)
    payload = decode(blob)
    _assert_no_float(payload)
    assert set(payload) == R.STATE_FIELDS
    assert set(payload["serialized_hashes"]) == set(R.HASHED_FIELDS) | {
        "metadata"}
    assert (payload["identity_posterior"][R.NEW_IDENTITY]
            == payload["new_mass"])
    assert (payload["identity_posterior"][R.OUT_OF_FAMILY]
            == payload["out_mass"])
    assert sum(payload["identity_posterior"].values(), Fraction(0)) == 1
    assert "z" not in payload["task_evidence"]
    assert payload["query_policy_state"]["history"]
    assert payload["post_query_new_support"]
    assert payload["retrieval_accounting"][
        "total_retrieval_node_equivalents"] == 8
    assert payload["retrieval_accounting"]["incomplete_retrieval"]
    assert not payload["retrieval_accounting"]["four_node_claim"]
    round_tripped = R.state_from_payload(payload)
    assert encode(round_tripped) == blob


def test_every_required_component_has_drop_and_mutation_calibration(tmp_path):
    result = R.cycle(tmp_path / "calibrated.json", overlap="shared",
                     run_calibrations=True)
    expected = {f"{mode}:{field}" for field in R.AUDITED_FIELDS
                for mode in ("drop", "mutate")}
    assert set(result["calibrations"]) == expected
    assert result["all_calibrations_rejected"]
    assert all(v["returncode"] != 0 and v["rejected"]
               for v in result["calibrations"].values())


def test_a_resealed_caller_supplied_state_is_the_actual_checkpoint(tmp_path):
    state = R.fixture_state("shared")
    records = list(state.confirmed_records)
    records[0] = replace(records[0], version=9)
    custom = R.seal_state(replace(state, confirmed_records=tuple(records)))
    R.validate_state(custom)
    result = R.cycle(tmp_path / "custom.json", overlap="shared", state=custom,
                     suffix=R.fixture_suffix(custom), run_calibrations=False)
    assert result["ok"], result
    assert result["checkpoint_sha256"] == R._sha(encode(custom))
    loaded = R.state_from_payload(decode((tmp_path / "custom.json").read_bytes()))
    assert loaded.confirmed_records[0].version == 9


@pytest.mark.parametrize("overlap", ("shared", "disjoint_op"))
def test_actual_eight_record_main_state_survives_restart(tmp_path, overlap):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    identities = LS.build_identities(fam, 6400)
    probes = LS.build_probes(
        fam, beh, EP.Config(overlap=overlap), identities, 6400)
    probe = next(p for p in probes if p.slot >= 0 and p.task.live)
    masks = [SM.surviving_mask(fam, i.grounded) for i in identities]
    records = {i.slot: SM.SemanticRecord(
        f"record:{i.slot}", i.grounded) for i in identities}
    exact = RET.build_global_exact_index(records)
    retrieval = RET.retrieve_protocol_a(
        exact, fam, probe.task, k=4, strategy="exact_likelihood", seed=6400)
    initial = M.subset_state(fam, probe.task, masks,
                             retrieval.selected_keys)
    run = M.run_policy(initial, M.INFORMATION_GAIN, 1, probe.phi_true,
                       probe.task.z, tuple(range(8)), 6400)
    state = R.state_from_main(
        overlap, 6400, identities, run.state, retrieval.selected_keys,
        query_budget=2)
    assert len(state.confirmed_records) == 8
    assert len(state.record_convention_posteriors) == 4
    assert len(state.identity_posterior) == 6
    assert state.retrieval_accounting.total_retrieval_node_equivalents == 8
    assert state.retrieval_accounting.incomplete_retrieval
    reconstructed = R.open_world_from_state(state)
    assert reconstructed.supports == run.state.supports
    assert reconstructed.new_support == run.state.new_support
    assert reconstructed.history == run.state.history
    assert reconstructed.identity_posterior() == run.state.identity_posterior()
    assert reconstructed.convention_posteriors() == \
        run.state.convention_posteriors()
    assert dict(state.identity_posterior)[R.NEW_IDENTITY] == state.new_mass
    assert dict(state.identity_posterior)[R.OUT_OF_FAMILY] == state.out_mass
    suffix = R.truthful_main_suffix(state, probe.phi_true)
    uninterrupted = R.continue_suffix(state, suffix)
    expected_main = run.state.apply_truth(
        suffix[0]["query"], probe.phi_true)
    reconstructed_final = R.open_world_from_state(uninterrupted)
    assert reconstructed_final.supports == expected_main.supports
    assert reconstructed_final.new_support == expected_main.new_support
    assert reconstructed_final.history == expected_main.history
    assert uninterrupted.confirmed_records == state.confirmed_records
    assert uninterrupted.retrieval_shortlist == state.retrieval_shortlist
    assert uninterrupted.retrieval_accounting == state.retrieval_accounting
    result = R.cycle(tmp_path / f"actual-{overlap}.json", overlap=overlap,
                     state=state,
                     suffix=suffix,
                     run_calibrations=False)
    assert result["ok"], result
    assert result["final_hashes_identical"]
    assert result["final_state_identical"]
    assert result["real_main_continuation"]


def test_resealed_synthetic_transition_and_missing_reconstructing_state_fail():
    state = R.fixture_state("shared")
    with pytest.raises(R.RestartIntegrityError, match="keys mismatch"):
        R.advance_state(state, {
            "identity_likelihoods": {}, "convention_likelihoods": {},
            "confirm_observation": None, "branch_updates": {},
            "shortlist": [],
        })

    corrupt = R.seal_state(replace(
        state, post_query_new_support=state.post_query_new_support[1:]))
    with pytest.raises(R.RestartIntegrityError, match="NEW support"):
        R.validate_state(corrupt)


def test_resealed_policy_history_and_selection_weight_corruptions_fail():
    state = R.fixture_state("disjoint_op")
    bad_history = replace(
        state.query_policy_state,
        history=((31, state.query_policy_state.history[0][1]),))
    history_state = R.seal_state(replace(
        state, query_policy_state=bad_history,
        provisional_branches=(replace(
            state.provisional_branches[0], asked=bad_history.history),)))
    with pytest.raises(R.RestartIntegrityError, match="not chosen"):
        R.validate_state(history_state)

    weights = replace(state.selection_weights, scaled_sha256="0" * 64)
    weight_state = R.seal_state(replace(state, selection_weights=weights))
    with pytest.raises(R.RestartIntegrityError, match="selection-aware"):
        R.validate_state(weight_state)


@pytest.mark.parametrize("field", R.AUDITED_FIELDS)
def test_unsealed_required_field_mutation_is_rejected(field):
    state = R.fixture_state("disjoint_op")
    payload = decode(encode(state))
    before = encode(payload[field])
    R._mutate_payload(payload, field, "mutate")
    assert encode(payload[field]) != before
    with pytest.raises(R.RestartIntegrityError):
        R.state_from_payload(payload)
