"""X65A-L1 regressions for sketch sufficiency and retrieval semantics."""

import sys
from collections.abc import Mapping

import pytest

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x64h import audit0c as A0
from x65a import l1_retrieval as L1
from x65a import l_suite as LS
from x65a.types import encode


def _fixture(overlap="shared", seed=400):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    cfg = EP.Config(overlap=overlap)
    identities = LS.build_identities(fam, seed)
    records = {identity.slot:
               type("Record", (), {"grounded": identity.grounded})()
               for identity in identities}
    probes = LS.build_probes(fam, beh, cfg, identities, seed)
    task = next(probe.task for probe in probes if probe.task.live)
    exact = L1.build_global_exact_index(records)
    store = {entry.record_key: entry.sketch for entry in exact.entries}
    coarse = L1.build_coarse_index(exact)
    return fam, identities, records, task, exact, store, coarse


def test_algebraic_certificate_covers_every_required_consequence_exactly():
    proof = L1.sufficiency_certificate()
    assert proof.valid()
    assert {o.name for o in proof.obligations} == set(L1.PROOF_COVERAGE)
    assert proof.proof_kind == "mathematical finite-algebra proof"
    assert proof.proof_assistant_verified is False
    assert proof.differential_role == "corroboration only"
    assert len(proof.proof_document_sha256) == 64
    assert {d.overlap for d in proof.domain_validations} == {
        "shared", "disjoint_op"}
    assert all(d.passed for d in proof.domain_validations)
    # Canonical serialization is also the regression that no float entered
    # the proof/report state.
    blob = encode(proof)
    assert b"stored_posterior" in blob
    assert b"selection_aware_likelihood" in blob


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_same_domain_validator_accepts_authored_model_and_rejects_countermodel(
        overlap):
    authored = L1.validate_sufficiency_domain(
        L1.authored_sufficiency_domain(overlap))
    assert authored.passed, authored.failed_checks
    assert authored.persistent_factor_entries_checked > 1_000_000
    assert authored.legal_grounded_pairs > 0

    planted_spec = L1.planted_nonindicator_hidden_weight_domain(overlap)
    planted = L1.validate_sufficiency_domain(planted_spec)
    witness = L1.countermodel_witness(planted_spec)
    assert not planted.passed
    assert "all_persistent_factors_are_indicators" in planted.failed_checks
    assert "no_hidden_persistent_weights" in planted.failed_checks
    # This is the dangerous countermodel: support equality still holds while
    # a full record and support-only sketch assign different exact mass.
    assert planted.checks["grounded_pair_support_reconstructed_exactly"]
    assert witness.support_preserved and witness.posterior_gap
    assert witness.full_mass != witness.sketch_mass
    encode(authored)
    encode(planted)
    encode(witness)


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_protocol_a_charges_the_global_exact_index_and_makes_no_four_node_claim(
        overlap):
    fam, _ids, _records, task, exact, _store, _coarse = _fixture(overlap)
    result = L1.retrieve_protocol_a(exact, fam, task)
    stats = result.accounting
    assert exact.bytes() <= L1.RETRIEVAL_BYTES
    assert stats.index_bytes_scanned == exact.bytes()
    assert stats.total_retrieval_bytes == exact.bytes()
    assert stats.identity_specific_summaries_inspected == len(exact.entries) == 8
    assert stats.identity_likelihoods_evaluated == 8
    assert len(result.likelihoods) == 8
    assert stats.total_retrieval_node_equivalents == 8
    assert stats.shortlist_size == 4
    assert stats.full_records_loaded == 0
    assert stats.sketch_bytes_loaded == sum(e.sketch.bytes()
                                              for e in exact.entries)
    assert stats.within_512
    assert not stats.four_node_claim
    assert stats.incomplete_retrieval
    # Full ordered-container/key accounting must cost more than the sum of
    # bare summary payloads that X65A-L previously called the index.
    assert exact.bytes() > stats.sketch_bytes_loaded
    contract = L1.protocol_a_accounting_contract(exact)
    assert L1.validate_retrieval_accounting(stats, contract).passed
    planted = L1.planted_undercharged_exact_index_row(stats)
    rejected = L1.validate_retrieval_accounting(planted, contract)
    assert not rejected.passed
    assert "full_canonical_index_charged" in rejected.failed_checks
    assert "global_exact_scan_inspects_all_identities" in rejected.failed_checks
    assert "global_exact_scan_makes_no_four_node_claim" in rejected.failed_checks
    encode(result)  # exact likelihoods are Fractions, never serialized floats


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_protocol_a_control_rankings_are_physically_resource_matched(overlap):
    fam, _ids, _records, task, exact, _store, _coarse = _fixture(overlap)
    rows = [L1.retrieve_protocol_a(
        exact, fam, task, strategy=strategy, seed=6400)
        for strategy in ("exact_likelihood", "random", "recency",
                         "surface_nearest")]
    charged = tuple(
        (row.accounting.index_bytes_scanned,
         row.accounting.identity_specific_summaries_inspected,
         row.accounting.identity_likelihoods_evaluated,
         row.accounting.shortlist_size,
         row.accounting.sketch_bytes_loaded,
         row.accounting.total_retrieval_bytes,
         row.accounting.total_retrieval_node_equivalents)
        for row in rows)
    assert len(set(charged)) == 1
    assert charged[0][-1] == 8
    assert all(row.accounting.incomplete_retrieval for row in rows)
    assert all(L1.validate_retrieval_accounting(
        row.accounting,
        L1.protocol_a_accounting_contract(exact, 4)).passed
        for row in rows)

    exact_all = L1.rerank_protocol_a(
        exact, task, rows[0], 8, strategy="all_records")
    assert exact_all.selected_keys == tuple(range(8))
    assert not exact_all.accounting.incomplete_retrieval
    assert L1.validate_retrieval_accounting(
        exact_all.accounting,
        L1.protocol_a_accounting_contract(exact, 8)).passed


class CountingStore(Mapping):
    def __init__(self, values):
        self.values = dict(values)
        self.loads = []

    def __getitem__(self, key):
        self.loads.append(key)
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_protocol_b_calibrates_nonsufficiency_then_loads_only_four(overlap):
    fam, _ids, _records, task, _exact, store, coarse = _fixture(overlap)
    witness = L1.coarse_collision_witness(coarse, store, fam)
    assert witness is not None
    assert witness.left_support_digest != witness.right_support_digest

    counted = CountingStore(store)
    result = L1.retrieve_protocol_b(coarse, counted, fam, task, witness)
    stats = result.accounting
    assert len(counted.loads) == 4
    assert set(counted.loads) == set(L1.nominate_coarse(coarse, task))
    assert stats.index_bytes_scanned == coarse.bytes()
    assert stats.identity_specific_summaries_inspected == 4
    assert stats.identity_likelihoods_evaluated == 4
    assert stats.shortlist_size == 4
    assert stats.full_records_loaded == 0
    assert stats.total_retrieval_node_equivalents == 4
    assert stats.total_retrieval_bytes == (
        stats.index_bytes_scanned + stats.sketch_bytes_loaded)
    assert stats.total_retrieval_bytes <= L1.RETRIEVAL_BYTES
    assert stats.within_512 and stats.four_node_claim
    assert stats.incomplete_retrieval
    assert result.collision_witness == witness
    contract = L1.protocol_b_accounting_contract(
        coarse, len(store), stats.sketch_bytes_loaded,
        stats.identity_specific_summaries_inspected)
    assert L1.validate_retrieval_accounting(stats, contract).passed
    # The task-time index carries only key and one-bit bucket; exact pairs are
    # loaded from the counted store only after nomination.
    assert set(coarse.canon()) == {"e", "v"}
    assert all(len(entry.canon()) == 2
               and isinstance(entry.canon()[1], int)
               for entry in coarse.entries)
    encode(result)


def test_protocol_b_refuses_an_uncalibrated_nonsufficient_index():
    fam, _ids, _records, task, _exact, store, coarse = _fixture()
    with pytest.raises(Exception, match="nonsufficiency calibration"):
        L1.retrieve_protocol_b(coarse, store, fam, task, None)


def test_weight_caches_are_isolated_across_interleaved_alphabet_strata():
    shared, *_shared_rest = _fixture("shared")
    shared_task = _shared_rest[2]
    disjoint, *_disjoint_rest = _fixture("disjoint_op")
    disjoint_task = _disjoint_rest[2]
    shared_again, *_shared_again_rest = _fixture("shared", seed=401)
    shared_again_task = _shared_again_rest[2]

    ws = L1.exact_selection_weights(shared, shared_task)
    wd = L1.exact_selection_weights(disjoint, disjoint_task)
    ws2 = L1.exact_selection_weights(shared_again, shared_again_task)
    assert L1.family_signature(shared) != L1.family_signature(disjoint)
    assert L1.family_signature(shared) == L1.family_signature(shared_again)
    assert ws.scaled.shape[0] == shared.n == 13824
    assert wd.scaled.shape[0] == disjoint.n == 2304
    assert ws2.scaled.shape[0] == shared_again.n == 13824

    # Deterministically simulate the frozen cache failure caused by a reused
    # Python object id.  L1's corroboration wrapper must discard it before
    # calling the frozen implementation.
    live = tuple(shared_task.live)
    stale_key = (id(shared), live, int(shared_task.u),
                 tuple(shared_task.pool))
    A0._WCACHE[stale_key] = wd.scaled[:, :len(live)].astype(float)
    repaired = L1._frozen_selection_weights_uncached(shared, shared_task)
    assert repaired.shape == (shared.n, len(live))


@pytest.mark.parametrize("overlap", ["shared", "disjoint_op"])
def test_full_record_and_sketch_match_end_to_end_on_reachable_paths(overlap):
    audit = L1.verify_generated_paths(overlap, seeds=(400,), task_limit=6,
                                      query_depth=2)
    assert audit.passed, audit.mismatches
    assert audit.tasks >= 4
    assert audit.reachable_states > audit.tasks
    assert audit.clarification_answers > 0
    assert audit.exact_comparisons > 0
    assert all(value == 0 for value in audit.mismatches.values())
    # These two calibration observations ensure the query proof did not pass
    # merely because every surviving phi happened to carry equal weight.
    assert audit.selection_weight_nonuniform_states > 0
    assert audit.weighted_query_differs_from_uniform > 0
    encode(audit)


def test_clarification_updates_new_support_as_well_as_old_records():
    fam, _ids, records, task, _exact, _store, _coarse = _fixture()
    supports = tuple((key, L1.support_from_full_record(fam, record.grounded))
                     for key, record in sorted(records.items()))
    new = tuple(range(fam.n))
    weights = L1.exact_selection_weights(fam, task)
    identity = L1.exact_identity_posterior(fam, supports, weights, new)
    query, _distribution = L1.choose_query(
        fam, supports, new, weights, identity, range(fam.m))
    assert query is not None
    # Choose a reachable answer.  Both NEW and every existing component are
    # intersected with the same grounded answer event.
    answer = int(fam.u3[supports[0][1][0], query])
    new_after = L1.clarify_support(fam, new, query, answer)
    old_after = tuple((key, L1.clarify_support(fam, support, query, answer))
                      for key, support in supports)
    assert 0 < len(new_after) < len(new)
    post = L1.exact_identity_posterior(
        fam, old_after, weights, new_after)
    assert L1.NEW_IDENTITY in post and L1.OUT_OF_FAMILY in post
    assert sum(post.values()) == 1
