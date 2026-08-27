"""X65A-0 regressions.

Covers the mandatory pre-final-seed tests from the implementation spec that
apply to this phase, plus the addendum's structural requirements. Tests
belonging to later phases (retrieval, consolidation, streams, arms) are
deliberately absent rather than stubbed: a stubbed gate reads as coverage it
does not have.

Every zero-error check is paired with a planted defect it must reject.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, "experiments")

from x65a import graphs as G
from x65a import latent as L
from x65a import leakage as LK
from x65a import microcases as MC
from x65a import posterior as PO
from x65a import prereq as PQ
from x65a import restart as RS
from x65a import store as ST
from x65a import types as T


@pytest.fixture(scope="module")
def prov():
    return (T.ProvenanceRef("src", "h0", 0, "ctx", T.Taint.OBSERVED),)


# ------------------------------------------------ spec test 1: prerequisite

def test_x65_requires_the_final_frozen_x64h():
    r = PQ.check()
    assert r.ok, r.failures
    assert r.checks["all_gates_passed"]["pass"]
    assert r.checks["both_strata_present"]["pass"]
    assert r.checks["freeze_valid"]["pass"]


@pytest.mark.parametrize("field", ["freeze_digest", "manifest_sha256",
                                   "final_result_commit", "seeds_sha256"])
def test_a_broken_prerequisite_field_fails_closed(field):
    """A prerequisite that cannot fail is not a prerequisite."""
    assert not PQ.simulate_broken(field).ok


def test_the_prerequisite_raises_rather_than_returning_a_flag():
    b = PQ.simulate_broken("freeze_digest")
    with pytest.raises(T.TaintError):
        b.require()


def test_no_x65a_stream_seed_exists_yet():
    assert not Path("experiments/x65a/final_seeds.json").exists()
    assert not Path("experiments/x65a/x65a-freeze-manifest.json").exists()


# ---------------------------------------- addendum 7: latent cardinality

def test_latent_cardinality_is_declared_and_bounded():
    assert L.cardinality(False) == 64
    assert L.cardinality(True) == 256
    assert L.cardinality(True) <= L.K_MAX
    assert len(L.states(False)) == 64
    assert len(L.states(True)) == 256


def test_the_joint_is_really_enumerated_not_hidden_behind_arrays():
    """The addendum forbids hiding a factorized product larger than the
    budget behind separate arrays while calling the joint exact."""
    st = L.states(True)
    assert len(set(st)) == len(st) == L.cardinality(True)


def test_a_third_procedure_validity_factor_would_break_the_budget():
    extra = L.CORE + L.PROCEDURE_VALIDITY + (
        L.Factor("pi_2_valid", (0, 1), "procedural", "one too many"),)
    k = 1
    for f in extra:
        k *= f.cardinality
    assert k == 512 > L.K_MAX


def test_the_table_names_every_factor_and_its_origin():
    tb = L.table()
    assert {f["name"] for f in tb["factors"]} == {
        "phi_0", "phi_1", "sigma_0", "sigma_1", "context",
        "source_reliability"}
    assert all(f["origin"] and f["note"] for f in tb["factors"])


# ------------------------------------------------- addendum 2: four graphs

def test_independence_is_never_inferred_from_missing_provenance():
    p = G.ProvenanceGraph()
    with pytest.raises(T.TaintError):
        G.ProbabilisticFactorGraph.from_provenance(p)


def test_an_undeclared_pair_is_unknown_not_independent():
    f = G.ProbabilisticFactorGraph()
    assert f.independent("a", "b") == "UNKNOWN"
    f.declare_independent("a", "b", "declared_model")
    assert f.independent("a", "b") == "DECLARED_INDEPENDENT"


def test_graph_edges_need_an_admissible_justification():
    p = G.ProvenanceGraph()
    e = T.DependencyEdge("a", "b", T.EdgeKind.DERIVES, "*", ())
    with pytest.raises(T.TaintError):
        p.add_edge(e, "surface_similarity")
    p.add_edge(e, "verified_derivation")
    assert p.successors("a") == ["b"]


def test_each_graph_refuses_the_other_graphs_edge_kinds():
    prog = G.ProgramDependencyGraph()
    with pytest.raises(T.TaintError):
        prog.add_edge(T.DependencyEdge("a", "b", T.EdgeKind.SUPPORTS, "*", ()),
                      "verified_composition")


def test_the_evaluator_dag_is_oracle_only_and_unserializable():
    ev = G.EvaluatorDependencyDAG()
    ev.add("t1", "t5", "prerequisite")
    assert ev.taint is T.Taint.ORACLE_ONLY
    with pytest.raises(T.TaintError):
        T.encode(ev)
    with pytest.raises(T.TaintError):
        ev.oracle_view("AGENT")
    assert ev.oracle_view("ORACLE") == [("t1", "t5", "prerequisite")]


def test_graph_closure_contains_planted_dependents_and_excludes_independents():
    """Spec test 10."""
    p = G.ProvenanceGraph()
    for a, b in (("claim", "schema"), ("schema", "plan")):
        p.add_edge(T.DependencyEdge(a, b, T.EdgeKind.DERIVES, "*", ()),
                   "verified_derivation")
    p.add_node("unrelated_skill")
    cl = p.closure("claim")
    assert cl == {"claim", "schema", "plan"}
    assert "unrelated_skill" not in cl


def test_context_gates_the_closure():
    p = G.ProvenanceGraph()
    p.add_edge(T.DependencyEdge("a", "b", T.EdgeKind.DERIVES, "ctx1", ()),
               "verified_derivation")
    assert p.closure("a", context="ctx1") == {"a", "b"}
    assert p.closure("a", context="ctx2") == {"a"}


# --------------------------------------- spec tests 4-6: exact posterior

def test_the_posterior_is_exact_and_normalizes():
    p = PO.ExactPosterior.uniform((0, 1))
    e = PO.Likelihood("e", True, {0: Fraction(4, 5), 1: Fraction(1, 5)})
    q = p.update([e])
    assert sum(q.q.values(), Fraction(0)) == 1
    assert q.q[0] == Fraction(4, 5)
    q.check()


def test_the_posterior_refuses_floats():
    with pytest.raises(T.TaintError):
        PO.Likelihood("e", True, {0: 0.5, 1: 0.5})


def test_equal_posteriors_give_equal_predictives():
    """Spec test 5 / Theorem 2."""
    ch = {0: {"a": Fraction(3, 4), "b": Fraction(1, 4)},
          1: {"a": Fraction(1, 4), "b": Fraction(3, 4)}}
    a = MC.convention_posterior((1, 1, 0))
    b = MC.convention_posterior((1, 0, 1))
    assert a.q == b.q
    assert a.predictive(ch) == b.predictive(ch)


def test_zero_evidence_returns_an_open_world_state():
    """Spec test 6: never silently renormalize a surviving candidate."""
    p = PO.ExactPosterior.uniform((0, 1))
    z = PO.Likelihood("z", True, {0: Fraction(0), 1: Fraction(0)})
    out = p.update([z])
    assert out.status is T.OpenWorld.MISSING_REPRESENTATION
    assert out.q == p.q
    assert len([s for s in out.states if out.q[s] > 0]) == 2


# ------------------------------------- addendum 3: no evidence double count

def test_a_deterministic_summary_adds_no_likelihood():
    p = PO.ExactPosterior.uniform((0, 1))
    e = PO.Likelihood("e", True, {0: Fraction(4, 5), 1: Fraction(1, 5)})
    assert PO.invariant_summary_is_free(p, e)


def test_many_descendants_do_not_multiply_one_evidence_item():
    p = PO.ExactPosterior.uniform((0, 1))
    e = PO.Likelihood("e", True, {0: Fraction(4, 5), 1: Fraction(1, 5)})
    assert PO.invariant_descendants_do_not_multiply(p, e, n=12)


def test_consolidation_preserves_the_posterior_predictive():
    p = PO.ExactPosterior.uniform((0, 1))
    e1 = PO.Likelihood("a", True, {0: Fraction(4, 5), 1: Fraction(1, 5)})
    e2 = PO.Likelihood("b", True, {0: Fraction(1, 5), 1: Fraction(4, 5)})
    ch = {0: {"x": Fraction(1, 3), "y": Fraction(2, 3)},
          1: {"x": Fraction(2, 3), "y": Fraction(1, 3)}}
    assert PO.invariant_consolidation_preserves_predictive(p, [e1, e2], ch)


def test_the_dedup_rule_is_not_vacuous():
    """PLANTED: a factor that lies about being base is counted twice, which
    is exactly the failure the invariant exists to exclude."""
    p = PO.ExactPosterior.uniform((0, 1))
    e = PO.Likelihood("e", True, {0: Fraction(4, 5), 1: Fraction(1, 5)})
    liar = PO.Likelihood("e-copy", True, dict(e.values))   # a NEW evidence id
    once, twice = p.update([e]), p.update([e, liar])
    assert once.q != twice.q
    assert twice.q[0] == Fraction(16, 17)


# ------------------------------------------ spec test 7: canonical round trip

@pytest.mark.parametrize("kind", ["episodic", "semantic", "procedural",
                                  "negative", "edge"])
def test_every_memory_type_round_trips_canonically(kind, prov):
    made = {
        "episodic": T.EpisodicEntry("e", "u", (), (), (), None, None, (),
                                    None, (), "c", prov, 0),
        "semantic": T.SemanticEntry("s", "claim", {0: Fraction(1)}, None, "c",
                                    prov, 0, 1, None, T.Status.CONFIRMED,
                                    frozenset({"e"})),
        "procedural": T.ProcedureEntry(
            "p", ("SEQ",), None, None, None, (), None, "sig", Fraction(9, 10),
            "c", prov, 0, 1, None, T.Status.CONFIRMED, frozenset({"e"}),
            ("domain",), "interp:abc", "probe:def", None, 1,
            "finite-domain-checked"),
        "negative": T.NegativeEntry("n", "s", "c", None, "refuted", None,
                                    prov, 0, 1, None, T.Status.DEFEATED,
                                    frozenset({"e"})),
        "edge": T.DependencyEdge("a", "b", T.EdgeKind.DERIVES, "*", prov),
    }[kind]
    blob = T.encode(made)
    assert T.encode(T.decode(blob)) == blob
    assert T.byte_cost(made) == len(blob)


def test_a_procedure_records_the_domain_it_was_verified_over(prov):
    """Addendum 9: never call a procedure universally verified."""
    p = T.ProcedureEntry("p", ("SEQ",), None, None, None, (), None, "sig",
                         Fraction(1), "c", prov, 0, 1, None,
                         T.Status.CONFIRMED, frozenset({"e"}), ("d",),
                         "interp:abc", "probe:def", None, 1,
                         "finite-domain-checked")
    assert p.proof_status == "finite-domain-checked"
    for f in ("verification_domain", "trusted_interpreter_digest",
              "probe_set_digest", "continuation_effect_summary",
              "effect_summary_version", "proof_status"):
        assert hasattr(p, f)


# ----------------------------------------- spec test 8: taint at the writer

@pytest.mark.parametrize("payload", [
    {"answer_key": 1}, {"z_true": 2}, {"future": 3}, {"convention": 4},
    {"taint": "ORACLE_ONLY"}, {"taint": "TARGET_ONLY"}, {"taint": "FUTURE"},
])
def test_forbidden_payloads_are_rejected_at_the_writer(payload):
    with pytest.raises(T.TaintError):
        ST._walk_reject(payload)


def test_permitted_taints_pass():
    ST._walk_reject({"taint": "OBSERVED", "value": 1})


def test_a_quarantined_record_cannot_enter_active_memory_as_belief(prov):
    """Addendum 11. X64H's open-world detector fired on only 0.417 of
    out-of-space tasks, so unresolved observations are held, not confirmed."""
    am = ST.ActiveMemory()
    q = T.SemanticEntry("q", "unknown", {0: Fraction(1)}, None, "c", prov, 0,
                        1, None, T.Status.QUARANTINED, frozenset({"e"}))
    with pytest.raises(T.TaintError):
        am.write(q, T.MemoryKind.SEMANTIC)
    ok = T.SemanticEntry("s", "known", {0: Fraction(1)}, None, "c", prov, 0,
                         1, None, T.Status.CONFIRMED, frozenset({"e"}))
    am.write(ok, T.MemoryKind.SEMANTIC)
    assert len(am.retrievable()) == 1


def test_an_unnormalized_posterior_is_not_persistable(tmp_path):
    am, ar = ST.ActiveMemory(), ST.AuditArchive()
    bad = PO.ExactPosterior((0, 1), {0: Fraction(1, 3), 1: Fraction(1, 3)})
    st = ST.PersistentState(ST.SCHEMA_VERSION, "d", 0, am, ar, bad)
    with pytest.raises(T.TaintError):
        ST.save(tmp_path / "s.json", st)


# ----------------------------------------- spec test 9: leakage snapshots

def _state(n=8):
    am, ar = ST.ActiveMemory(), ST.AuditArchive()
    p = (T.ProvenanceRef("s", "h", 0, "c", T.Taint.OBSERVED),)
    for i in range(n):
        ar.append({"evidence_id": f"obs{i}", "acquired_at": i})
        if i % 4 == 0:
            am.write(T.SemanticEntry(f"sem{i}", f"claim{i}",
                                     {0: Fraction(1)}, None, "c", p, i, 1,
                                     None, T.Status.CONFIRMED,
                                     frozenset({f"obs{i}"})),
                     T.MemoryKind.SEMANTIC)
    return ST.PersistentState(ST.SCHEMA_VERSION, "d", n, am, ar,
                              PO.ExactPosterior.uniform((0, 1)))


def test_a_clean_snapshot_passes():
    assert LK.assert_clean(LK.snapshot(_state()), task_index=8)["clean"]


def test_a_planted_canary_is_caught():
    snap = LK.snapshot(_state())
    bad = LK.assert_clean(LK.contaminated_fixture(snap), task_index=8)
    assert not bad["clean"] and "forbidden canary value" in bad["violations"]


def test_a_verbatim_target_is_caught():
    snap = LK.snapshot(_state())
    r = LK.assert_clean(snap, 8, target_logical_form="claim4")
    assert not r["clean"]


def test_a_future_dated_entry_is_caught():
    assert not LK.assert_clean(LK.snapshot(_state()), task_index=0)["clean"]


def test_the_leakage_note_does_not_overclaim():
    r = LK.assert_clean(LK.snapshot(_state()), task_index=8)
    assert "does not establish" in r["note"]


# ------------------------------------- spec tests 27-29: genuine restart

@pytest.fixture(scope="module")
def cycle(tmp_path_factory):
    p = tmp_path_factory.mktemp("x65a") / "state.json"
    return p, RS.restart_cycle(p, [1, 1, 0], next_obs=1)


def test_restart_uses_a_distinct_process_and_the_parent_is_gone(cycle):
    _p, r = cycle
    assert r["ok"], r
    assert r["distinct_process"] and r["parent_pid_gone"]


def test_restart_preserves_the_exact_posterior_and_hash(cycle):
    _p, r = cycle
    assert r["posterior_exactly_preserved"] and r["hash_preserved"]
    assert r["child"]["posterior"] == {"0": "1/5", "1": "4/5"}


def test_the_forbidden_channel_is_closed_after_restart(cycle):
    _p, r = cycle
    assert r["forbidden_channel_closed"]
    assert not r["child"]["forbidden_in_bytes"]
    assert not r["child"]["forbidden_in_env"]
    assert not r["child"]["forbidden_in_globals"]


def test_a_contaminated_fixture_fails(cycle):
    """Spec test 28: the canary must be able to fire."""
    p, _r = cycle
    assert RS.contaminated_cycle(p, [1, 1, 0])["detected"]


def test_the_child_environment_is_scrubbed(cycle):
    _p, r = cycle
    assert r["child"]["env_size"] <= 8
    assert not r["child"]["random_state_inherited"]


def test_nothing_but_schema_state_survives_reload(cycle):
    p, _r = cycle
    d = ST.load(p)
    assert set(d) >= {"schema_version", "posterior", "archive", "active"}
    for forbidden in ("rng", "random_state", "cache", "candidate_pool",
                      "history"):
        assert forbidden not in d


# ------------------------------------------------------- the microcases

def test_all_microcases_reproduce_the_published_rationals():
    r = MC.run_all()
    assert r["all_match_published"], r


def test_bounded_memory_exhibits_a_separating_collision():
    r = MC.bounded_memory()
    assert not r["injective"]
    a, b = r["collision"]
    assert a[r["separating_index_query"]] != b[r["separating_index_query"]]


def test_general_retrieval_utility_is_neither_monotone_nor_submodular():
    r = MC.retrieval()
    assert not r["general_monotone"] and not r["general_submodular"]
    assert r["coverage_monotone"] and r["coverage_submodular"]


def test_revision_is_local_under_declared_factorization():
    """Spec test 11."""
    r = MC.revision()
    assert r["ordering_holds"]
    assert r["unrelated_before"] == r["unrelated_after"]


# --------------------------------------------- active / archive accounting

def test_active_and_archive_bytes_are_reported_separately():
    st = _state(16)
    r = st.report_bytes()
    assert r["active_bytes"] > 0 and r["archive_bytes"] > 0
    assert r["total_bytes"] == r["active_bytes"] + r["archive_bytes"]
    assert r["archive_is_retrievable"] is False


def test_the_archive_grows_at_least_linearly():
    """Theorem 4A. This is why no bounded-TOTAL-memory claim is available."""
    a, b = _state(8).report_bytes(), _state(64).report_bytes()
    assert b["archive_bytes"] >= a["archive_bytes"] + (64 - 8)


def test_the_archive_is_not_reachable_from_retrieval():
    st = _state(16)
    ids = {e.id for e in st.active.retrievable()}
    assert not any(r["evidence_id"] in ids for r in st.archive.records)


def test_the_audit_chain_detects_tampering():
    st = _state(8)
    before = st.archive.chain
    st.archive.records[0] = {"evidence_id": "tampered", "acquired_at": 0}
    replay = ST.AuditArchive()
    for r in st.archive.records:
        replay.append(r)
    assert replay.chain != before


def test_saved_state_round_trips_with_an_identical_hash(tmp_path):
    st = _state(8)
    p = tmp_path / "s.json"
    d = ST.save(p, st)
    assert ST.load(p)["_sha256"] == d
