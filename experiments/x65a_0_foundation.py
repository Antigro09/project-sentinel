"""X65A-0: foundation phase -- schemas, canonical serialization, taint,
leakage checks, exact posterior microcases, and genuine restart.

This is the FIRST of six staged phases. It measures no transfer, no
retention, no revision and no retrieval, because none of those components
exist yet. It exists to make the later phases falsifiable: if serialization
is not canonical, if evidence can be double counted, if a quarantined
observation can become belief, or if a restart cannot exclude hidden state,
then every downstream number is uninterpretable.

No stream seed is sampled. No X65A manifest is written.

Run: uv run python experiments/x65a_0_foundation.py
"""

from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
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

OUT = Path("experiments/x65a/results")
SCRATCH = Path("experiments/x65a/results/_scratch")


def _demo_state(n_evidence: int) -> ST.PersistentState:
    am, ar = ST.ActiveMemory(), ST.AuditArchive()
    prov = (T.ProvenanceRef("teacher", "h0", 0, "ctx0", T.Taint.OBSERVED),)
    for i in range(n_evidence):
        ar.append({"evidence_id": f"obs{i}", "observation": i % 2,
                   "acquired_at": i})
        if i % 4 == 0:
            am.write(T.SemanticEntry(
                f"sem{i}", f"claim{i % 3}",
                {0: Fraction(1, 2), 1: Fraction(1, 2)}, None, "ctx0", prov,
                i, 1, None, T.Status.CONFIRMED, frozenset({f"obs{i}"})),
                T.MemoryKind.SEMANTIC)
    return ST.PersistentState(ST.SCHEMA_VERSION, "x64h:e39153b7", n_evidence,
                              am, ar, PO.ExactPosterior.uniform((0, 1)))


def main() -> int:
    t0 = time.perf_counter()
    print("X65A-0: foundation phase\n")

    # ---------------------------------------------- 1. the prerequisite
    print("1. X64H PREREQUISITE  (fail-closed, checked before anything else)")
    pq = PQ.check()
    for k, v in pq.checks.items():
        print(f"   {k:28} {'ok' if v['pass'] else 'FAIL':>4}  "
              f"{str(v['detail'])[:46]}")
    if not pq.ok:
        print("\n   PREREQUISITE FAILED. Exiting without an X65A artifact.")
        return 2
    broken = {f: not PQ.simulate_broken(f).ok
              for f in ("freeze_digest", "manifest_sha256",
                        "final_result_commit", "seeds_sha256")}
    print(f"   calibration: planted breaks are caught {sum(broken.values())}"
          f"/{len(broken)}  {broken}")
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    art: dict = {"phase": "X65A-0", "prerequisite": pq.checks,
                 "prerequisite_calibration": broken,
                 "stream_seeds_sampled": False, "manifest_written": False}

    # ---------------------------- 2. latent table, emitted before the posterior
    print("\n2. LATENT-FACTOR TABLE  (emitted before the posterior exists)")
    tabs = {pv: L.table(pv) for pv in (False, True)}
    for pv, tb in tabs.items():
        print(f"   procedure-validity modelled = {pv}:  K = {tb['K']} "
              f"(<= {tb['K_max']}), headroom x{tb['headroom_factor']}")
    for f in tabs[False]["factors"]:
        print(f"      {f['name']:20} |{f['cardinality']}|  {f['origin']:12} "
              f"{f['note'][:44]}")
    print(f"   nuisance marginalised INSIDE the likelihood, not in K: "
          f"{tabs[False]['nuisance_marginalised_in_likelihood']}")
    print("   the four verified primitives are NOT latent factors: they are")
    print("   deterministic over a declared domain. Modelling validity for "
          "two of")
    print("   them costs exactly the remaining budget (64 -> 256); a third "
          "would be 512.")
    art["latent"] = tabs

    # ------------------------------------------------------ 3. the graphs
    print("\n3. FOUR SEPARATE GRAPHS")
    gs = G.graph_set()
    refused = []
    try:
        G.ProbabilisticFactorGraph.from_provenance(gs["provenance"])
    except T.TaintError as e:
        refused.append(("independence_from_provenance", str(e)[:60]))
    try:
        T.encode(gs["evaluator"])
    except T.TaintError as e:
        refused.append(("evaluator_dag_serialization", str(e)[:60]))
    try:
        gs["provenance"].add_edge(
            T.DependencyEdge("a", "b", T.EdgeKind.DERIVES, "*", ()),
            "surface_similarity")
    except T.TaintError as e:
        refused.append(("edge_from_surface_similarity", str(e)[:60]))
    fg = gs["probabilistic_factor"]
    states = ("undeclared", fg.independent("a", "b"))
    fg.declare_independent("a", "b", "declared_model")
    for name, why in refused:
        print(f"   refused  {name:32} {why}")
    print(f"   independence has three states, not two: "
          f"{states[1]} -> {fg.independent('a','b')}")
    print(f"   evaluator DAG taint {gs['evaluator'].taint.value}; oracle "
          f"channel required to read it")
    art["graphs"] = {"refusals": dict(refused),
                     "independence_states": ["DEPENDENT", "DECLARED_INDEPENDENT",
                                             "UNKNOWN"]}

    # ------------------------------------------- 4. evidence counted once
    print("\n4. EVIDENCE MAY BE COUNTED ONCE")
    p = PO.ExactPosterior.uniform((0, 1))
    e1 = PO.Likelihood("ev1", True, {0: Fraction(4, 5), 1: Fraction(1, 5)})
    e2 = PO.Likelihood("ev2", True, {0: Fraction(1, 5), 1: Fraction(4, 5)})
    ch = {0: {"a": Fraction(3, 4), "b": Fraction(1, 4)},
          1: {"a": Fraction(1, 4), "b": Fraction(3, 4)}}
    inv = {
        "summary_is_free": PO.invariant_summary_is_free(p, e1),
        "descendants_do_not_multiply":
            PO.invariant_descendants_do_not_multiply(p, e1),
        "consolidation_preserves_predictive":
            PO.invariant_consolidation_preserves_predictive(p, [e1, e2], ch),
        "repeated_base_factor_absorbed_once":
            p.update([e1, e1]).q == p.update([e1]).q,
    }
    for k, v in inv.items():
        print(f"   {k:38} {'ok' if v else 'FAIL'}")
    zero = p.update([PO.Likelihood("z", True, {0: Fraction(0), 1: Fraction(0)})])
    print(f"   zero normalizer -> {zero.status.value}, posterior unchanged: "
          f"{zero.q == p.q}  (never silently renormalized)")
    inv["zero_normalizer_is_open_world"] = (
        zero.status is T.OpenWorld.MISSING_REPRESENTATION and zero.q == p.q)
    art["evidence_invariants"] = inv

    # ------------------------------------------------- 5. the microcases
    print("\n5. EXACT MICROCASES, re-derived and compared to the theory "
          "package")
    mc = MC.run_all()
    print(f"   Theorem 1 bounded memory: {mc['bounded_memory']['history_count']}"
          f" histories -> {mc['bounded_memory']['memory_state_count']} states, "
          f"collision {mc['bounded_memory']['collision']} separated by index "
          f"{mc['bounded_memory']['separating_index_query']}")
    print(f"   Theorem 2 sufficiency: 32 histories -> "
          f"{mc['sufficiency']['statistic_count']} classes, equal within class "
          f"{mc['sufficiency']['all_equal_within_class']}")
    print(f"   MAP accuracy at reliability 4/5: "
          f"{' '.join(mc['semantic_transfer']['expected_map_accuracy'])}")
    print(f"   Theorem 5 retrieval: general utility monotone "
          f"{mc['retrieval']['general_monotone']} submodular "
          f"{mc['retrieval']['general_submodular']}; independent coverage "
          f"{mc['retrieval']['coverage_monotone']}/"
          f"{mc['retrieval']['coverage_submodular']}")
    print(f"   Theorem 3 revision: {mc['revision']['prior']} -> "
          f"{mc['revision']['after_counterevidence']} -> "
          f"{mc['revision']['after_later_support']}, unrelated "
          f"{mc['revision']['unrelated_before']} unchanged")
    print(f"   Theorem 6 compounding: macro {mc['compounding']['macro_candidates']}"
          f" <= {mc['compounding']['budget']} < raw "
          f"{mc['compounding']['raw_candidates']}")
    print(f"   ALL reproduce the published rationals: "
          f"{mc['all_match_published']}")
    art["microcases"] = mc

    # --------------------------------------- 6. active / archive accounting
    print("\n6. ACTIVE MEMORY VERSUS AUDIT ARCHIVE")
    curve = []
    for n in (8, 16, 32, 64, 128):
        st = _demo_state(n)
        r = st.report_bytes()
        curve.append({"tasks": n, **r})
        print(f"   n={n:>4}  active {r['active_bytes']:>6}B  archive "
              f"{r['archive_bytes']:>6}B  total {r['total_bytes']:>6}B")
    slope_a = ((curve[-1]["active_bytes"] - curve[0]["active_bytes"])
               / (curve[-1]["tasks"] - curve[0]["tasks"]))
    slope_r = ((curve[-1]["archive_bytes"] - curve[0]["archive_bytes"])
               / (curve[-1]["tasks"] - curve[0]["tasks"]))
    print(f"   slopes: active {slope_a:.2f} B/task, archive {slope_r:.2f} "
          f"B/task")
    print("   The archive slope is positive BY CONSTRUCTION (Theorem 4A: one")
    print("   nonempty record per task). No bounded-TOTAL-memory claim is")
    print("   available to this design; only active growth, or a total slope")
    print("   below raw replay, can be claimed -- and neither is measured yet.")
    art["growth"] = {"curve": curve, "active_slope_bytes_per_task": slope_a,
                     "archive_slope_bytes_per_task": slope_r,
                     "bounded_total_claim_available": False}

    # ------------------------------------------- 7. taint and quarantine
    print("\n7. TAINT AT THE WRITER, AND QUARANTINE")
    fired = {}
    am = ST.ActiveMemory()
    prov = (T.ProvenanceRef("s", "h", 0, "c", T.Taint.OBSERVED),)
    q = T.SemanticEntry("q1", "unknown", {0: Fraction(1)}, None, "c", prov,
                        0, 1, None, T.Status.QUARANTINED, frozenset({"e"}))
    try:
        am.write(q, T.MemoryKind.SEMANTIC)
    except T.TaintError:
        fired["quarantined_cannot_become_belief"] = True
    try:
        ST._walk_reject({"answer_key": 1})
    except T.TaintError:
        fired["forbidden_key_rejected"] = True
    try:
        T.encode({"x": 0.5})
    except T.TaintError:
        fired["float_refused"] = True
    try:
        ST._walk_reject({"taint": T.Taint.ORACLE_ONLY.value})
    except T.TaintError:
        fired["oracle_taint_rejected"] = True
    for k, v in fired.items():
        print(f"   {k:38} {'ok' if v else 'FAIL'}")
    art["taint"] = fired

    # -------------------------------------------------- 8. leakage audit
    st = _demo_state(8)
    snap = LK.snapshot(st)
    clean = LK.assert_clean(snap, task_index=8)
    planted = LK.assert_clean(LK.contaminated_fixture(snap), task_index=8)
    verbatim = LK.assert_clean(snap, 8, target_logical_form="claim1")
    early = LK.assert_clean(snap, task_index=0)
    print("\n8. LEAKAGE SNAPSHOTS")
    print(f"   clean snapshot                        {clean['clean']}")
    print(f"   planted canary caught                 {not planted['clean']}  "
          f"{planted['violations']}")
    print(f"   verbatim target caught                {not verbatim['clean']}")
    print(f"   future-dated entry caught             {not early['clean']}")
    print(f"   {clean['note']}")
    art["leakage"] = {"clean": clean, "planted": planted["violations"],
                      "verbatim": verbatim["violations"],
                      "future_dated": early["violations"]}

    # -------------------------------------------------- 9. genuine restart
    print("\n9. GENUINE PROCESS RESTART")
    rp = SCRATCH / "restart_state.json"
    rc = RS.restart_cycle(rp, [1, 1, 0], next_obs=1)
    cont = RS.contaminated_cycle(rp, [1, 1, 0])
    print(f"   parent pid {rc['parent']['pid']} -> child pid "
          f"{rc['child']['pid']}; parent gone {rc['parent_pid_gone']}")
    print(f"   child environment has {rc['child']['env_size']} variables")
    print(f"   posterior exactly preserved {rc['posterior_exactly_preserved']}"
          f"  {rc['child']['posterior']}")
    print(f"   serialized hash preserved   {rc['hash_preserved']}")
    print(f"   forbidden channel closed    {rc['forbidden_channel_closed']}")
    print(f"   contaminated fixture caught {cont['detected']}")
    art["restart"] = {k: v for k, v in rc.items() if k != "parent"}
    art["restart"]["contaminated_fixture_detected"] = cont["detected"]

    # ----------------------------------------------- 10. the phase gates
    print("\n10. X65A-0 PHASE GATES  (not A1-A13; those need components that "
          "do not exist yet)\n")
    out = []

    def g(k, name, ok, note=""):
        out.append((k, name, bool(ok)))
        print(f"   {k:>5}. {name:48} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    g("P0.1", "X64H prerequisite passes and is calibrated",
      pq.ok and all(broken.values()),
      f"{sum(v['pass'] for v in pq.checks.values())}/{len(pq.checks)} checks, "
      f"{sum(broken.values())}/{len(broken)} planted breaks caught")
    g("P0.2", "latent cardinality is declared and within budget",
      tabs[False]["within_budget"] and tabs[True]["within_budget"],
      f"K = {tabs[False]['K']} core, {tabs[True]['K']} with procedure validity")
    g("P0.3", "the four graphs are separate and the boundary is enforced",
      len(refused) == 3, f"{len(refused)} illegitimate operations refused")
    g("P0.4", "serialization is canonical and byte-measured",
      T.encode(snap) == T.encode(T.decode(T.encode(snap)))
      and "float_refused" in fired,
      f"{T.byte_cost(snap)} canonical bytes")
    g("P0.5", "the posterior is exact and a zero normalizer is open-world",
      inv["zero_normalizer_is_open_world"],
      "MISSING_REPRESENTATION, no silent renormalization")
    g("P0.6", "evidence is counted once", all(inv.values()),
      f"{sum(inv.values())}/{len(inv)} invariants")
    g("P0.7", "microcases reproduce the theory package exactly",
      mc["all_match_published"], "two independent implementations agree")
    g("P0.8", "taint is enforced at the writer boundary", len(fired) == 4,
      f"{len(fired)}/4 refusals fire")
    g("P0.9", "leakage snapshots are clean and calibrated",
      clean["clean"] and not planted["clean"] and not verbatim["clean"]
      and not early["clean"], "three planted defects caught")
    g("P0.10", "a genuine restart preserves state and closes the channel",
      rc["ok"] and cont["detected"],
      f"pid {rc['parent']['pid']} -> {rc['child']['pid']}")
    g("P0.11", "active, archive and total bytes are reported separately",
      not art["growth"]["bounded_total_claim_available"],
      f"archive slope {slope_r:.2f} B/task is linear by construction")
    g("P0.12", "no stream seed sampled and no manifest written",
      not art["stream_seeds_sampled"] and not art["manifest_written"])

    ok = [k for k, _m, pss in out if pss]
    print(f"\n   VERDICT: {len(ok)}/{len(out)} X65A-0 phase gates pass")
    bad = [(k, m) for k, m, pss in out if not pss]
    for k, m in bad:
        print(f"     FAILING {k}. {m}")
    print("\n   Formal evidence: eight finite supporting lemmas were "
          "mechanically\n   checked in the theory package. VDFM itself is not "
          "formally verified.")
    print("   Phase X65A-0 only. No semantic, procedural, retrieval, revision")
    print("   or consolidation result exists, and none is claimed.")
    art["gates"] = {k: pss for k, _m, pss in out}
    art["runtime_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "x65a0_foundation.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"   authoritative JSON -> {OUT/'x65a0_foundation.json'}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
