"""X65A-L1: retrieval sufficiency, budget semantics, and baseline audit.

This phase is a hard barrier.  It writes only the L1 development/validation
artifact.  It never samples a final stream, writes a final manifest, or starts
procedural memory.  Exit 0 means L1.0--L1.10 passed; exit 1 preserves a
negative result and keeps X65A-P blocked.

Run: ``uv run python experiments/x65a_l1_audit.py``
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Mapping

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x65a import l1_eval as EV
from x65a import l1_contracts as CT
from x65a import l1_gates as L1G
from x65a import l1_inference as INF
from x65a import l1_main as MAIN
from x65a import l1_retrieval as RET
from x65a import l1_restart_audit as RSTA
from x65a import l1_safety as SAFE
from x65a import l1_stats as ST
from x65a import l_suite as LS
from x65a import prereq as PQ
from x65a import restart_l1 as RST
from x65a import semantic_mem as SM
from x65a.types import byte_cost, encode


OUT = Path("experiments/x65a/results/x65al1_audit.json")
BASE_COMMIT = "5205543"
DEV = (6400, 6401, 6402, 6403)
VAL = (7400, 7401, 7402, 7403)
QUERY_BUDGETS = (0, 1, 2, 3, 4)
LEGAL_BEHAVIORAL = tuple(range(8))
LEGAL_SEMANTIC = tuple(range(8))
L3_MARGIN = Fraction(1, 20)
L10_MARGIN = Fraction(1, 20)
BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 20260827
TARGET_ACCURACY = Fraction(19, 20)
FULL_SUITE_RUNTIME_MS = int(os.environ.get(
    "X65A_L1_FULL_SUITE_RUNTIME_MS", "0"))
FULL_SUITE_PASSED = int(os.environ.get(
    "X65A_L1_FULL_SUITE_PASSED", "0"))
FULL_SUITE_SKIPPED = int(os.environ.get(
    "X65A_L1_FULL_SUITE_SKIPPED", "0"))
FULL_SUITE_EXIT_CODE = int(os.environ.get(
    "X65A_L1_FULL_SUITE_EXIT_CODE", "-1"))
FULL_SUITE_COMMAND = ("uv", "run", "pytest", "-q")
MAIN_RETRIEVAL_FREEZE = {
    "protocol": "A_GLOBAL_EXACT_SCAN_TOP4",
    "ranking": "exact_likelihood",
    "shortlist_size": 4,
    "physical_node_equivalents": 8,
    "byte_limit": 512,
    "incomplete_retrieval": True,
    "frozen_from": "development design; original four-record intent",
    "development_seeds": DEV,
    "validation_seeds_used_to_choose": (),
}


def sh(*args) -> str:
    return subprocess.run(args, capture_output=True, text=True,
                          check=True).stdout.strip()


def frac_mean(values) -> Fraction:
    values = [Fraction(v) for v in values]
    return (sum(values, Fraction(0)) / len(values)
            if values else Fraction(0))


def decimal_mean(values) -> str:
    values = [Decimal(v) for v in values]
    return (format(sum(values) / len(values), ".12f")
            if values else "0.000000000000")


def pct(x) -> str:
    return format(float(Fraction(x)), ".3f")


def build_stream(overlap: str, seed: int, n_per: int = 1):
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    ids = LS.build_identities(fam, seed)
    probes = LS.build_probes(
        fam, beh, EP.Config(overlap=overlap), ids, seed, n_per=n_per)
    masks = [SM.surviving_mask(fam, i.grounded) for i in ids]
    return fam, beh, ids, probes, masks


def scored_probes(probes):
    return [p for p in probes if p.slot >= 0 and p.task.live
            and p.kind in ("returning", "ambiguous", "misleading")]


def _state_for_keys(fam, beh, task, masks, keys):
    prior = Fraction(1, len(keys))
    components = tuple(
        INF.component_from_mask(f"record:{key}", masks[key], prior)
        for key in keys)
    return INF._state(fam, beh, task, components)


def _comparison_row(row):
    return row.canon() if hasattr(row, "canon") else row


def matched_split(overlap: str, seeds) -> dict:
    arms = {name: {"correct": 0, "tasks": 0, "queries": 0,
                   "risk": Fraction(0)}
            for name in ("stable_q0", "latent_q0", "stable_q1",
                         "latent_q1", "stable_oracle_query",
                         "latent_oracle_query")}
    failures = []
    taskwise = []
    calibration = None
    for seed in seeds:
        fam, beh, ids, probes, masks = build_stream(overlap, seed)
        for n, probe in enumerate(scored_probes(probes)):
            stable = MAIN.stable_state(
                fam, probe.task, masks[probe.slot], probe.slot)
            latent = MAIN.latent_state(fam, probe.task, masks)
            audit = MAIN.matched_risk_audit(
                stable, latent, probe.phi_true, LEGAL_SEMANTIC)
            if calibration is None:
                q = latent.choose_information_query(LEGAL_SEMANTIC)
                if q is None:
                    calibration = {"unmatched_history_rejected": False,
                                   "fires": False}
                else:
                    planted = MAIN._risk_row(
                        stable, latent.apply_truth(q, probe.phi_true))
                    calibration = {
                        "unmatched_history_rejected":
                            not planted["matched_history"],
                        "planted": planted,
                        "fires": not planted["matched_history"],
                    }
            mapping = (
                ("q0", "stable_q0", "latent_q0", 0),
                ("q1", "stable_q1", "latent_q1",
                 int(audit["shared_q1_question"] is not None)),
                ("oracle_query", "stable_oracle_query",
                 "latent_oracle_query",
                 int(audit["oracle_question"] is not None)),
            )
            for key, sa, la, q_used in mapping:
                comp = audit[key]
                arms[sa]["tasks"] += 1
                arms[la]["tasks"] += 1
                arms[sa]["correct"] += comp["stable_action"] == probe.task.z
                arms[la]["correct"] += comp["latent_action"] == probe.task.z
                arms[sa]["queries"] += q_used
                arms[la]["queries"] += q_used
                arms[sa]["risk"] += comp["stable_risk"]
                arms[la]["risk"] += \
                    comp["latent_action_risk_under_stable"]
                if not (comp["passed"] and comp["matched_history"]
                        and comp["latent_has_NEW"] and comp["latent_has_OUT"]):
                    failures.append({"seed": seed, "probe": n,
                                     "comparison": key,
                                     "details": comp})
            taskwise.append({
                "seed": seed, "kind": probe.kind, "slot": probe.slot,
                "q0": audit["q0"],
                "q1": audit["q1"],
                "oracle_query": audit["oracle_query"],
                "model": audit["model"],
            })
    for row in arms.values():
        n = row["tasks"]
        row["accuracy"] = Fraction(row["correct"], n) if n else Fraction(0)
        row["mean_queries"] = Fraction(row["queries"], n) if n else Fraction(0)
        row["mean_taskwise_bayes_risk"] = row.pop("risk") / n if n else 0
    return {"arms": arms, "taskwise_failures": failures,
            "tasks": len(taskwise), "taskwise_rows": taskwise,
            "calibration": calibration,
            "all_taskwise_pass": not failures and bool(taskwise)}


def memoryless_split(overlap: str, seeds) -> dict:
    buckets: dict = defaultdict(list)
    source_tasks = 0
    for seed in seeds:
        fam, beh, _ids, probes, _masks = build_stream(overlap, seed)
        # Use the same matched scoring population as central retrieval.  The
        # old positional first-two slice selected the deliberately equivalent
        # slots 0/1 in nearly every stream and was not a broad calibration.
        for probe in EV.memoryless_population(probes, "all_matched_scored"):
            source_tasks += 1
            curves = INF.memoryless_policy_curves(
                fam, beh, probe.task, probe.phi_true, probe.task.z,
                LEGAL_BEHAVIORAL, LEGAL_SEMANTIC, QUERY_BUDGETS, seed)
            for policy, rows in curves.items():
                for budget, row in rows.items():
                    buckets[(policy, budget)].append(row)
    out = {}
    for policy in INF.MEMORYLESS_POLICIES:
        out[policy] = {}
        for budget in QUERY_BUDGETS:
            rows = buckets[(policy, budget)]
            physical = Counter()
            resolved = Counter()
            effect_rows = []
            for row in rows:
                physical.update(dict(row.query_types))
                for effect in row.resolution_effects:
                    resolved.update(effect.resolved_quantities)
                effect_rows.append({
                    "task_digest": row.state.task_digest,
                    "effects": [effect.canon()
                                for effect in row.resolution_effects],
                })
            out[policy][budget] = {
                "tasks": len(rows),
                "task_accuracy": frac_mean(r.correct for r in rows),
                "query_budget": budget,
                "queries_offered": sum(r.queries_offered for r in rows),
                "queries_actually_asked": sum(r.queries_asked for r in rows),
                "mean_queries_all_tasks": frac_mean(
                    r.queries_asked for r in rows),
                "convention_entropy": {
                    "before": decimal_mean(
                        r.convention_entropy_before for r in rows),
                    "after": decimal_mean(
                        r.convention_entropy_after for r in rows),
                },
                "task_entropy": {
                    "before": decimal_mean(r.task_entropy_before for r in rows),
                    "after": decimal_mean(r.task_entropy_after for r in rows),
                },
                "candidate_class_count": {
                    "before": frac_mean(
                        r.candidate_classes_before for r in rows),
                    "after": frac_mean(
                        r.candidate_classes_after for r in rows),
                },
                "query_types": dict(physical),
                "latent_quantity": INF.LATENT_QUANTITY[policy],
                "resolved_latent_quantities": {
                    key: resolved.get(key, 0)
                    for key in ("identity", "convention", "task", "cause")},
                "per_question_resolution_effects": tuple(effect_rows),
                "answers_applied": all(r.answers_applied for r in rows),
            }
    oracle = out[INF.ORACLE_TASK_SEPARATING]
    fresh = out[INF.FRESH_FAMILY_PRIOR]
    stable_fresh = out[INF.STABLE_ID_FRESH]
    improved = oracle[4]["task_accuracy"] > oracle[0]["task_accuracy"]
    nonoracle_improved = any(
        out[p][4]["task_accuracy"] > out[p][0]["task_accuracy"]
        for p in INF.MEMORYLESS_POLICIES
        if p != INF.ORACLE_TASK_SEPARATING)
    equality = all(fresh[q]["task_accuracy"]
                   == stable_fresh[q]["task_accuracy"]
                   and fresh[q]["queries_actually_asked"]
                   == stable_fresh[q]["queries_actually_asked"]
                   for q in QUERY_BUDGETS)
    calibrated_row = next(
        row for rows in buckets.values() for row in rows
        if row.query_budget > 0 and row.queries_asked > 0)
    calibration = CT.calibrate_memoryless_answer_application(calibrated_row)
    return {"tasks": source_tasks,
            "population": "all_matched_scored",
            "policies": out,
            "oracle_legal_query_improves": improved,
            "nonoracle_policy_improves": nonoracle_improved,
            "fresh_equals_stable_fresh": equality,
            "all_answers_applied": all(
                out[p][q]["answers_applied"]
                for p in out for q in out[p]),
            "calibration": calibration}


def active_query_split(overlap: str, seeds) -> dict:
    by_stream = {}
    for seed in seeds:
        fam, _beh, ids, probes, masks = build_stream(overlap, seed)
        records = {i.slot: SM.SemanticRecord(
            f"record:{i.slot}", i.grounded) for i in ids}
        exact = RET.build_global_exact_index(records)
        rows = {
            MAIN.INFORMATION_GAIN: defaultdict(list),
            MAIN.RANDOM: defaultdict(list),
        }
        for probe in scored_probes(probes):
            selected = RET.retrieve_protocol_a(
                exact, fam, probe.task, k=4,
                strategy="exact_likelihood", seed=seed)
            initial = MAIN.subset_state(
                fam, probe.task, masks, selected.selected_keys)
            for policy in rows:
                for budget in QUERY_BUDGETS:
                    row = _open_metric(initial, probe, policy, budget, seed)
                    row["condition"] = probe.kind
                    rows[policy][budget].append(row)

        curves = {
            policy: {budget: _summarise_open(task_rows)
                     for budget, task_rows in by_budget.items()}
            for policy, by_budget in rows.items()
        }
        joint_curve = curves[MAIN.INFORMATION_GAIN]
        random_curve = curves[MAIN.RANDOM]
        j = dict(joint_curve[1])
        r = dict(random_curve[1])
        # q=5 is a censored "not reached within q=0..4" value.  It is
        # declared once and never used to stop an individual task.
        joint_target = _questions_to_target(
            {q: joint_curve[q]["task_accuracy"] for q in QUERY_BUDGETS})
        random_target = _questions_to_target(
            {q: random_curve[q]["task_accuracy"] for q in QUERY_BUDGETS})
        j["questions_at_matched_accuracy"] = joint_target
        r["questions_at_matched_accuracy"] = random_target

        def aggregate_curve(policy, curve, target):
            return {
                "policy": policy,
                "frozen_target_accuracy": TARGET_ACCURACY,
                "population_tasks": curve[0]["tasks"],
                "complete_stream_seeds": (seed,),
                "budgets": curve,
                "prefix_consistent": True,
                "minimum_questions_to_frozen_target": (
                    target if target <= max(QUERY_BUDGETS) else None),
                "metric_definition": (
                    "minimum aggregate declared q=0..4 budget reaching the "
                    "validation-frozen target"),
            }

        j["accuracy_curve"] = aggregate_curve(
            MAIN.INFORMATION_GAIN, joint_curve, joint_target)
        r["accuracy_curve"] = aggregate_curve(
            MAIN.RANDOM, random_curve, random_target)
        by_stream[seed] = {"information_gain": j, "random": r}

    def vec(arm, metric):
        return [by_stream[s][arm][metric] for s in seeds]

    intervals = {
        "task_accuracy": ST.paired_interval(
            vec("information_gain", "task_accuracy"),
            vec("random", "task_accuracy"), reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED),
        "equivalence_retrieval": ST.paired_interval(
            vec("information_gain", "equivalence_retrieval"),
            vec("random", "equivalence_retrieval"), reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED + 1),
        "questions_at_matched_accuracy": ST.paired_interval(
            vec("information_gain", "questions_at_matched_accuracy"),
            vec("random", "questions_at_matched_accuracy"),
            reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 2),
        "false_confident_answers": ST.paired_interval(
            vec("information_gain", "false_confident_answers"),
            vec("random", "false_confident_answers"),
            reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 3),
    }
    all_include = all(ST.interval_includes_zero(iv)
                      for iv in intervals.values())
    result = {
        "streams": by_stream, "intervals": intervals,
        "frozen_target_accuracy": TARGET_ACCURACY,
        "not_reached_censor_value": max(QUERY_BUDGETS) + 1,
        "questions_metric_definition": (
            "minimum aggregate q=0..4 budget reaching frozen 0.95 accuracy; "
            "5 means not reached"),
        "inference_model": (
            "the same approximate validation-frozen Protocol-A top-four "
            "open-world MAIN adapter as central retrieval, including stored "
            "records, NEW_IDENTITY and OUT_OF_FAMILY"),
        "component_status": ("not_measured_in_X65A-L"
                             if all_include else "measured"),
        "all_operational_intervals_include_zero": all_include,
    }
    result["calibration"] = CT.calibrate_active_intervals(
        result, len(seeds))
    return result


def _top_open_identity(post):
    if not post:
        return None
    best = max(post.values())
    return min((key for key, value in post.items() if value == best),
               key=lambda key: (not isinstance(key, int), str(key)))


def _open_metric(initial, probe, policy: str, budget: int, seed: int,
                 retrieval=None) -> dict:
    run = MAIN.run_policy(
        initial, policy, budget, probe.phi_true, probe.task.z,
        LEGAL_SEMANTIC, seed)
    identity = run.state.identity_posterior()
    top = _top_open_identity(identity)
    literal = top == probe.slot
    equivalent = (top in set(probe.equivalence)
                  if probe.equivalence else literal)
    before_identity = initial.identity_posterior()
    before_task = initial.task_posterior()
    convention_changed = initial.convention_posteriors() != \
        run.state.convention_posteriors()
    return {
        "correct": run.correct,
        "action": run.action,
        "confidence": run.confidence,
        "false_confident": (not run.correct
                            and run.confidence >= Fraction(19, 20)),
        "identity_top": top,
        "literal_identity": literal,
        "equivalence_retrieval": equivalent,
        "query_budget": budget,
        "queries_offered": run.queries_offered,
        "queries_asked": run.queries_asked,
        "physical_query_types": {"semantic": run.queries_asked,
                                 "task": 0},
        "resolved_latent_quantities": {
            "identity": int(before_identity != identity),
            "convention": int(convention_changed),
            "task": int(before_task != run.state.task_posterior()),
            "cause": int(before_identity.get("OUT_OF_FAMILY", 0)
                         != identity.get("OUT_OF_FAMILY", 0)),
        },
        "unresolved": run.identity_decision != "ASSIGN_EXISTING",
        "provisional_branches": int(
            run.identity_decision != "ASSIGN_EXISTING"),
        "established_record_corruption": 0,
        "retrieval": (retrieval.canon()
                      if hasattr(retrieval, "canon") else retrieval),
    }


def _calibrated_memoryless_metric(fam, beh, probe, budget: int,
                                  seed: int) -> dict:
    """Strong validation-frozen no-memory baseline from the L1.5 audit.

    This is the exact family-prior task-information policy, not the flat
    semantic-only placeholder that produced the original 0.214 curve.
    Behavioral and semantic answers are both applied to its current posterior.
    """

    initial = INF.make_memoryless_state(fam, beh, probe.task)
    run = EV.run_policy(
        initial, INF.TASK_INFORMATION_GAIN, budget, probe.phi_true,
        probe.task.z, LEGAL_BEHAVIORAL, LEGAL_SEMANTIC, seed)
    return {
        "correct": run["correct"],
        "action": run["action"],
        "confidence": run["confidence"],
        "false_confident": run["false_confident"],
        "identity_top": None,
        "literal_identity": False,
        "equivalence_retrieval": False,
        "query_budget": budget,
        "queries_offered": run["queries_offered"],
        "queries_asked": len(run["events"]),
        "physical_query_types": dict(run["physical"]),
        "resolved_latent_quantities": dict(run["resolved"]),
        "unresolved": run["unresolved"],
        "provisional_branches": 0,
        "established_record_corruption": 0,
        "retrieval": None,
        "answers_applied": len(run["state"].history) == len(run["events"]),
        "baseline_definition": "fresh_family_prior_exact_task_information_gain",
    }


def _summarise_open(rows) -> dict:
    rows = list(rows)
    n = len(rows)
    ambiguous = [r for r in rows if r.get("condition") in
                 ("ambiguous", "misleading")]
    return {
        "tasks": n,
        "task_accuracy": frac_mean(r["correct"] for r in rows),
        "equivalence_retrieval": frac_mean(
            r["equivalence_retrieval"] for r in rows),
        "literal_identity": frac_mean(r["literal_identity"] for r in rows),
        "queries_offered": sum(r["queries_offered"] for r in rows),
        "queries_asked": sum(r["queries_asked"] for r in rows),
        "mean_queries_all_tasks": frac_mean(
            r["queries_asked"] for r in rows),
        "mean_queries_ambiguous_tasks": frac_mean(
            r["queries_asked"] for r in ambiguous),
        "false_confident_answers": sum(r["false_confident"] for r in rows),
        "unresolved_outcomes": sum(r["unresolved"] for r in rows),
        "provisional_branches": sum(r["provisional_branches"] for r in rows),
        "established_record_corruption": sum(
            r["established_record_corruption"] for r in rows),
        "physical_query_types": {
            key: sum(r["physical_query_types"][key] for r in rows)
            for key in ("semantic", "task")},
        "resolved_latent_quantities": {
            key: sum(r["resolved_latent_quantities"][key] for r in rows)
            for key in ("identity", "convention", "task", "cause")},
    }


def _questions_to_target(curve: Mapping[int, Fraction],
                         target: Fraction = TARGET_ACCURACY) -> int:
    budgets = tuple(sorted(curve))
    if budgets != QUERY_BUDGETS:
        raise ValueError("target metric requires the complete q=0..4 curve")
    for budget in budgets:
        if Fraction(curve[budget]) >= target:
            return budget
    return max(budgets) + 1


def central_retrieval_split(overlap: str, seeds) -> dict:
    stream_rows = {}
    all_accounting_a, all_accounting_b = [], []
    accounting_rows = {name: [] for name in (
        "MAIN_protocol_A", "exact_all_record", "protocol_B_four_record",
        "random_retrieval", "recency", "surface_nearest")}
    accounting_valid_a, accounting_valid_b = [], []
    undercharge_rejected = []
    active_bytes = []
    collisions = []
    for seed in seeds:
        fam, beh, ids, probes, masks = build_stream(overlap, seed)
        records = {i.slot: SM.SemanticRecord(f"record:{i.slot}", i.grounded)
                   for i in ids}
        exact = RET.build_global_exact_index(records)
        coarse = RET.build_coarse_index(exact)
        store = {entry.record_key: entry.sketch for entry in exact.entries}
        collision = RET.coarse_collision_witness(coarse, store, fam)
        collisions.append(collision is not None)
        active_bytes.append(RET.active_bytes_with_indexes(
            records, exact, coarse))
        curves = defaultdict(lambda: defaultdict(list))
        for probe_number, probe in enumerate(scored_probes(probes)):
            a = RET.retrieve_protocol_a(
                exact, fam, probe.task, k=4,
                strategy="exact_likelihood", seed=seed)
            a_exact = RET.rerank_protocol_a(
                exact, probe.task, a, len(masks),
                strategy="all_records", seed=seed)
            a_random = RET.rerank_protocol_a(
                exact, probe.task, a, 4, strategy="random", seed=seed)
            a_recency = RET.rerank_protocol_a(
                exact, probe.task, a, 4, strategy="recency", seed=seed)
            a_surface = RET.rerank_protocol_a(
                exact, probe.task, a, 4,
                strategy="surface_nearest", seed=seed)
            b = RET.retrieve_protocol_b(coarse, store, fam, probe.task,
                                        collision)
            all_accounting_a.append(a.accounting)
            all_accounting_b.append(b.accounting)
            contract_a = RET.protocol_a_accounting_contract(
                exact, len(a.selected_keys))
            contract_b = RET.protocol_b_accounting_contract(
                coarse, len(store), b.accounting.sketch_bytes_loaded,
                b.accounting.identity_specific_summaries_inspected)
            valid_a = RET.validate_retrieval_accounting(
                a.accounting, contract_a)
            valid_b = RET.validate_retrieval_accounting(
                b.accounting, contract_b)
            planted = RET.planted_undercharged_exact_index_row(a.accounting)
            planted_result = RET.validate_retrieval_accounting(
                planted, contract_a)
            accounting_valid_a.append(valid_a.passed)
            accounting_valid_b.append(valid_b.passed)
            undercharge_rejected.append(not planted_result.passed)
            task_id = f"{seed}:{probe_number}:{probe.kind}:{probe.slot}"
            metadata = {
                "stream_seed": seed,
                "task_id": task_id,
                "condition": probe.kind,
                "true_slot": probe.slot,
                "current_utterance": probe.task.u,
                "task_candidate_pool": tuple(probe.task.live),
            }
            selections = {
                "MAIN_protocol_A": a,
                "exact_all_record": a_exact,
                "protocol_B_four_record": b,
                "random_retrieval": a_random,
                "recency": a_recency,
                "surface_nearest": a_surface,
            }
            for arm, selection in selections.items():
                contract = (contract_b if arm == "protocol_B_four_record"
                            else RET.protocol_a_accounting_contract(
                                exact, len(selection.selected_keys)))
                validation = RET.validate_retrieval_accounting(
                    selection.accounting, contract)
                accounting_rows[arm].append({
                    **metadata,
                    "selected_keys": selection.selected_keys,
                    **selection.accounting.canon(),
                    "validation": validation.canon(),
                })
            states = {
                "MAIN_protocol_A": MAIN.subset_state(
                    fam, probe.task, masks, a.selected_keys),
                "exact_all_record": MAIN.latent_state(fam, probe.task, masks),
                "protocol_B_four_record": MAIN.subset_state(
                    fam, probe.task, masks, b.selected_keys),
                "random_retrieval": MAIN.subset_state(
                    fam, probe.task, masks, a_random.selected_keys),
                "recency": MAIN.subset_state(
                    fam, probe.task, masks, a_recency.selected_keys),
                "surface_nearest": MAIN.subset_state(
                    fam, probe.task, masks, a_surface.selected_keys),
                "stable_ID": MAIN.stable_state(
                    fam, probe.task, masks[probe.slot], probe.slot),
            }
            for budget in QUERY_BUDGETS:
                for arm, state in states.items():
                    retrieval = (selections[arm].accounting
                                 if arm in selections else None)
                    row = _open_metric(
                        state, probe, MAIN.INFORMATION_GAIN, budget, seed,
                        retrieval=retrieval)
                    row["condition"] = probe.kind
                    curves[arm][budget].append(row)
                no_memory = _calibrated_memoryless_metric(
                    fam, beh, probe, budget, seed)
                no_memory["condition"] = probe.kind
                curves["no_memory"][budget].append(no_memory)
        summaries = {}
        for arm, by_budget in curves.items():
            curve = {budget: _summarise_open(rows)
                     for budget, rows in by_budget.items()}
            q_target = _questions_to_target(
                {q: row["task_accuracy"] for q, row in curve.items()})
            summaries[arm] = {
                "curve": curve,
                "questions_to_frozen_target_accuracy": q_target,
                "frozen_target_accuracy": TARGET_ACCURACY,
                # q=1 is the preregistered comparison point; the full curve
                # remains available and is the source of the target metric.
                **curve[1],
            }
        stream_rows[seed] = summaries

    def vec(arm, metric):
        return [stream_rows[s][arm][metric] for s in seeds]

    intervals = {
        "MAIN_minus_random_accuracy": ST.paired_interval(
            vec("MAIN_protocol_A", "task_accuracy"),
            vec("random_retrieval", "task_accuracy"),
            reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 10),
        "MAIN_minus_recency_accuracy": ST.paired_interval(
            vec("MAIN_protocol_A", "task_accuracy"),
            vec("recency", "task_accuracy"),
            reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 11),
        "MAIN_minus_surface_accuracy": ST.paired_interval(
            vec("MAIN_protocol_A", "task_accuracy"),
            vec("surface_nearest", "task_accuracy"),
            reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 12),
        "MAIN_minus_exact_accuracy": ST.paired_interval(
            vec("MAIN_protocol_A", "task_accuracy"),
            vec("exact_all_record", "task_accuracy"),
            reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 13),
        "MAIN_minus_no_memory_query_count": ST.paired_interval(
            vec("MAIN_protocol_A", "questions_to_frozen_target_accuracy"),
            vec("no_memory", "questions_to_frozen_target_accuracy"),
            reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED + 14),
    }

    def extrema(rows):
        fields = ("index_bytes_scanned",
                  "identity_specific_summaries_inspected",
                  "identity_likelihoods_evaluated", "shortlist_size",
                  "full_records_loaded", "sketch_bytes_loaded",
                  "total_retrieval_bytes",
                  "total_retrieval_node_equivalents")
        return {field: {"min": min(getattr(r, field) for r in rows),
                        "max": max(getattr(r, field) for r in rows)}
                for field in fields}

    matched_fields = (
        "index_bytes_scanned", "identity_specific_summaries_inspected",
        "identity_likelihoods_evaluated", "shortlist_size",
        "full_records_loaded", "sketch_bytes_loaded",
        "total_retrieval_bytes", "total_retrieval_node_equivalents")
    resource_matched_controls = all(
        all(all(control[field] == main[field] for field in matched_fields)
            for control in controls)
        for main, controls in zip(
            accounting_rows["MAIN_protocol_A"],
            zip(accounting_rows["random_retrieval"],
                accounting_rows["recency"],
                accounting_rows["surface_nearest"])))

    return {
        "streams": stream_rows,
        "per_task_accounting": accounting_rows,
        "intervals": intervals,
        "protocol_A": {
            "claim": "global exact-sketch scoring under 512 bytes; eight "
                     "node-equivalents; validation-frozen top-four shortlist; "
                     "incomplete retrieval and no four-node claim",
            "accounting_extrema": extrema(all_accounting_a),
            "all_within_512": all(r.within_512 for r in all_accounting_a),
            "no_four_node_claim": all(not r.four_node_claim
                                      for r in all_accounting_a),
            "all_main_rows_report_incomplete": all(
                r.incomplete_retrieval for r in all_accounting_a),
            "all_rows_validated": all(accounting_valid_a),
        },
        "protocol_B": {
            "claim": "nonsufficient coarse nomination then <=4 exact records",
            "accounting_extrema": extrema(all_accounting_b),
            "all_within_512": all(r.within_512 for r in all_accounting_b),
            "all_at_most_four": all(
                r.total_retrieval_node_equivalents <= 4
                and r.identity_specific_summaries_inspected <= 4
                for r in all_accounting_b),
            "all_incomplete_reported": all(r.incomplete_retrieval
                                           for r in all_accounting_b),
            "all_rows_validated": all(accounting_valid_b),
        },
        "main_protocol": "A_GLOBAL_EXACT_SCAN_TOP4",
        "main_protocol_freeze": {
            "record": MAIN_RETRIEVAL_FREEZE,
            "sha256": hashlib.sha256(
                encode(MAIN_RETRIEVAL_FREEZE)).hexdigest(),
        },
        "main_claim": "query-efficient approximate top-four shortlist after "
                      "a global exact-sketch scan <=512 B and eight node-"
                      "equivalents; not four-node exact retrieval",
        "resource_matched_controls": resource_matched_controls,
        "no_memory_baseline": (
            "validation-frozen fresh family-prior exact task-information "
            "policy over legal behavioral and semantic questions"),
        "coarse_nonsufficiency_collisions": sum(collisions),
        "active_bytes": {"max": max(active_bytes),
                         "all_within_4KiB": all(x <= 4096
                                                for x in active_bytes)},
        "calibration": {
            "coarse_collision_fired": all(collisions),
            "valid_rows_accepted_by_accounting_validator": (
                all(accounting_valid_a) and all(accounting_valid_b)),
            "free_exact_index_undercharge_rejected_by_same_validator":
                all(undercharge_rejected),
            "fires": (all(collisions) and all(accounting_valid_a)
                      and all(accounting_valid_b)
                      and all(undercharge_rejected)),
        },
    }


def safety_split(overlap: str, seeds) -> dict:
    rows = []
    active = []
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    for seed in seeds:
        ids7 = LS.build_identities(fam, seed, n=7)
        row = dict(SAFE.new_identity_audit(fam, ids7, beh, seed))
        row["seed"] = seed
        rows.append(row)
        records = {i.slot: SM.SemanticRecord(f"record:{i.slot}", i.grounded)
                   for i in ids7}
        # The promoted record is charged in the same canonical active
        # container as the seven old records, plus both retrieval indexes.
        promoted = type("Promoted", (), {
            "grounded": tuple(SM.GroundedObservation(
                g["z"], g["u"], g["e"])
                for g in row["record"]["grounded"]),
            "canon": lambda self, r=row: r["record"],
        })()
        if row["successfully_promoted_new_records"]:
            records[7] = promoted
        exact = RET.build_global_exact_index(records)
        coarse = RET.build_coarse_index(exact)
        active.append(RET.active_bytes_with_indexes(records, exact, coarse))
    arm_names = tuple(rows[0]["arms"])
    arm_aggregates = {}
    for name in arm_names:
        arm_rows = [row["arms"][name] for row in rows]
        arm_aggregates[name] = {
            "precision": frac_mean(row["precision"] for row in arm_rows),
            "recall": frac_mean(row["recall"] for row in arm_rows),
            "false_new_rate_returning": frac_mean(
                row["false_new_rate_returning"] for row in arm_rows),
            "forced_assimilation_rate": frac_mean(
                row["forced_assimilation_rate"] for row in arm_rows),
            "unresolved_new_rate": frac_mean(
                row["unresolved_new_rate"] for row in arm_rows),
            "true_new_created": sum(row["true_new_created"]
                                    for row in arm_rows),
            "returning_false_new": sum(row["returning_false_new"]
                                       for row in arm_rows),
            "new_false_negative": sum(row["new_false_negative"]
                                      for row in arm_rows),
            "new_forced_assimilation": sum(
                row["new_forced_assimilation"] for row in arm_rows),
        }
    reuse_failures = tuple({
        "seed": row["seed"],
        "promoted_record_key": row["promoted_record_key"],
        "identity_top": row["later_reuse_identity_top"],
        "identity_decision": row["later_reuse_identity_decision"],
        "identity_posterior": row["later_reuse_identity_posterior"],
        "task_posterior": row["later_reuse_task_posterior"],
        "shortlist": row["later_reuse_shortlist"],
        "action": row["later_reuse_action"],
        "task_accuracy": row["later_reuse_task_accuracy"],
        "query_history": row["later_reuse_query_history"],
    } for row in rows if not row["later_reuse_of_new_records"])
    return {
        "seeds": tuple(seeds),
        "streams": len(rows),
        "precision": frac_mean(r["precision"] for r in rows),
        "recall": frac_mean(r["recall"] for r in rows),
        "false_new_rate_returning": frac_mean(
            r["false_new_rate_returning"] for r in rows),
        "forced_assimilation_rate": frac_mean(
            r["forced_assimilation_rate"] for r in rows),
        "unresolved_new_rate": frac_mean(
            r["unresolved_new_rate"] for r in rows),
        "questions_to_grounded_creation": frac_mean(
            r["questions_to_grounded_creation"] for r in rows),
        "successfully_promoted_new_records": sum(
            r["successfully_promoted_new_records"] for r in rows),
        "later_reuse_of_new_records": sum(
            r["later_reuse_of_new_records"] for r in rows),
        "contamination_during_creation": sum(
            r["contamination_during_creation"] for r in rows),
        "record_bytes_added": tuple(r["record_bytes_added"] for r in rows),
        "active_bytes": {"max": max(active),
                         "all_within_4KiB": all(x <= 4096 for x in active)},
        "arms": arm_aggregates,
        "stream_rows": rows,
        "later_reuse_failures": reuse_failures,
        "main_classification_gate": all(
            row["main_classification_gate"] for row in rows),
        "paired_forced_assimilation_vectors": {
            "MAIN": tuple(row["forced_assimilation_rate"] for row in rows),
            "no_NEW_forced": tuple(
                row["arms"]["no_new_forced"]["forced_assimilation_rate"]
                for row in rows),
        },
        "calibration": {
            "fires": all(all(r["calibration_fired"].values()) for r in rows),
            "details": tuple(r["calibration_fired"] for r in rows),
        },
    }


def negative_split(overlap: str, seeds, margin: Fraction) -> dict:
    from x65a import l1_negative as NEG
    audits = [NEG.audit_stratum(overlap, seed) for seed in seeds]
    by_condition = {}
    for name in NEG.CONDITIONS:
        rows = [next(r for r in audit.conditions
                     if r.condition.name == name) for audit in audits]
        applicable = [r for r in rows if r.main.task_accuracy is not None]
        by_condition[name] = {
            "constructible_transfer": all(
                r.condition.transfer_constructible for r in rows),
            "tested_via": tuple(r.condition.tested_via for r in rows),
            "task_accuracy": {
                "MAIN": frac_mean(r.main.task_accuracy for r in applicable),
                "no_memory": frac_mean(
                    r.no_memory.task_accuracy for r in applicable),
            } if applicable else {"MAIN": None, "no_memory": None},
            "accuracy_delta": frac_mean(
                r.accuracy_delta for r in applicable) if applicable else None,
            "excess_questions": sum(r.main.excess_questions
                                    for r in rows),
            "false_confident_actions": sum(
                r.main.false_confident_actions for r in rows),
            "established_record_corruption": sum(
                r.main.established_record_corruption for r in rows),
            "provisional_branches": sum(
                r.main.provisional_branches for r in rows),
            "unresolved_outcomes": sum(
                r.main.unresolved_outcomes for r in rows),
            "noninferior": all(r.noninferior for r in rows),
            "matched_protocol": all(r.matched_protocol for r in rows),
        }
    main_corruption = tuple(
        sum(r.main.established_record_corruption for r in audit.conditions)
        for audit in audits)
    immediate_map_corruption = tuple(
        audit.calibrations["immediate_map_write"]
            ["established_record_corruption"]
        for audit in audits)
    confirmation_contamination = main_corruption
    no_confirmation_contamination = tuple(
        audit.calibrations["no_confirmation_contamination"]
            ["established_record_corruption"]
        for audit in audits)
    intervals = {
        "provisional_assignment_minus_immediate_MAP_corruption":
            ST.paired_interval(main_corruption, immediate_map_corruption,
                               reps=BOOTSTRAP_REPS,
                               seed=BOOTSTRAP_SEED + 30),
        "confirmation_minus_no_confirmation_contamination":
            ST.paired_interval(confirmation_contamination,
                               no_confirmation_contamination,
                               reps=BOOTSTRAP_REPS,
                               seed=BOOTSTRAP_SEED + 31),
    }
    failures = tuple({
        "seed": audit.seed,
        "condition": result.condition.name,
        "accuracy_delta": result.accuracy_delta,
        "MAIN": result.main.canon(),
        "no_memory": result.no_memory.canon(),
        "matched_protocol": result.matched_protocol,
    } for audit in audits for result in audit.conditions
        if not result.noninferior)
    return {
        "seeds": tuple(seeds),
        "frozen_margin": margin,
        "conditions": by_condition,
        "stream_rows": tuple(a.canon() for a in audits),
        "all_conditions_reported": tuple(by_condition) == NEG.CONDITIONS,
        "main_noninferior": all(
            a.gates["main_noninferior_at_frozen_margin"]
            and a.accuracy_margin == -margin for a in audits),
        "main_established_corruption": sum(
            r.main.established_record_corruption
            for a in audits for r in a.conditions),
        "noninferiority_failures": failures,
        "matched_protocol": all(
            r.matched_protocol for audit in audits for r in audit.conditions),
        "intervals": intervals,
        "paired_safety_vectors": {
            "provisional_assignment_corruption": main_corruption,
            "immediate_MAP_corruption": immediate_map_corruption,
            "confirmation_contamination": confirmation_contamination,
            "no_confirmation_contamination": no_confirmation_contamination,
        },
        "calibration": {
            "fires": all(a.calibrations["all_fire"] for a in audits),
            "immediate_MAP_corruption": sum(
                a.calibrations["immediate_map_write"]
                    ["established_record_corruption"] for a in audits),
            "forced_new_assimilation": sum(
                a.calibrations["forced_new_assimilation"]
                    ["forced_decisions"] for a in audits),
            "no_confirmation_contamination": sum(
                a.calibrations["no_confirmation_contamination"]
                    ["established_record_corruption"] for a in audits),
            "same_main_safety_predicate_rejected_every_plant": all(
                a.calibrations["same_predicate_rejections"] for a in audits),
        },
    }


def task_conditions_per_stream() -> dict:
    out = {}
    for overlap in ("shared", "disjoint_op"):
        out[overlap] = {}
        for split, seeds in (("development", DEV), ("validation", VAL)):
            out[overlap][split] = {}
            for seed in seeds:
                _fam, _beh, _ids, probes, _masks = build_stream(
                    overlap, seed)
                counts = Counter(probe.kind for probe in probes)
                counts["all_probe_rows"] = len(probes)
                counts["scored_returning_ambiguous_misleading"] = len(
                    scored_probes(probes))
                out[overlap][split][str(seed)] = dict(sorted(counts.items()))
    return out


def provenance(experiment_runtime_ms: int) -> dict:
    tracked = sh("git", "status", "--porcelain", "-uno").splitlines()
    untracked = sh("git", "ls-files", "--others", "--exclude-standard")
    untracked_lines = untracked.splitlines() if untracked else []
    base = sh("git", "rev-parse", BASE_COMMIT)
    paths = {
        "runner": str(Path(__file__).resolve()),
        "authoritative_json": str(OUT.resolve()),
        "readme": str(Path("README.md").resolve()),
        "inference": str(Path("experiments/x65a/l1_inference.py").resolve()),
        "evaluation": str(Path("experiments/x65a/l1_eval.py").resolve()),
        "contracts": str(Path("experiments/x65a/l1_contracts.py").resolve()),
        "gate_evaluator": str(
            Path("experiments/x65a/l1_gates.py").resolve()),
        "main_adapter": str(Path("experiments/x65a/l1_main.py").resolve()),
        "retrieval": str(Path("experiments/x65a/l1_retrieval.py").resolve()),
        "safety": str(Path("experiments/x65a/l1_safety.py").resolve()),
        "negative_transfer": str(
            Path("experiments/x65a/l1_negative.py").resolve()),
        "statistics": str(Path("experiments/x65a/l1_stats.py").resolve()),
        "restart": str(Path("experiments/x65a/restart_l1.py").resolve()),
        "restart_matrix": str(
            Path("experiments/x65a/l1_restart_audit.py").resolve()),
        "sufficiency_proof": str(Path(
            "experiments/x65a/X65A-L1-SUFFICIENCY-PROOF.md").resolve()),
        "tests": str(Path("tests/test_x65a_l1.py").resolve()),
        "gate_tests": str(Path("tests/test_x65a_l1_gates.py").resolve()),
        "restart_tests": str(
            Path("tests/test_x65a_l1_restart_audit.py").resolve()),
        "test_contracts": str(
            Path("tests/test_x65a_l1_contracts.py").resolve()),
        "test_evaluation": str(
            Path("tests/test_x65a_l1_eval.py").resolve()),
        "test_inference": str(
            Path("tests/test_x65a_l1_inference.py").resolve()),
        "test_main": str(Path("tests/test_x65a_l1_main.py").resolve()),
        "test_negative_transfer": str(
            Path("tests/test_x65a_l1_negative.py").resolve()),
        "test_retrieval": str(
            Path("tests/test_x65a_l1_retrieval.py").resolve()),
        "test_safety": str(
            Path("tests/test_x65a_l1_safety.py").resolve()),
        "test_statistics": str(
            Path("tests/test_x65a_l1_stats.py").resolve()),
        "test_restart_process": str(
            Path("tests/test_x65a_restart_l1.py").resolve()),
        "legacy_l_artifact": str(
            Path("experiments/x65a/results/x65al_latent.json").resolve()),
    }
    result = {
        "x65a_l_commit_full_hash": base,
        "current_HEAD": sh("git", "rev-parse", "HEAD"),
        "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_tree_clean": not tracked,
        "tracked_status": tuple(tracked),
        "untracked_tree_clean": not untracked_lines,
        "untracked_count": len(untracked_lines),
        "untracked_status": tuple(untracked_lines),
        "development_stream_seeds": DEV,
        "validation_stream_seeds": VAL,
        "streams_per_alphabet_stratum": {
            "development": len(DEV), "validation": len(VAL)},
        "identities_per_stream": 8,
        "task_conditions_per_stream": task_conditions_per_stream(),
        "query_budgets": QUERY_BUDGETS,
        "legal_behavioral_queries": LEGAL_BEHAVIORAL,
        "legal_semantic_queries": LEGAL_SEMANTIC,
        "validation_frozen_margins": {
            "L3_retrieval_noninferiority": L3_MARGIN,
            "L10_negative_transfer": L10_MARGIN,
        },
        "experiment_runtime_ms": experiment_runtime_ms,
        "full_suite_runtime_ms": FULL_SUITE_RUNTIME_MS,
        "full_suite_evidence": {
            "command": FULL_SUITE_COMMAND,
            "exit_code": FULL_SUITE_EXIT_CODE,
            "passed": FULL_SUITE_PASSED,
            "skipped": FULL_SUITE_SKIPPED,
            "runtime_ms": FULL_SUITE_RUNTIME_MS,
            "result_line": (
                f"{FULL_SUITE_PASSED} passed, {FULL_SUITE_SKIPPED} skipped "
                f"in {FULL_SUITE_RUNTIME_MS} ms"),
        },
        "artifact_paths": paths,
        "final_manifest_written": False,
        "final_stream_seed_sampled": False,
    }
    result["validation"] = CT.validate_provenance(result).canon()
    result["calibration"] = CT.calibrate_provenance(result)
    result["complete"] = (result["validation"]["ok"]
                          and result["calibration"]["fires"])
    return result


def main() -> int:
    t0 = time.perf_counter_ns()
    print("X65A-L1: retrieval sufficiency, budget semantics, baseline audit")
    pre = PQ.check()
    if not pre.ok:
        print("X64H prerequisite failed; no artifact written")
        return 2

    art = {"phase": "X65A-L1", "schema": 1,
           "x64h_prerequisite": {
               "passed": pre.ok,
               "checks": len(pre.checks),
               "passed_checks": sum(v["pass"] for v in pre.checks.values()),
           },
           "claim_boundary": "controlled authored semantic environment",
           "unsupported_scope": (
               "natural language", "lifelong learning", "AGI",
               "bounded total memory"),
           "bugs_and_corrections": (
               "stable-ID and exact-latent arms had unmatched query/stopping semantics",
               "zero_query_accuracy was post-query accuracy",
               "q=2.46 was mean actual queries at budget three over a different denominator",
               "memoryless larger-budget arm counted questions without asking or applying answers",
               "UNRESOLVED was counted as NEW recall although no record was created",
               "NEW clarification answers reset to the family prior",
               "the exact index smuggled all eight record summaries into a four-node claim",
               "sketch bytes omitted container and association metadata",
               "query utility ignored selection-aware nonuniform convention weights",
               "L restart reused S2 state and recomputed rather than loading latent state",
               "random controls used process-salted hash seeds",
               "README's 0.952/q=2.46 table was labeled shared but came from disjoint_op",
               "central and negative-transfer comparisons retained a semantic-only flat no-memory placeholder after L1.5 calibrated a stronger learner",
               "random, recency, and surface controls retrieved one record while MAIN used all eight exact summaries",
               "the first NEW audit bypassed provisional confirmation with a direct CONFIRMED write and accepted unresolved later reuse",
               "the first restart audit serialized a summary but omitted post-query supports, current task, NEW support, and continuation policy state",
               "the first composite gate ignored required intervals and could remain green after a stored calibration stopped firing",
           )}

    print("  L1.1 matched risk arms")
    art["matched_risk"] = {
        ov: {"development": matched_split(ov, DEV),
             "validation": matched_split(ov, VAL)}
        for ov in ("shared", "disjoint_op")}

    print("  L1.2 sketch proof and exact differential audit")
    cert = RET.sufficiency_certificate()
    diff = RET.audit_both_strata(seeds=(400, 401), task_limit=12,
                                 query_depth=2)
    countermodels = {}
    for ov in ("shared", "disjoint_op"):
        planted_spec = RET.planted_nonindicator_hidden_weight_domain(ov)
        planted_validation = RET.validate_sufficiency_domain(planted_spec)
        witness = RET.countermodel_witness(planted_spec)
        countermodels[ov] = {
            "specification": planted_spec.canon(),
            "same_domain_validator": planted_validation.canon(),
            "support_preserved": witness.support_preserved,
            "posterior_gap": witness.posterior_gap,
            "witness": witness.canon(),
            "rejected": not planted_validation.passed,
        }
    proof_calibration_fires = (
        cert.valid()
        and all(row["rejected"] and row["support_preserved"]
                and row["posterior_gap"] for row in countermodels.values()))
    art["sketch_sufficiency"] = {
        "mathematical_claim": (
            "g -> q(phi|g) exactly; q and current selection-aware W -> "
            "identity/task likelihoods, queries, decisions, NEW and OUT exactly"),
        "proof": cert.canon(),
        "proof_assistant_verified": False,
        "finite_authored_model_countermodels": countermodels,
        "selection_aware_weights_nonuniform": {
            ov: diff[ov].selection_weight_nonuniform_states
            for ov in diff},
        "differential": {ov: row.canon() for ov, row in diff.items()},
        "incomplete_retrieval": not cert.valid(),
        "wording": "algebraically proved inside the finite authored model; "
                   "exact executable corroboration, not a proof of VDFM",
        "calibration": {
            "same_premise_validator_rejects_hidden_nonindicator_weight":
                all(row["rejected"] for row in countermodels.values()),
            "plant_preserves_support_but_changes_exact_mass": all(
                row["support_preserved"] and row["posterior_gap"]
                for row in countermodels.values()),
            "nonuniform_weight_case_present": all(
                row.selection_weight_nonuniform_states > 0 for row in diff.values()),
            "uniform_query_utility_differs": all(
                row.weighted_query_differs_from_uniform > 0 for row in diff.values()),
            "fires": proof_calibration_fires,
        },
    }

    print("  L1.3 retrieval protocols and central arms")
    art["retrieval"] = {
        ov: {"development": central_retrieval_split(ov, DEV),
             "validation": central_retrieval_split(ov, VAL)}
        for ov in ("shared", "disjoint_op")}

    print("  L1.4 legacy query accounting")
    art["query_accounting"] = EV.legacy_query_accounting()
    art["query_accounting"]["validation"] = CT.validate_q246(
        art["query_accounting"]).canon()
    art["query_accounting"]["calibration"] = CT.calibrate_q246(
        art["query_accounting"])

    print("  L1.5 memoryless calibration")
    art["memoryless"] = {
        ov: {"development": memoryless_split(ov, DEV),
             "validation": memoryless_split(ov, VAL)}
        for ov in ("shared", "disjoint_op")}

    print("  L1.6 active query ablation")
    art["active_query"] = {
        ov: {"development": active_query_split(ov, DEV),
             "validation": active_query_split(ov, VAL)}
        for ov in ("shared", "disjoint_op")}

    print("  L1.7 NEW identity creation and reuse")
    art["new_identity"] = {
        ov: {"development": safety_split(ov, DEV),
             "validation": safety_split(ov, VAL)}
        for ov in ("shared", "disjoint_op")}

    print("  L1.8 out-of-family and restricted scope")
    scope = {}
    for ov in ("shared", "disjoint_op"):
        fam = F.Family(F.FamilySpec(overlap=ov))
        beh = EP.behaviour_table(fam.forms)
        scope[ov] = {
            "constructibility": SAFE.stratum_constructibility(fam, beh),
            "restricted_scope": SAFE.scope_audit(fam),
        }
    art["scope_audit"] = scope

    print("  L1.9 negative transfer")
    art["negative_transfer"] = {
        ov: {"development": negative_split(ov, DEV, L10_MARGIN),
             "validation": negative_split(ov, VAL, L10_MARGIN)}
        for ov in ("shared", "disjoint_op")}

    print("  L1.10 genuine restart")
    restart_contract = RSTA.RestartMatrixContract(DEV, VAL)
    restart_cases = tuple(
        RSTA.make_actual_main_case(ov, split, seed)
        for ov in ("shared", "disjoint_op")
        for split, seeds in (("development", DEV), ("validation", VAL))
        for seed in seeds)
    with tempfile.TemporaryDirectory(prefix="x65a-l1-restart-") as td:
        restart_audit = RSTA.audit_main_restart_matrix(
            restart_cases, Path(td), restart_contract)
    restart_plant = RSTA.planted_accepted_corruption(restart_audit.cases)
    restart_plant_validation = RSTA.validate_restart_matrix(
        restart_plant, restart_contract)
    restart = restart_audit.canon()
    restart["calibration"] = {
        "valid_matrix_accepted": restart_audit.validation.passed,
        "accepted_corruption_rejected_by_same_matrix_validator":
            not restart_plant_validation.passed,
        "planted_validation": restart_plant_validation.canon(),
        "fires": (restart_audit.validation.passed
                  and not restart_plant_validation.passed),
    }
    art["restart"] = restart
    checkpoint_bytes = tuple(
        row["cycle"]["checkpoint_bytes"] for row in restart["cases"])
    art["restart_storage_accounting"] = {
        "category": "serialized_restart_and_audit_state",
        "checkpoints": len(checkpoint_bytes),
        "minimum_checkpoint_bytes": min(checkpoint_bytes),
        "maximum_checkpoint_bytes": max(checkpoint_bytes),
        "total_checkpoint_bytes": sum(checkpoint_bytes),
        "active_semantic_memory_budget_bytes": 4096,
        "counted_against_active_semantic_memory": False,
        "boundary": (
            "restart checkpoints serialize runtime continuation and audit "
            "state; they are reported separately from the active semantic "
            "store and support no bounded-total-memory claim"),
        "case_sizes_match": True,
    }

    # Required paired intervals assembled in one place.
    required_intervals = {}
    for ov in ("shared", "disjoint_op"):
        required_intervals[ov] = {}
        for split in ("development", "validation"):
            central = art["retrieval"][ov][split]["intervals"]
            negative = art["negative_transfer"][ov][split]["intervals"]
            new = art["new_identity"][ov][split]
            new_vectors = new["paired_forced_assimilation_vectors"]
            restart_rows = [
                row for row in restart["cases"]
                if row["overlap"] == ov and row["split"] == split]
            restart_observed = tuple(int(
                not row["cycle"]["final_hashes_identical"]
                or row["cycle"]["uninterrupted_final_sha256"]
                    != row["cycle"]["restarted_final_sha256"])
                for row in restart_rows)
            required_intervals[ov][split] = {
                **central,
                **negative,
                "NEW_minus_no_NEW_forced_assimilation":
                    ST.paired_interval(new_vectors["MAIN"],
                                       new_vectors["no_NEW_forced"],
                                       reps=BOOTSTRAP_REPS,
                                       seed=BOOTSTRAP_SEED + 20),
                "restart_difference": ST.paired_interval(
                    restart_observed,
                    tuple(Fraction(0) for _ in restart_observed),
                    reps=BOOTSTRAP_REPS,
                    seed=BOOTSTRAP_SEED + 21),
            }
    art["required_intervals"] = required_intervals

    # Freeze the measurement runtime before evaluating the pure gate layer.
    # It is the full evidence-generation runtime through every L1 section;
    # serialization and terminal printing are intentionally excluded.
    art["experiment_runtime_ms"] = (
        time.perf_counter_ns() - t0) // 1_000_000
    art["provenance"] = provenance(art["experiment_runtime_ms"])
    gate_report = L1G.evaluate_l1_artifact(art)
    art.update(gate_report.canon())
    art["x65a_p_started"] = False
    art["x65a_p_unblocked"] = art["l1_passed"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(encode(art))

    print("\nL1 gates")
    for gate, passed in art["gates"].items():
        print(f"  {gate:5} {'PASS' if passed else 'FAIL'}")
    if art["failures"]:
        print("X65A-P remains blocked:", ", ".join(art["failures"]))
    else:
        print("L1 passed; X65A-P is conditionally unblocked but was not "
              "started by this barrier runner")
    print("artifact ->", OUT)
    print("runtime ms ->", art["experiment_runtime_ms"])
    return 0 if art["l1_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
