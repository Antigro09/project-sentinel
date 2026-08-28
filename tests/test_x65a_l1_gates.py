"""Mutation tests for the pure X65A-L1 gate evaluator."""

from __future__ import annotations

import copy
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, "experiments")

from x65a import l1_gates as G
from x65a import restart_l1 as R


DEV = (6400, 6401, 6402, 6403)
VAL = (7400, 7401, 7402, 7403)
F = Fraction


def _interval(delta=F(0), lo=None, hi=None, clusters=2, seed=1):
    return {
        "lo": delta if lo is None else lo,
        "delta": delta,
        "hi": delta if hi is None else hi,
        "unit": "complete_stream_or_latent_identity",
        "clusters": clusters,
        "resamples": 3000,
        "seed": seed,
    }


def _paths():
    root = Path.cwd().resolve()
    names = ("runner", "authoritative_json", "readme", "inference",
             "evaluation", "contracts", "retrieval", "safety",
             "negative_transfer", "statistics", "restart")
    return {name: str(root / f"{name}.json") for name in names}


def _provenance():
    conditions = {
        overlap: {
            "development": {seed: {"returning": 2} for seed in DEV},
            "validation": {seed: {"returning": 2} for seed in VAL},
        }
        for overlap in G.OVERLAPS
    }
    return {
        "x65a_l_commit_full_hash":
            "5205543b110ba6da2e3f6da30630809941f821c4",
        "current_HEAD": "a" * 40,
        "branch": "phase-1-verifier",
        "tracked_tree_clean": False,
        "tracked_status": (" M README.md",),
        "untracked_tree_clean": False,
        "untracked_count": 1,
        "untracked_status": ("new.py",),
        "development_stream_seeds": DEV,
        "validation_stream_seeds": VAL,
        "streams_per_alphabet_stratum": {
            "development": len(DEV), "validation": len(VAL)},
        "identities_per_stream": 8,
        "task_conditions_per_stream": conditions,
        "query_budgets": G.QUERY_BUDGETS,
        "validation_frozen_margins": {
            "L3_retrieval_noninferiority": F(1, 20),
            "L10_negative_transfer": F(1, 20),
        },
        "experiment_runtime_ms": 100,
        "full_suite_runtime_ms": 200,
        "full_suite_evidence": {
            "command": ("uv", "run", "pytest", "-q"),
            "exit_code": 0,
            "passed": 640,
            "skipped": 1,
            "runtime_ms": 200,
            "result_line": "640 passed, 1 skipped in 200 ms",
        },
        "artifact_paths": _paths(),
        "final_manifest_written": False,
        "final_stream_seed_sampled": False,
        "complete": True,
        "calibration": {"fires": True},
    }


def _risk_row(history=()):
    return {
        "stable_action": 0,
        "latent_action": 1,
        "stable_risk": F(0),
        "latent_action_risk_under_stable": F(1, 4),
        "passed": True,
        "matched_history": True,
        "stable_history": history,
        "latent_history": history,
        "latent_has_NEW": True,
        "latent_has_OUT": True,
    }


def _matched_split():
    tasks = 2
    arms = {}
    for name in G.RISK_ARMS:
        stable = name.startswith("stable")
        q = 0 if name.endswith("q0") else 1
        arms[name] = {
            "correct": tasks,
            "tasks": tasks,
            "queries": tasks * q,
            "accuracy": F(1),
            "mean_queries": F(q),
            "mean_taskwise_bayes_risk": F(0) if stable else F(1, 4),
        }
    rows = []
    for i in range(tasks):
        rows.append({
            "seed": DEV[i], "kind": "returning", "slot": i,
            "q0": _risk_row(),
            "q1": _risk_row(((0, 0),)),
            "oracle_query": _risk_row(((1, 1),)),
            "model": {
                "latent_components": (
                    "stored_records", "NEW_IDENTITY", "OUT_OF_FAMILY"),
                "out_query_semantics": "declared",
            },
        })
    return {
        "arms": arms,
        "taskwise_failures": (),
        "tasks": tasks,
        "taskwise_rows": rows,
        "calibration": {
            "unmatched_history_rejected": True, "fires": True},
        "all_taskwise_pass": True,
    }


def _sketch():
    digest = "b" * 64
    domains = []
    for overlap in G.OVERLAPS:
        domains.append({
            "overlap": overlap,
            "checks": {"indicator": True, "record_independent": True},
            "convention_count": 10,
            "meaning_count": 32,
            "legal_grounded_pairs": 100,
            "persistent_factor_entries_checked": 1000,
            "override_entries_checked": 0,
            "proof_document_sha256": digest,
            "passed": True,
            "failed_checks": (),
        })
    differential = {}
    for overlap in G.OVERLAPS:
        differential[overlap] = {
            "overlap": overlap, "seeds": (400, 401), "tasks": 4,
            "reachable_states": 8, "clarification_answers": 16,
            "exact_comparisons": 200,
            "mismatches": {name: 0 for name in G.PROOF_OBLIGATIONS},
            "selection_weight_nonuniform_states": 2,
            "weighted_query_differs_from_uniform": 1,
            "passed": True,
        }
    countermodels = {
        overlap: {
            "specification": {"overlap": overlap, "factor_overrides": ({},)},
            "same_domain_validator": {
                "overlap": overlap, "checks": {"indicator": False},
                "passed": False, "failed_checks": ("indicator",),
            },
            "support_preserved": True,
            "posterior_gap": True,
            "witness": {"support_preserved": True,
                        "posterior_gap": True,
                        "full_mass": F(1, 3),
                        "sketch_mass": F(1, 2)},
            "rejected": True,
        }
        for overlap in G.OVERLAPS
    }
    return {
        "mathematical_claim": "g -> q(phi|g) exactly -> L(e|g) exactly",
        "proof": {
            "theorem": "finite exact sufficiency",
            "assumptions": ("indicator factors",),
            "obligations": [
                {"name": name, "equality": "full=sketch", "reason": "finite"}
                for name in G.PROOF_OBLIGATIONS
            ],
            "domain_validations": domains,
            "proof_document": "/tmp/proof.md",
            "proof_document_sha256": digest,
            "arithmetic": "exact integer counts and Fraction posteriors",
            "proof_kind": "mathematical finite-algebra proof",
            "proof_assistant_verified": False,
            "differential_role": "corroboration only",
            "valid": True,
        },
        "selection_aware_weights_nonuniform": {
            overlap: 2 for overlap in G.OVERLAPS},
        "finite_authored_model_countermodels": countermodels,
        "differential": differential,
        "incomplete_retrieval": False,
        "wording": "algebraically proved inside the finite authored model",
        "calibration": {
            "same_premise_validator_rejects_hidden_nonindicator_weight": True,
            "plant_preserves_support_but_changes_exact_mass": True,
            "nonuniform_weight_case_present": True,
            "uniform_query_utility_differs": True,
            "fires": True,
        },
    }


def _summary_row(tasks=1, accuracy=F(1), asked=1):
    return {
        "tasks": tasks,
        "task_accuracy": accuracy,
        "equivalence_retrieval": F(1),
        "literal_identity": F(1),
        "queries_offered": tasks * 8,
        "queries_asked": asked,
        "mean_queries_all_tasks": F(asked, tasks),
        "mean_queries_ambiguous_tasks": F(0),
        "false_confident_answers": 0,
        "unresolved_outcomes": 0,
        "provisional_branches": 0,
        "established_record_corruption": 0,
        "physical_query_types": {"semantic": asked, "task": 0},
        "resolved_latent_quantities": {
            "identity": asked, "convention": asked,
            "task": asked, "cause": 0,
        },
    }


def _retrieval_curve(arm):
    delayed = arm in {
        "random_retrieval", "recency", "surface_nearest", "no_memory"}
    curve = {
        0: _summary_row(1, F(1, 2), 0),
        1: _summary_row(1, F(3, 4) if delayed else F(1), 1),
        2: _summary_row(1, F(1), 2),
        3: _summary_row(1, F(1), 3),
        4: _summary_row(1, F(1), 4),
    }
    return {
        "curve": curve,
        "questions_to_frozen_target_accuracy": 2 if delayed else 1,
        "frozen_target_accuracy": G.TARGET_ACCURACY,
        **curve[1],
    }


def _accounting(arm, seed, task):
    is_a = arm != "protocol_B_four_record"
    inspected = 8 if is_a else 4
    shortlist = 8 if arm == "exact_all_record" else 4
    return {
        "stream_seed": seed,
        "task_id": task,
        "condition": "returning",
        "true_slot": 0,
        "current_utterance": 10,
        "task_candidate_pool": ((0, 1),),
        "selected_keys": tuple(range(shortlist)),
        "protocol": ("A_GLOBAL_EXACT_SCAN" if is_a
                     else "B_FOUR_RECORD_COARSE_NOMINATION"),
        "index_bytes_scanned": 345 if is_a else 40,
        "identity_specific_summaries_inspected": inspected,
        "identity_likelihoods_evaluated": inspected,
        "shortlist_size": shortlist,
        "full_records_loaded": 0,
        "sketch_bytes_loaded": 304 if is_a else 152,
        "total_retrieval_bytes": 345 if is_a else 192,
        "total_retrieval_node_equivalents": inspected,
        "incomplete_retrieval": arm != "exact_all_record",
        "within_512": True,
        "four_node_claim": not is_a,
        "validation": {
            "checks": {"contract": True, "physical_bytes": True},
            "passed": True,
            "failed_checks": (),
        },
    }


def _extrema(arm):
    row = _accounting(arm, 0, "x")
    return {name: {"min": row[name], "max": row[name]}
            for name in G.RETRIEVAL_ACCOUNTING_FIELDS}


def _central_intervals(clusters):
    positive = _interval(F(1, 4), clusters=clusters)
    return {
        "MAIN_minus_random_accuracy": copy.deepcopy(positive),
        "MAIN_minus_recency_accuracy": copy.deepcopy(positive),
        "MAIN_minus_surface_accuracy": copy.deepcopy(positive),
        "MAIN_minus_exact_accuracy": _interval(F(0), clusters=clusters),
        "MAIN_minus_no_memory_query_count":
            _interval(F(-1), clusters=clusters),
    }


def _retrieval_split(seeds):
    streams = {
        seed: {arm: _retrieval_curve(arm) for arm in G.RETRIEVAL_ARMS}
        for seed in seeds
    }
    accounting = {
        arm: [_accounting(arm, seed, f"{seed}:0") for seed in seeds]
        for arm in G.RETRIEVAL_ACCOUNTING_ARMS
    }
    freeze_record = {
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
    return {
        "streams": streams,
        "per_task_accounting": accounting,
        "intervals": _central_intervals(len(seeds)),
        "protocol_A": {
            "claim": "global exact scan, no four node claim",
            "accounting_extrema": _extrema("MAIN_protocol_A"),
            "all_within_512": True,
            "no_four_node_claim": True,
            "all_main_rows_report_incomplete": True,
            "all_rows_validated": True,
        },
        "protocol_B": {
            "claim": "nonsufficient coarse nomination then <=4 exact records",
            "accounting_extrema": _extrema("protocol_B_four_record"),
            "all_within_512": True,
            "all_at_most_four": True,
            "all_incomplete_reported": True,
            "all_rows_validated": True,
        },
        "main_protocol": "A_GLOBAL_EXACT_SCAN_TOP4",
        "main_protocol_freeze": {
            "record": freeze_record, "sha256": G._canon_sha(freeze_record)},
        "main_claim": "query-efficient approximate top-four shortlist after "
                      "a global exact-sketch scan <=512 B and eight node-"
                      "equivalents; not four-node exact retrieval",
        "resource_matched_controls": True,
        "no_memory_baseline": (
            "validation-frozen fresh family-prior exact task-information "
            "policy over legal behavioral and semantic questions"),
        "coarse_nonsufficiency_collisions": len(seeds),
        "active_bytes": {"max": 3042, "all_within_4KiB": True},
        "calibration": {
            "coarse_collision_fired": True,
            "valid_rows_accepted_by_accounting_validator": True,
            "free_exact_index_undercharge_rejected_by_same_validator": True,
            "fires": True,
        },
    }


def _query_accounting():
    def row(q, asked, accuracy):
        return {
            "query_budget": q,
            "queries_offered": asked + 10,
            "queries_actually_asked": asked,
            "mean_over_all_tasks": F(asked, 120),
            "mean_over_ambiguous_tasks": F(asked, 120),
            "mean_over_scored_returning_tasks": F(asked, 120),
            "total_per_stream": (asked,),
            "task_accuracy": accuracy,
            "metric_denominator": "returning+ambiguous+misleading",
            "query_type": {
                "semantic": asked, "identity": asked,
                "convention": asked, "task": 0, "cause": 0,
                "note": "overlapping latent effects",
            },
        }
    return {
        "overlap": "disjoint_op",
        "seeds": (400, 401, 402),
        "tasks": 120,
        "budgets": {
            0: row(0, 0, F(53, 84)),
            1: row(1, 100, F(80, 84)),
            3: row(3, 295, F(1)),
        },
        "published_curve_reproduced": True,
        "published_curve": "0.631 -> 0.952 at budget one",
        "q_2_46_definition": "295/120 mean actual questions",
        "metrics_internally_consistent": True,
        "calibration": {
            "old_zero_query_label_rejected": True, "fires": True},
    }


def _effect(i):
    return {
        "event": {"query": {"kind": "semantic", "value": i},
                  "answer": i},
        "changed": {
            "identity": False, "convention": True,
            "task": True, "cause": False,
        },
        "support": {
            "identity": (1, 1), "convention": (8, 4), "task": (4, 2)},
        "resolved_quantities": ("convention", "task"),
    }


def _memoryless_policy(policy):
    curve = {}
    for q in G.QUERY_BUDGETS:
        per_task = [
            {"task_digest": f"task-{task}",
             "effects": [_effect(i) for i in range(q)]}
            for task in range(2)
        ]
        asked = 2 * q
        curve[q] = {
            "tasks": 2,
            "task_accuracy": F(1, 2) if q == 0 else F(1),
            "query_budget": q,
            "queries_offered": asked + q,
            "queries_actually_asked": asked,
            "mean_queries_all_tasks": F(q),
            "convention_entropy": {"before": "4.0", "after": "2.0"},
            "task_entropy": {"before": "2.0", "after": "1.0"},
            "candidate_class_count": {"before": F(4), "after": F(2)},
            "query_types": {"semantic": asked, "behavioral": 0},
            "latent_quantity": G.LATENT_QUANTITIES[policy],
            "resolved_latent_quantities": {
                "identity": 0, "convention": asked,
                "task": asked, "cause": 0,
            },
            "per_question_resolution_effects": per_task,
            "answers_applied": True,
        }
    return curve


def _memoryless_split():
    return {
        "tasks": 2,
        "population": "all_matched_scored",
        "policies": {
            policy: _memoryless_policy(policy)
            for policy in G.MEMORYLESS_POLICIES
        },
        "oracle_legal_query_improves": True,
        "nonoracle_policy_improves": True,
        "fresh_equals_stable_fresh": True,
        "all_answers_applied": True,
        "calibration": {
            "valid_input": True,
            "rejected": {
                "counted_without_answer": True,
                "missing_resolution_effect": True,
                "unapplied_answer": True,
            },
            "fires": True,
        },
    }


def _aggregate_curve(seed, policy):
    rows = {
        q: _summary_row(2, F(1, 2) if q == 0 else F(1), 2 * q)
        for q in G.QUERY_BUDGETS
    }
    return {
        "policy": policy,
        "frozen_target_accuracy": G.TARGET_ACCURACY,
        "population_tasks": 2,
        "complete_stream_seeds": (seed,),
        "budgets": rows,
        "prefix_consistent": True,
        "minimum_questions_to_frozen_target": 1,
        "metric_definition": "minimum aggregate declared query budget",
    }


def _active_split(seeds):
    intervals = {
        name: _interval(F(0), clusters=len(seeds), seed=i)
        for i, name in enumerate(sorted(G.CT.REQUIRED_ACTIVE_METRICS))
    }
    streams = {}
    for seed in seeds:
        streams[seed] = {}
        for name, policy in (("information_gain", "joint_information_gain"),
                             ("random", "random_legal")):
            curve = _aggregate_curve(seed, policy)
            streams[seed][name] = {
                **curve["budgets"][1],
                "questions_at_matched_accuracy": 1,
                "accuracy_curve": curve,
            }
    return {
        "streams": streams,
        "intervals": intervals,
        "frozen_target_accuracy": G.TARGET_ACCURACY,
        "not_reached_censor_value": 5,
        "questions_metric_definition":
            "minimum aggregate q=0..4 budget reaching frozen 0.95 accuracy",
        "component_status": "not_measured_in_X65A-L",
        "all_operational_intervals_include_zero": True,
        "calibration": {
            "valid_input": True,
            "rejected": {
                "missing_metric": True, "reversed_interval": True,
                "wrong_cluster_count": True,
                "wrong_component_status": True,
            },
            "fires": True,
        },
    }


def _new_arm():
    return {
        "precision": F(1), "recall": F(1),
        "forced_assimilation_rate": F(0),
        "returning_false_new": 0,
    }


def _new_split(seeds):
    arms = {name: _new_arm() for name in G.NEW_CONTROL_ARMS}
    arms["always_reuse_nearest"]["forced_assimilation_rate"] = F(1)
    arms["no_new_unresolved"]["recall"] = F(0)
    arms["no_new_forced"]["forced_assimilation_rate"] = F(1)
    arms["always_create_new"]["returning_false_new"] = 7
    stream_rows = []
    fired = {
        "always_reuse_forces_assimilation": True,
        "always_create_false_new": True,
        "unresolved_does_not_count_as_recall": True,
        "no_new_forced_is_rejected": True,
        "oracle_control_passes_same_gate": True,
        "promotion_same_validator_rejects_bad_arms": True,
        "reuse_same_validator_rejects_plants": True,
    }
    for _seed in seeds:
        stream_rows.append({
            "constructible": True,
            "main_classification_gate": True,
            "contamination_during_creation": 0,
            "successfully_promoted_new_records": 1,
            "later_reuse_of_new_records": 1,
            "calibration_fired": copy.deepcopy(fired),
        })
    return {
        "streams": len(seeds),
        "precision": F(1), "recall": F(1),
        "false_new_rate_returning": F(0),
        "forced_assimilation_rate": F(0),
        "unresolved_new_rate": F(0),
        "questions_to_grounded_creation": F(2),
        "successfully_promoted_new_records": len(seeds),
        "later_reuse_of_new_records": len(seeds),
        "contamination_during_creation": 0,
        "record_bytes_added": tuple(120 for _ in seeds),
        "active_bytes": {"max": 3000, "all_within_4KiB": True},
        "arms": arms,
        "stream_rows": stream_rows,
        "later_reuse_failures": (),
        "main_classification_gate": True,
        "paired_forced_assimilation_vectors": {
            "MAIN": tuple(F(0) for _ in seeds),
            "no_NEW_forced": tuple(F(1) for _ in seeds),
        },
        "calibration": {
            "fires": True,
            "details": tuple(copy.deepcopy(fired) for _ in seeds),
        },
    }


def _scope(overlap):
    restricted_case = {
        "phi_a": 0, "phi_b": 1, "query_set": (0, 1, 2, 3),
        "outside_witness": 4, "restricted_equal": True,
        "globally_equal": False,
    }
    transfer = {
        "constructible": overlap == "disjoint_op",
        "utterance": (1, 2) if overlap == "disjoint_op" else None,
        "scope": ("nonvacuous frozen-alphabet transfer"
                  if overlap == "disjoint_op"
                  else "untestable in frozen shared two-token alphabet"),
        "zero_family_likelihood": overlap == "disjoint_op",
    }
    constructibility = {
        "out_of_family_convention": {
            "constructible": True,
            "convention": {"defect": "noninjective"},
            "family_membership_count": 0,
            "tested_via": "authored noninjective role map, then grounded",
            "grounded_contradiction": {
                "events": ((0, 1),), "zero_survivors": True},
        },
        "out_of_family_transfer_utterance": transfer,
        "out_of_family_grounded_event": {
            "constructible": True, "events": ((0, 1),),
            "minimum_event_count": 1,
            "source": "authored_out_of_family_convention",
            "zero_survivors": True,
        },
        "UNKNOWN_MEANING": {
            "constructible": True, "demonstrations": ((0, 1),),
            "derived_live_count": 0,
        },
        "MISSING_REPRESENTATION": {
            "constructible": True, "tested": True,
            "outcome": "MISSING_REPRESENTATION",
            "cause_posterior": {"MISSING_REPRESENTATION": F(1)},
        },
        "restricted_query_indistinguishable": {
            "constructible": True, "case": restricted_case},
    }
    restricted_scope = {
        "constructible": True,
        "case": copy.deepcopy(restricted_case),
        "promotions": 1,
        "record": {
            "scope": {
                "challenge_universe_digest": "c" * 64,
                "query_set_digest": "d" * 64,
                "validity_scope": "controlled authored X64H semantic family",
                "status": "empirical",
            },
        },
        "false_global_promotions": 0,
        "calibration_false_global_promotions": 1,
        "validation": {
            "checks": {"empirical_scope": True, "confirmed": True},
            "passed": True,
        },
        "calibration": {
            "plant": {"scope": {"status": "global_in_finite_model"}},
            "same_validator": {
                "checks": {"global_status_requires_global_equivalence": False},
                "passed": False,
            },
            "fires": True,
        },
    }
    return {"constructibility": constructibility,
            "restricted_scope": restricted_scope}


def _negative_split(overlap, seeds):
    conditions = {}
    for name in G.NEGATIVE_CONDITIONS:
        untestable = overlap == "shared" and name == "out_of_family_partner"
        conditions[name] = {
            "constructible_transfer": not untestable,
            "tested_via": ("grounded_event" if untestable else "transfer_task",),
            "task_accuracy": ({"MAIN": None, "no_memory": None}
                              if untestable else
                              {"MAIN": F(1), "no_memory": F(1)}),
            "accuracy_delta": None if untestable else F(0),
            "excess_questions": 0,
            "false_confident_actions": 0,
            "established_record_corruption": 0,
            "provisional_branches": 1,
            "unresolved_outcomes": 1,
            "noninferior": True,
            "matched_protocol": True,
        }
    def arm_metrics(accuracy):
        return {
            "task_accuracy": accuracy,
            "queries_offered": 8 if accuracy is not None else 0,
            "queries_asked": 1 if accuracy is not None else 0,
            "excess_questions": 0,
            "false_confident_actions": 0,
            "established_record_corruption": 0,
            "provisional_branches": 0,
            "unresolved_outcomes": int(accuracy is None),
            "action": 0 if accuracy is not None else None,
            "query_policy": "information_gain",
            "identity_decision": "UNRESOLVED" if accuracy is None else "record:0",
            "has_new_component": True,
            "has_out_component": True,
        }

    def protocol_contract(scored):
        contract = {
            "same_current_task": True,
            "same_truthful_answer_channel": True,
            "same_query_budget": True,
            "same_zero_one_task_loss": True,
            "same_budget_exhaustion_stopping_rule": True,
            "same_metric_denominator": True,
            "main_legal_query_types": ("semantic",),
            "no_memory_legal_query_types": ("behavioral", "semantic"),
            "different_query_universes_explicit": True,
            "answers_applied_to_current_posterior": True,
            "both_within_budget": True,
        }
        if scored:
            contract.update({
                "main_retrieval": {
                    "protocol": "A_GLOBAL_EXACT_SCAN",
                    "index_bytes_scanned": 38,
                    "identity_specific_summaries_inspected": 1,
                    "identity_likelihoods_evaluated": 1,
                    "shortlist_size": 1,
                    "full_records_loaded": 0,
                    "sketch_bytes_loaded": 38,
                    "total_retrieval_bytes": 38,
                    "total_retrieval_node_equivalents": 1,
                    "incomplete_retrieval": False,
                    "within_512": True,
                    "four_node_claim": False,
                },
                "main_retrieval_selected_keys": (0,),
            })
        return contract

    rows = []
    for seed in seeds:
        condition_rows = []
        for name in G.NEGATIVE_CONDITIONS:
            untestable = overlap == "shared" and name == "out_of_family_partner"
            accuracy = None if untestable else F(1)
            condition_rows.append({
                "condition": {"name": name},
                "main": arm_metrics(accuracy),
                "no_memory": arm_metrics(accuracy),
                "accuracy_delta": None if untestable else F(0),
                "frozen_margin": F(-1, 20),
                "noninferior": True,
                "matched_protocol": True,
                "protocol_contract": protocol_contract(not untestable),
            })
        rows.append({
            "phase": "X65A-L1-negative-transfer",
            "overlap": overlap,
            "seed": seed,
            "query_budget": 1,
            "accuracy_margin": F(-1, 20),
            "metric_denominator": "all_tasks_per_constructible_condition",
            "conditions": condition_rows,
            "gates": {"all_conditions": True, "noninferior": True},
            "calibrations": {"all_fire": True},
        })
    safety = {
        G.SAFETY_INTERVALS[0]: _interval(F(-1), clusters=len(seeds)),
        G.SAFETY_INTERVALS[2]: _interval(F(-1), clusters=len(seeds)),
    }
    return {
        "seeds": seeds,
        "frozen_margin": F(1, 20),
        "conditions": conditions,
        "stream_rows": rows,
        "all_conditions_reported": True,
        "main_noninferior": True,
        "main_established_corruption": 0,
        "noninferiority_failures": (),
        "matched_protocol": True,
        "intervals": safety,
        "paired_safety_vectors": {
            "provisional_assignment_corruption":
                tuple(F(0) for _ in seeds),
            "immediate_MAP_corruption": tuple(F(1) for _ in seeds),
            "confirmation_contamination": tuple(F(0) for _ in seeds),
            "no_confirmation_contamination": tuple(F(1) for _ in seeds),
        },
        "calibration": {
            "fires": True,
            "immediate_MAP_corruption": len(seeds),
            "forced_new_assimilation": len(seeds),
            "no_confirmation_contamination": len(seeds),
            "same_main_safety_predicate_rejected_every_plant": True,
        },
    }


_RESTART_TEMPLATES = {}


def _restart_template(overlap):
    if overlap not in _RESTART_TEMPLATES:
        state = R.fixture_state(overlap)
        suffix = R.fixture_suffix(state)
        final = R.advance_state(state, suffix[0])
        _RESTART_TEMPLATES[overlap] = (
            state.canon(), suffix, G._canon_sha(final))
    return _RESTART_TEMPLATES[overlap]


def _restart_case(overlap, split, seed, plant=None):
    state, suffix, final_digest = _restart_template(overlap)
    state = copy.deepcopy(state)
    case = {
        "overlap": overlap, "split": split, "seed": seed,
        "post_query_history": copy.deepcopy(
            state["query_policy_state"]["history"]),
        "state_schema": state.pop("schema"),
        "state_step": state.pop("step"),
        **state,
    }
    state_payload = {
        "schema": case["state_schema"], "overlap": overlap,
        "step": case["state_step"],
        **{field: case[field] for field in G.RESTART_FIELDS},
    }
    queries = [int(row["query"]) for row in suffix]
    answers = [int(row["answer"]) for row in suffix]
    case["cycle"] = {
        "ok": True, "overlap": overlap,
        "parent_pid": seed, "child_pid": seed + 100000,
        "parent_pid_gone": True,
        "checkpoint_sha256": G._canon_sha(state_payload),
        "checkpoint_bytes": len(G.encode(state_payload)),
        "checkpoint_hashes": copy.deepcopy(case["serialized_hashes"]),
        "child_loaded_parent_state": True,
        "uninterrupted_final_sha256": final_digest,
        "restarted_final_sha256": final_digest,
        "final_hashes_identical": True,
        "final_state_identical": True,
        "loaded_step": case["state_step"],
        "final_step": case["state_step"] + 1,
        "real_main_continuation": True,
        "continuation_policy": "information_gain",
        "continuation_queries": queries,
        "continuation_answers": answers,
        "forbidden_channel_closed": True,
        "child_env_size": 3,
        "calibrations": {},
        "all_calibrations_rejected": True,
    }
    case["calibration_plants"] = [] if plant is None else [[plant, "mutate"]]
    case["corrupt_checkpoint_calibrations"] = (
        {} if plant is None else {
            f"mutate:{plant}": {
                "rejected": True, "returncode": 1, "error": "rejected",
                "same_child_validator":
                    "x65a.restart_l1 child/state_from_payload",
            }})
    return case


def _restart_matrix():
    keys = [(overlap, split, seed)
            for overlap in G.OVERLAPS
            for split, seeds in (("development", DEV), ("validation", VAL))
            for seed in seeds]
    cases = [_restart_case(*key,
                           G.RESTART_FIELDS[index]
                           if index < len(G.RESTART_FIELDS) else None)
             for index, key in enumerate(keys)]
    case_checks = {
        f"{overlap}/{split}/{seed}": {
            name: True for name in G.RESTART_MATRIX_CASE_CHECKS}
        for overlap, split, seed in keys
    }
    validation = {
        "checks": {name: True for name in G.RESTART_MATRIX_CHECKS},
        "case_checks": case_checks,
        "errors": (),
        "passed": True,
    }
    planted = copy.deepcopy(validation)
    planted["case_checks"][next(iter(planted["case_checks"]))][
        "all_scheduled_corrupt_checkpoints_rejected"] = False
    planted["errors"] = ("accepted corrupt checkpoint",)
    planted["passed"] = False
    return {
        "schema": G.RESTART_MATRIX_SCHEMA,
        "contract": {
            "development_seeds": DEV,
            "validation_seeds": VAL,
            "overlaps": G.OVERLAPS,
            "identities_per_stream": 8,
            "required_corruption_fields": G.RESTART_FIELDS,
            "corruption_mode": "mutate",
            "expected_streams": 16,
        },
        "cases": cases,
        "validation": validation,
        "calibration": {
            "valid_matrix_accepted": True,
            "accepted_corruption_rejected_by_same_matrix_validator": True,
            "planted_validation": planted,
            "fires": True,
        },
    }


def _artifact():
    restart = _restart_matrix()
    checkpoint_sizes = tuple(
        row["cycle"]["checkpoint_bytes"] for row in restart["cases"])
    art = {
        "phase": "X65A-L1", "schema": 1,
        "provenance": _provenance(),
        "x64h_prerequisite": {"passed": True, "checks": 21,
                               "passed_checks": 21},
        "claim_boundary": "controlled authored semantic environment",
        "unsupported_scope": (
            "natural language", "lifelong learning", "AGI",
            "bounded total memory"),
        "x65a_p_started": False,
        "matched_risk": {}, "sketch_sufficiency": _sketch(),
        "retrieval": {}, "query_accounting": _query_accounting(),
        "memoryless": {}, "active_query": {}, "new_identity": {},
        "scope_audit": {overlap: _scope(overlap)
                        for overlap in G.OVERLAPS},
        "negative_transfer": {},
        "restart": restart,
        "restart_storage_accounting": {
            "category": "serialized_restart_and_audit_state",
            "checkpoints": len(checkpoint_sizes),
            "minimum_checkpoint_bytes": min(checkpoint_sizes),
            "maximum_checkpoint_bytes": max(checkpoint_sizes),
            "total_checkpoint_bytes": sum(checkpoint_sizes),
            "active_semantic_memory_budget_bytes": 4096,
            "counted_against_active_semantic_memory": False,
            "boundary": (
                "reported separately; no bounded-total-memory claim"),
            "case_sizes_match": True,
        },
        "required_intervals": {},
    }
    for overlap in G.OVERLAPS:
        art["matched_risk"][overlap] = {
            split: _matched_split() for split in G.SPLITS}
        art["retrieval"][overlap] = {
            "development": _retrieval_split(DEV),
            "validation": _retrieval_split(VAL),
        }
        art["memoryless"][overlap] = {
            split: _memoryless_split() for split in G.SPLITS}
        art["active_query"][overlap] = {
            "development": _active_split(DEV),
            "validation": _active_split(VAL),
        }
        art["new_identity"][overlap] = {
            "development": _new_split(DEV),
            "validation": _new_split(VAL),
        }
        art["negative_transfer"][overlap] = {
            "development": _negative_split(overlap, DEV),
            "validation": _negative_split(overlap, VAL),
        }
        art["required_intervals"][overlap] = {}
        for split in G.SPLITS:
            retrieval = art["retrieval"][overlap][split]
            negative = art["negative_transfer"][overlap][split]
            art["required_intervals"][overlap][split] = {
                **copy.deepcopy(retrieval["intervals"]),
                **copy.deepcopy(negative["intervals"]),
                "NEW_minus_no_NEW_forced_assimilation":
                    _interval(F(-1), clusters=len(
                        DEV if split == "development" else VAL)),
                "restart_difference": _interval(F(0), clusters=len(
                    DEV if split == "development" else VAL)),
            }
    return art


def test_complete_synthetic_artifact_passes_every_pure_gate():
    report = G.evaluate_l1_artifact(_artifact())
    assert report.passed, report.canon()["gate_errors"]
    assert report.failures == ()
    assert set(report.gates) == set(G.GATE_NAMES)


@pytest.mark.parametrize("gate", G.GATE_NAMES)
def test_each_gate_same_validator_mutations_are_red(gate):
    artifact = _artifact()
    assert G.validate_gate(artifact, gate).ok
    calibration = G.calibrate_gate(artifact, gate)
    assert calibration["valid_input"]
    assert calibration["same_validator"]
    assert calibration["fires"], calibration
    assert all(calibration["rejected"].values())
    assert all(calibration["rejection_errors"].values())
    if gate == "L1.10":
        assert {
            "resealed_retrieval_undercharge", "resealed_record_support",
            "resealed_new_support", "resealed_selection_weights",
            "resealed_inference_priors", "resealed_task_truth_leak",
            "resealed_nonmain_policy", "different_exact_final_state",
            "fake_main_continuation", "wrong_main_continuation_query",
            "misclassified_restart_storage",
        }.issubset(calibration["rejected"])


def test_l110_requires_strict_interval_directions_and_zero_restart():
    artifact = _artifact()
    row = artifact["required_intervals"]["shared"]["validation"]
    row["MAIN_minus_random_accuracy"]["lo"] = F(0)
    row["MAIN_minus_no_memory_query_count"]["hi"] = F(0)
    row["confirmation_minus_no_confirmation_contamination"]["hi"] = F(0)
    row["restart_difference"]["delta"] = F(1)
    result = G.validate_gate(artifact, "L1.10")
    assert not result.ok
    assert any("lower bound is not > 0" in error for error in result.errors)
    assert any("query advantage upper bound is not < 0" in error
               for error in result.errors)
    assert any("confirmation_minus_no_confirmation_contamination" in error
               and "not < 0" in error for error in result.errors)
    assert any("restart interval is not exactly zero" in error
               for error in result.errors)


def test_l13_rejects_exact_summaries_hidden_in_a_free_index():
    artifact = _artifact()
    row = artifact["retrieval"]["disjoint_op"]["validation"][
        "per_task_accounting"]["MAIN_protocol_A"][0]
    row["identity_specific_summaries_inspected"] = 4
    row["identity_likelihoods_evaluated"] = 4
    row["total_retrieval_node_equivalents"] = 1
    result = G.validate_gate(artifact, "L1.3")
    assert not result.ok
    assert any("hides identity summaries" in error or
               "charge all eight" in error for error in result.errors)


def test_l11_recomputes_inequality_instead_of_trusting_passed_flag():
    artifact = _artifact()
    comparison = artifact["matched_risk"]["shared"]["development"][
        "taskwise_rows"][0]["oracle_query"]
    comparison["stable_risk"] = F(1)
    comparison["latent_action_risk_under_stable"] = F(0)
    comparison["passed"] = True
    result = G.validate_gate(artifact, "L1.1")
    assert not result.ok
    assert any("stable risk <= latent risk" in error for error in result.errors)


def test_l12_requires_support_preserving_countermodel_calibration():
    artifact = _artifact()
    artifact["sketch_sufficiency"]["calibration"][
        "plant_preserves_support_but_changes_exact_mass"] = False
    result = G.validate_gate(artifact, "L1.2")
    assert not result.ok
    assert any("sufficiency calibration did not fire" in error
               for error in result.errors)


def test_l110_rejects_incomplete_matrix_and_failed_cycle():
    artifact = _artifact()
    artifact["restart"]["cases"].pop()
    artifact["restart"]["cases"][0]["cycle"]["ok"] = False
    result = G.validate_gate(artifact, "L1.10")
    assert not result.ok
    assert any("exactly 16 cases" in error for error in result.errors)
    assert any("cycle did not succeed" in error for error in result.errors)


def test_l13_requires_exact_freeze_and_protocol_b_incomplete_disclosure():
    artifact = _artifact()
    section = artifact["retrieval"]["shared"]["validation"]
    section["main_protocol_freeze"]["record"]["byte_limit"] = 511
    section["main_protocol_freeze"]["sha256"] = G._canon_sha(
        section["main_protocol_freeze"]["record"])
    section["per_task_accounting"]["protocol_B_four_record"][0][
        "incomplete_retrieval"] = False
    result = G.validate_gate(artifact, "L1.3")
    assert not result.ok
    assert any("freeze record differs" in error for error in result.errors)
    assert any("Protocol B incompleteness" in error for error in result.errors)


@pytest.mark.parametrize("policy", (
    "no_memory_exact_task_information_gain",
    "no_memory_exact_convention_task_information_gain",
))
def test_l15_requires_both_exact_information_gain_policies_to_improve(policy):
    artifact = _artifact()
    curve = artifact["memoryless"]["shared"]["validation"]["policies"][policy]
    curve[4]["task_accuracy"] = curve[0]["task_accuracy"]
    result = G.validate_gate(artifact, "L1.5")
    assert not result.ok
    assert any(policy in error and "does not improve" in error
               for error in result.errors)


def test_l18_requires_scope_plant_rejected_by_same_validator():
    artifact = _artifact()
    calibration = artifact["scope_audit"]["shared"]["restricted_scope"][
        "calibration"]
    calibration["same_validator"]["passed"] = True
    calibration["fires"] = True
    result = G.validate_gate(artifact, "L1.8")
    assert not result.ok
    assert any("scope plant was not rejected" in error
               for error in result.errors)


def test_l19_requires_explicit_matched_protocol_contract():
    artifact = _artifact()
    condition = artifact["negative_transfer"]["disjoint_op"]["validation"][
        "stream_rows"][0]["conditions"][0]
    condition["protocol_contract"]["same_query_budget"] = False
    condition["matched_protocol"] = True
    result = G.validate_gate(artifact, "L1.9")
    assert not result.ok
    assert any("protocol contract has a false clause" in error
               for error in result.errors)


def test_l110_recomputes_bootstrap_bounds_from_paired_streams():
    artifact = _artifact()
    required = artifact["required_intervals"]["shared"]["validation"][
        "MAIN_minus_random_accuracy"]
    central = artifact["retrieval"]["shared"]["validation"]["intervals"][
        "MAIN_minus_random_accuracy"]
    required["hi"] += F(1, 100)
    central["hi"] += F(1, 100)
    result = G.validate_gate(artifact, "L1.10")
    assert not result.ok
    assert any("differs from exact paired resampling" in error
               for error in result.errors)


def test_l110_rejects_weak_safety_calibration_and_unmatched_resources():
    artifact = _artifact()
    negative = artifact["negative_transfer"]["shared"]["validation"]
    negative["calibration"][
        "same_main_safety_predicate_rejected_every_plant"] = False
    control = artifact["retrieval"]["shared"]["validation"][
        "per_task_accounting"]["surface_nearest"][0]
    control["total_retrieval_node_equivalents"] = 4
    result = G.validate_gate(artifact, "L1.10")
    assert not result.ok
    assert any("safety calibration is weak" in error
               for error in result.errors)
    assert any("central retrieval contract" in error and
               "resource-matched" in error for error in result.errors)
