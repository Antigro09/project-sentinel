"""Pure fail-closed evaluation of the X65A-L1 artifact.

The audit runner produces evidence; this module only reads that evidence.  It
does not import the runner, build streams, touch the filesystem, or recompute
an experiment.  Every gate is evaluated by a named pure validator and is then
red-calibrated by planting defects into a deep copy and sending those defects
through the *same* validator.

The public entry points are:

``validate_gate(artifact, gate)``
    Validate the measured evidence for one of ``L1.0`` through ``L1.10``.

``calibrate_gate(artifact, gate)``
    Run that validator on executable, gate-specific planted defects.

``evaluate_gate`` / ``evaluate_l1_artifact``
    Combine evidence and red calibration into a fail-closed result.

All probability, margin, and interval checks accept only ``int`` and
``Fraction``.  In particular, a decoded artifact containing a float cannot
pass by numerical tolerance.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from . import l1_contracts as CT
from . import restart_l1 as RL1
from . import l1_stats as ST
from .types import encode


GATE_NAMES = tuple(f"L1.{i}" for i in range(11))
OVERLAPS = ("shared", "disjoint_op")
SPLITS = ("development", "validation")
QUERY_BUDGETS = (0, 1, 2, 3, 4)
TARGET_ACCURACY = Fraction(19, 20)

RISK_ARMS = (
    "stable_q0", "latent_q0", "stable_q1", "latent_q1",
    "stable_oracle_query", "latent_oracle_query",
)
RISK_COMPARISONS = ("q0", "q1", "oracle_query")
MEMORYLESS_POLICIES = (
    "no_memory_random_legal",
    "no_memory_behavioral_disagreement",
    "no_memory_exact_task_information_gain",
    "no_memory_exact_convention_task_information_gain",
    "no_memory_oracle_task_separating",
    "fresh_x64h_family_prior",
    "stable_id_fresh_no_posterior",
)
LATENT_QUANTITIES = {
    "no_memory_random_legal": "mixed_random",
    "no_memory_behavioral_disagreement": "task_meaning",
    "no_memory_exact_task_information_gain": "task_meaning",
    "no_memory_exact_convention_task_information_gain":
        "convention_and_task",
    "no_memory_oracle_task_separating": "task_meaning_oracle",
    "fresh_x64h_family_prior": "task_meaning",
    "stable_id_fresh_no_posterior": "task_meaning",
}
RETRIEVAL_ARMS = (
    "MAIN_protocol_A", "exact_all_record", "protocol_B_four_record",
    "random_retrieval", "recency", "surface_nearest", "stable_ID",
    "no_memory",
)
RETRIEVAL_ACCOUNTING_ARMS = (
    "MAIN_protocol_A", "exact_all_record", "protocol_B_four_record",
    "random_retrieval", "recency", "surface_nearest",
)
RESOURCE_MATCHED_RETRIEVAL_CONTROLS = (
    "random_retrieval", "recency", "surface_nearest",
)
RETRIEVAL_ACCOUNTING_FIELDS = (
    "index_bytes_scanned",
    "identity_specific_summaries_inspected",
    "identity_likelihoods_evaluated",
    "shortlist_size",
    "full_records_loaded",
    "sketch_bytes_loaded",
    "total_retrieval_bytes",
    "total_retrieval_node_equivalents",
)
CENTRAL_INTERVALS = (
    "MAIN_minus_random_accuracy",
    "MAIN_minus_recency_accuracy",
    "MAIN_minus_surface_accuracy",
    "MAIN_minus_exact_accuracy",
    "MAIN_minus_no_memory_query_count",
)
SAFETY_INTERVALS = (
    "provisional_assignment_minus_immediate_MAP_corruption",
    "NEW_minus_no_NEW_forced_assimilation",
    "confirmation_minus_no_confirmation_contamination",
)
NEGATIVE_CONDITIONS = (
    "correct_returning_record",
    "wrong_similar_initially_favored",
    "stale_record",
    "two_equivalent_identities",
    "out_of_family_partner",
    "restricted_query_ambiguity",
    "new_identity",
    "multiple_old_records_consistent",
)
NEW_CONTROL_ARMS = (
    "always_reuse_nearest", "no_new_unresolved", "no_new_forced",
    "always_create_new", "oracle_new_returning_status",
)
PROOF_OBLIGATIONS = (
    "stored_posterior", "selection_aware_likelihood", "task_posterior",
    "clarification", "query_utility", "decision", "new_identity",
    "out_of_family",
)
RESTART_FIELDS = tuple(RL1.AUDITED_FIELDS)
HASHED_RESTART_FIELDS = tuple(RL1.HASHED_FIELDS)
RESTART_STATE_HASH_FIELDS = HASHED_RESTART_FIELDS + ("metadata",)
RESTART_MATRIX_SCHEMA = "x65a-l1-main-restart-matrix-v2"
RESTART_CASE_FIELDS = (
    "overlap", "split", "seed", "post_query_history", "state_schema",
    "state_step", *RESTART_FIELDS, "cycle",
    "calibration_plants", "corrupt_checkpoint_calibrations",
)
RESTART_MATRIX_CHECKS = (
    "schema_stream_matrix_complete", "no_duplicate_stream_reports",
    "all_required_corruption_fields_calibrated", "all_case_rows_valid",
)
RESTART_MATRIX_CASE_CHECKS = (
    "state_valid", "actual_post_query_history_present",
    "identity_posterior_exactly_normalized",
    "identity_posterior_has_shortlist_new_out", "new_mass_matches",
    "out_mass_matches", "shortlisted_convention_posteriors",
    "eight_confirmed_records", "provisional_branch_survives",
    "provisional_history_matches_main",
    "shortlist_is_nonempty_at_most_four",
    "protocol_a_scan_charged_as_eight_nodes",
    "serialized_hashes_complete",
    "cycle_reports_success", "parent_really_died", "child_pid_differs",
    "scrubbed_child_loaded_parent_bytes",
    "checkpoint_sha_matches_actual_state",
    "checkpoint_component_hashes_match",
    "restart_matches_uninterrupted_hash", "forbidden_channel_closed",
    "restart_matches_uninterrupted_exact_state",
    "child_continued_real_main_policy", "child_environment_scrubbed",
    "scheduled_calibration_rows_present",
    "all_scheduled_corrupt_checkpoints_rejected",
)


@dataclass(frozen=True)
class GateResult:
    gate: str
    evidence: CT.ValidationResult
    calibration: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return self.evidence.ok and self.calibration.get("fires") is True

    @property
    def errors(self) -> tuple[str, ...]:
        errors = list(self.evidence.errors)
        if self.calibration.get("fires") is not True:
            errors.append("same-validator red calibration did not fire")
        return tuple(errors)

    def canon(self) -> dict:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "errors": list(self.errors),
            "evidence": self.evidence.canon(),
            "calibration": dict(self.calibration),
        }


@dataclass(frozen=True)
class L1GateReport:
    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.results)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(row.gate for row in self.results if not row.passed)

    @property
    def gates(self) -> dict[str, bool]:
        return {row.gate: row.passed for row in self.results}

    def canon(self) -> dict:
        return {
            "gates": self.gates,
            "gate_calibrations": {
                row.gate: dict(row.calibration) for row in self.results},
            "gate_errors": {
                row.gate: list(row.errors) for row in self.results
                if row.errors
            },
            "failures": list(self.failures),
            "l1_passed": self.passed,
        }


def _finish(errors: Sequence[str]) -> CT.ValidationResult:
    unique = tuple(dict.fromkeys(str(error) for error in errors))
    return CT.ValidationResult(not unique, unique)


def _exact(value: Any) -> bool:
    return isinstance(value, (int, Fraction)) and not isinstance(value, bool)


def _fraction(value: Any, path: str, errors: list[str]) -> Fraction | None:
    if not _exact(value):
        errors.append(f"{path} must be an exact int or Fraction")
        return None
    return Fraction(value)


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be a mapping")
        return {}
    return value


def _required(row: Mapping, names: Sequence[str], path: str,
              errors: list[str]) -> None:
    for name in names:
        if name not in row:
            errors.append(f"{path} missing {name}")


def _budget(rows: Mapping, q: int):
    return rows[q] if q in rows else rows.get(str(q))


def _seed_count(artifact: Mapping, split: str, errors: list[str]) -> int:
    prov = _mapping(artifact.get("provenance"), "provenance", errors)
    name = ("development_stream_seeds" if split == "development"
            else "validation_stream_seeds")
    seeds = prov.get(name, ())
    if not isinstance(seeds, (list, tuple)) or not seeds:
        errors.append(f"provenance.{name} must be nonempty")
        return 0
    return len(seeds)


def _split_rows(artifact: Mapping, section_name: str,
                errors: list[str]):
    section = _mapping(artifact.get(section_name), section_name, errors)
    if set(section) != set(OVERLAPS):
        errors.append(f"{section_name} must contain exactly both strata")
    for overlap in OVERLAPS:
        by_split = _mapping(section.get(overlap),
                            f"{section_name}.{overlap}", errors)
        if set(by_split) != set(SPLITS):
            errors.append(
                f"{section_name}.{overlap} must contain development and validation")
        for split in SPLITS:
            yield overlap, split, _mapping(
                by_split.get(split),
                f"{section_name}.{overlap}.{split}", errors)


def _validate_interval(row: Any, expected_clusters: int, path: str,
                       errors: list[str]) -> None:
    row = _mapping(row, path, errors)
    _required(row, ("lo", "delta", "hi", "unit", "clusters",
                    "resamples", "seed"), path, errors)
    lo = _fraction(row.get("lo"), f"{path}.lo", errors)
    delta = _fraction(row.get("delta"), f"{path}.delta", errors)
    hi = _fraction(row.get("hi"), f"{path}.hi", errors)
    if lo is not None and delta is not None and hi is not None \
            and not lo <= delta <= hi:
        errors.append(f"{path} does not contain its estimate")
    if row.get("unit") != "complete_stream_or_latent_identity":
        errors.append(f"{path} has the wrong resampling unit")
    if row.get("clusters") != expected_clusters:
        errors.append(f"{path} has the wrong cluster count")
    if not isinstance(row.get("resamples"), int) \
            or isinstance(row.get("resamples"), bool) \
            or row.get("resamples", 0) <= 0:
        errors.append(f"{path} has an invalid resample count")
    if not isinstance(row.get("seed"), int) \
            or isinstance(row.get("seed"), bool):
        errors.append(f"{path} has an invalid resampling seed")


def _paired_delta(left: Any, right: Any, expected_clusters: int, path: str,
                  errors: list[str]) -> Fraction | None:
    if not isinstance(left, (list, tuple)) \
            or not isinstance(right, (list, tuple)) \
            or len(left) != expected_clusters \
            or len(right) != expected_clusters or not left:
        errors.append(f"{path} must contain paired complete-stream vectors")
        return None
    diffs = []
    for index, (a, b) in enumerate(zip(left, right)):
        fa = _fraction(a, f"{path}.left[{index}]", errors)
        fb = _fraction(b, f"{path}.right[{index}]", errors)
        if fa is not None and fb is not None:
            diffs.append(fa - fb)
    if len(diffs) != expected_clusters:
        return None
    return sum(diffs, Fraction(0)) / expected_clusters


def _validate_replicated_interval(row: Any, left: Any, right: Any,
                                  expected_clusters: int, path: str,
                                  errors: list[str]) -> None:
    """Recompute an interval from its declared paired vectors and plan."""
    observed = _paired_delta(left, right, expected_clusters,
                             f"{path}.vectors", errors)
    interval = _mapping(row, path, errors)
    if observed is None:
        return
    if interval.get("delta") != observed:
        errors.append(f"{path} estimate differs from paired vectors")
        return
    reps, seed = interval.get("resamples"), interval.get("seed")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps <= 0 \
            or not isinstance(seed, int) or isinstance(seed, bool):
        return
    try:
        recomputed = ST.paired_interval(left, right, reps=reps, seed=seed)
    except (TypeError, ValueError) as exc:
        errors.append(f"{path} cannot be recomputed: {exc}")
        return
    if set(interval) != set(recomputed) \
            or any(interval.get(name) != value
                   for name, value in recomputed.items()):
        errors.append(f"{path} differs from exact paired resampling")


def _validate_curve_rows(rows: Any, path: str, errors: list[str],
                         expected_tasks: int | None = None) -> None:
    rows = _mapping(rows, path, errors)
    if any(_budget(rows, q) is None for q in QUERY_BUDGETS):
        errors.append(f"{path} must contain the complete q=0..4 curve")
        return
    tasks_seen = set()
    for q in QUERY_BUDGETS:
        row = _mapping(_budget(rows, q), f"{path}[{q}]", errors)
        tasks = row.get("tasks")
        if not isinstance(tasks, int) or isinstance(tasks, bool) or tasks <= 0:
            errors.append(f"{path}[{q}].tasks must be positive")
            continue
        tasks_seen.add(tasks)
        acc = _fraction(row.get("task_accuracy"),
                        f"{path}[{q}].task_accuracy", errors)
        if acc is not None and not Fraction(0) <= acc <= Fraction(1):
            errors.append(f"{path}[{q}].task_accuracy is outside [0,1]")
        asked = row.get("queries_asked", row.get("queries_actually_asked"))
        if asked is not None:
            if not isinstance(asked, int) or isinstance(asked, bool) \
                    or not 0 <= asked <= tasks * q:
                errors.append(f"{path}[{q}] has impossible asked count")
        if q == 0 and asked not in (None, 0):
            errors.append(f"{path}[0] asks a query")
    if len(tasks_seen) > 1:
        errors.append(f"{path} changes its denominator across budgets")
    if expected_tasks is not None and tasks_seen != {expected_tasks}:
        errors.append(f"{path} denominator disagrees with population")


def _validate_l10(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    if artifact.get("phase") != "X65A-L1":
        errors.append("phase must be X65A-L1")
    if artifact.get("schema") != 1:
        errors.append("L1 schema must be 1")
    prov = _mapping(artifact.get("provenance"), "provenance", errors)
    got = CT.validate_provenance(prov)
    errors.extend(f"provenance: {error}" for error in got.errors)
    if not CT.calibrate_provenance(prov).get("fires"):
        errors.append("provenance same-validator calibration did not fire")
    stored_cal = prov.get("calibration")
    if stored_cal is not None and _mapping(
            stored_cal, "provenance.calibration", errors).get("fires") is not True:
        errors.append("stored provenance calibration did not fire")
    if prov.get("complete") is not None and prov.get("complete") is not True:
        errors.append("provenance.complete is false")
    pre = _mapping(artifact.get("x64h_prerequisite"),
                   "x64h_prerequisite", errors)
    if pre.get("passed") is not True:
        errors.append("X64H prerequisite did not pass")
    if not isinstance(pre.get("checks"), int) or pre.get("checks", 0) <= 0:
        errors.append("X64H prerequisite check count is invalid")
    if pre.get("passed_checks") != pre.get("checks"):
        errors.append("not every X64H prerequisite check passed")
    if artifact.get("claim_boundary") != \
            "controlled authored semantic environment":
        errors.append("claim boundary is missing or broadened")
    unsupported = set(artifact.get("unsupported_scope", ()))
    required_unsupported = {
        "natural language", "lifelong learning", "AGI",
        "bounded total memory",
    }
    if not required_unsupported <= unsupported:
        errors.append("unsupported-scope list is incomplete")
    if artifact.get("x65a_p_started") not in (None, False):
        errors.append("X65A-P was started inside the L1 barrier")
    return _finish(errors)


def _validate_risk_row(row: Any, path: str, errors: list[str]) -> None:
    row = _mapping(row, path, errors)
    _required(row, ("stable_action", "latent_action", "stable_risk",
                    "latent_action_risk_under_stable", "passed",
                    "matched_history", "stable_history", "latent_history",
                    "latent_has_NEW", "latent_has_OUT"), path, errors)
    stable = _fraction(row.get("stable_risk"), f"{path}.stable_risk", errors)
    latent = _fraction(row.get("latent_action_risk_under_stable"),
                       f"{path}.latent_action_risk_under_stable", errors)
    if stable is not None and latent is not None and stable > latent:
        errors.append(f"{path} violates stable risk <= latent risk")
    if row.get("passed") is not True:
        errors.append(f"{path}.passed is false")
    if row.get("matched_history") is not True \
            or row.get("stable_history") != row.get("latent_history"):
        errors.append(f"{path} compares unmatched histories")
    if row.get("latent_has_NEW") is not True \
            or row.get("latent_has_OUT") is not True:
        errors.append(f"{path} omits NEW/OUT from the latent arm")


def _validate_l11(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    for overlap, split, section in _split_rows(
            artifact, "matched_risk", errors):
        path = f"matched_risk.{overlap}.{split}"
        tasks = section.get("tasks")
        if not isinstance(tasks, int) or isinstance(tasks, bool) or tasks <= 0:
            errors.append(f"{path}.tasks must be positive")
            tasks = 0
        if section.get("all_taskwise_pass") is not True:
            errors.append(f"{path}.all_taskwise_pass is false")
        if section.get("taskwise_failures") not in ([], ()):
            errors.append(f"{path} contains taskwise failures")
        rows = section.get("taskwise_rows", ())
        if not isinstance(rows, (list, tuple)) or len(rows) != tasks:
            errors.append(f"{path}.taskwise_rows has the wrong length")
            rows = ()
        for index, task in enumerate(rows):
            task = _mapping(task, f"{path}.taskwise_rows[{index}]", errors)
            for name in RISK_COMPARISONS:
                _validate_risk_row(task.get(name),
                                   f"{path}.taskwise_rows[{index}].{name}",
                                   errors)
            model = _mapping(task.get("model"),
                             f"{path}.taskwise_rows[{index}].model", errors)
            if tuple(model.get("latent_components", ())) != (
                    "stored_records", "NEW_IDENTITY", "OUT_OF_FAMILY"):
                errors.append(f"{path}.taskwise_rows[{index}] model is incomplete")

        arms = _mapping(section.get("arms"), f"{path}.arms", errors)
        if set(arms) != set(RISK_ARMS):
            errors.append(f"{path}.arms is incomplete or unexpected")
        for name in RISK_ARMS:
            arm = _mapping(arms.get(name), f"{path}.arms.{name}", errors)
            _required(arm, ("correct", "tasks", "queries", "accuracy",
                            "mean_queries", "mean_taskwise_bayes_risk"),
                      f"{path}.arms.{name}", errors)
            if arm.get("tasks") != tasks:
                errors.append(f"{path}.arms.{name} denominator mismatch")
            correct, queries = arm.get("correct"), arm.get("queries")
            if tasks and isinstance(correct, int) and not isinstance(correct, bool):
                if arm.get("accuracy") != Fraction(correct, tasks):
                    errors.append(f"{path}.arms.{name} accuracy mismatch")
            else:
                errors.append(f"{path}.arms.{name}.correct is invalid")
            if tasks and isinstance(queries, int) and not isinstance(queries, bool):
                if arm.get("mean_queries") != Fraction(queries, tasks):
                    errors.append(f"{path}.arms.{name} query mean mismatch")
            else:
                errors.append(f"{path}.arms.{name}.queries is invalid")
            _fraction(arm.get("mean_taskwise_bayes_risk"),
                      f"{path}.arms.{name}.mean_taskwise_bayes_risk", errors)
        for stable, latent in (("stable_q0", "latent_q0"),
                               ("stable_q1", "latent_q1"),
                               ("stable_oracle_query",
                                "latent_oracle_query")):
            if stable in arms and latent in arms:
                if arms[stable].get("queries") != arms[latent].get("queries"):
                    errors.append(f"{path} {stable}/{latent} query mismatch")
                sr = arms[stable].get("mean_taskwise_bayes_risk")
                lr = arms[latent].get("mean_taskwise_bayes_risk")
                if _exact(sr) and _exact(lr) and Fraction(sr) > Fraction(lr):
                    errors.append(f"{path} {stable} has larger mean Bayes risk")
        cal = _mapping(section.get("calibration"),
                       f"{path}.calibration", errors)
        if cal.get("fires") is not True \
                or cal.get("unmatched_history_rejected") is not True:
            errors.append(f"{path} unmatched-history calibration did not fire")
    return _finish(errors)


def _validate_l12(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    section = _mapping(artifact.get("sketch_sufficiency"),
                       "sketch_sufficiency", errors)
    if not isinstance(section.get("mathematical_claim"), str) \
            or "q(phi|g)" not in section.get("mathematical_claim", ""):
        errors.append("sketch mathematical claim is missing q(phi|g)")
    if section.get("incomplete_retrieval") is not False:
        errors.append("sketch is marked incomplete")
    proof = _mapping(section.get("proof"), "sketch_sufficiency.proof", errors)
    if proof.get("valid") is not True:
        errors.append("algebraic sufficiency certificate is invalid")
    if proof.get("proof_assistant_verified") is not False:
        errors.append("proof incorrectly claims proof-assistant verification")
    if proof.get("differential_role") != "corroboration only":
        errors.append("differential audit is not scoped as corroboration")
    if proof.get("proof_kind") != "mathematical finite-algebra proof":
        errors.append("proof kind is missing or broadened")
    digest = proof.get("proof_document_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append("proof document digest is invalid")
    obligations = proof.get("obligations", ())
    names = tuple(row.get("name") for row in obligations
                  if isinstance(row, Mapping))
    if set(names) != set(PROOF_OBLIGATIONS) or len(names) != len(set(names)):
        errors.append("proof obligations are incomplete or duplicated")
    for index, row in enumerate(obligations):
        row = _mapping(row, f"sketch_sufficiency.proof.obligations[{index}]",
                       errors)
        if not row.get("equality") or not row.get("reason"):
            errors.append(f"proof obligation {index} lacks equality/reason")
    domains = proof.get("domain_validations", ())
    by_overlap = {row.get("overlap"): row for row in domains
                  if isinstance(row, Mapping)}
    if set(by_overlap) != set(OVERLAPS) or len(domains) != len(OVERLAPS):
        errors.append("proof domain validations do not cover both strata")
    for overlap in OVERLAPS:
        row = _mapping(by_overlap.get(overlap),
                       f"proof.domain_validations.{overlap}", errors)
        checks = _mapping(row.get("checks"),
                          f"proof.domain_validations.{overlap}.checks", errors)
        if not checks or not all(value is True for value in checks.values()) \
                or row.get("passed") is not True \
                or row.get("failed_checks") not in ([], ()):
            errors.append(f"proof domain {overlap} has failed premises")
        for name in ("convention_count", "meaning_count",
                     "legal_grounded_pairs",
                     "persistent_factor_entries_checked"):
            if not isinstance(row.get(name), int) or row.get(name, 0) <= 0:
                errors.append(f"proof domain {overlap}.{name} is invalid")
        if digest and row.get("proof_document_sha256") != digest:
            errors.append(f"proof domain {overlap} has a different proof digest")

    differential = _mapping(section.get("differential"),
                            "sketch_sufficiency.differential", errors)
    weights = _mapping(section.get("selection_aware_weights_nonuniform"),
                       "sketch_sufficiency.selection_aware_weights_nonuniform",
                       errors)
    if set(differential) != set(OVERLAPS) or set(weights) != set(OVERLAPS):
        errors.append("sketch differential evidence must cover both strata")
    for overlap in OVERLAPS:
        row = _mapping(differential.get(overlap),
                       f"sketch_sufficiency.differential.{overlap}", errors)
        if row.get("overlap") != overlap or row.get("passed") is not True:
            errors.append(f"sketch differential {overlap} did not pass")
        for name in ("tasks", "reachable_states", "clarification_answers",
                     "exact_comparisons"):
            if not isinstance(row.get(name), int) or row.get(name, 0) <= 0:
                errors.append(f"sketch differential {overlap}.{name} is invalid")
        mismatches = _mapping(row.get("mismatches"),
                              f"sketch differential {overlap}.mismatches", errors)
        if not mismatches or any(value != 0 for value in mismatches.values()):
            errors.append(f"sketch differential {overlap} has mismatches")
        nonuniform = row.get("selection_weight_nonuniform_states")
        weighted = row.get("weighted_query_differs_from_uniform")
        if not isinstance(nonuniform, int) or nonuniform <= 0 \
                or weights.get(overlap) != nonuniform:
            errors.append(f"selection-aware nonuniform audit failed for {overlap}")
        if not isinstance(weighted, int) or weighted <= 0:
            errors.append(f"weighted query-utility audit failed for {overlap}")
    cal = _mapping(section.get("calibration"),
                   "sketch_sufficiency.calibration", errors)
    if cal.get("fires") is not True \
            or cal.get(
                "same_premise_validator_rejects_hidden_nonindicator_weight") \
                is not True \
            or cal.get(
                "plant_preserves_support_but_changes_exact_mass") is not True \
            or cal.get("nonuniform_weight_case_present") is not True \
            or cal.get("uniform_query_utility_differs") is not True:
        errors.append("sketch sufficiency calibration did not fire")
    countermodels = _mapping(
        section.get("finite_authored_model_countermodels"),
        "sketch_sufficiency.finite_authored_model_countermodels", errors)
    if set(countermodels) != set(OVERLAPS):
        errors.append("proof countermodel calibration must cover both strata")
    for overlap in OVERLAPS:
        row = _mapping(countermodels.get(overlap),
                       f"sketch_sufficiency.countermodels.{overlap}", errors)
        validator = _mapping(
            row.get("same_domain_validator"),
            f"sketch_sufficiency.countermodels.{overlap}.validator", errors)
        if row.get("rejected") is not True \
                or validator.get("passed") is not False \
                or row.get("support_preserved") is not True \
                or row.get("posterior_gap") is not True:
            errors.append(f"{overlap} proof countermodel was not rejected")
        witness = _mapping(
            row.get("witness"),
            f"sketch_sufficiency.countermodels.{overlap}.witness", errors)
        full = _fraction(
            witness.get("full_mass"),
            f"sketch_sufficiency.countermodels.{overlap}.full_mass", errors)
        sketch = _fraction(
            witness.get("sketch_mass"),
            f"sketch_sufficiency.countermodels.{overlap}.sketch_mass", errors)
        if witness.get("support_preserved") is not True \
                or witness.get("posterior_gap") is not True \
                or (full is not None and sketch is not None and full == sketch):
            errors.append(f"{overlap} proof countermodel does not change mass")
    return _finish(errors)


def _validate_accounting_row(row: Any, arm: str, path: str,
                             errors: list[str]) -> tuple[Any, Any]:
    row = _mapping(row, path, errors)
    _required(row, ("stream_seed", "task_id", "condition", "true_slot",
                    "current_utterance", "task_candidate_pool", "protocol",
                    "selected_keys",
                    *RETRIEVAL_ACCOUNTING_FIELDS, "incomplete_retrieval",
                    "within_512", "four_node_claim", "validation"),
              path, errors)
    for name in RETRIEVAL_ACCOUNTING_FIELDS:
        value = row.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{path}.{name} must be a nonnegative integer")
    if row.get("identity_likelihoods_evaluated") != \
            row.get("identity_specific_summaries_inspected"):
        errors.append(f"{path} does not evaluate each inspected identity")
    if row.get("total_retrieval_node_equivalents") != \
            row.get("identity_specific_summaries_inspected"):
        errors.append(f"{path} hides identity summaries inside a node")
    if row.get("full_records_loaded") != 0:
        errors.append(f"{path} loads a full record")
    if row.get("total_retrieval_bytes", 513) > 512 \
            or row.get("within_512") is not True:
        errors.append(f"{path} exceeds or mislabels the 512-byte budget")
    selected = row.get("selected_keys")
    if not isinstance(selected, (list, tuple)) \
            or len(selected) != row.get("shortlist_size") \
            or len(selected) != len(set(selected)):
        errors.append(f"{path} selected keys disagree with shortlist size")
    validation = _mapping(row.get("validation"), f"{path}.validation", errors)
    checks = _mapping(validation.get("checks"),
                      f"{path}.validation.checks", errors)
    if validation.get("passed") is not True or not checks \
            or not all(value is True for value in checks.values()) \
            or validation.get("failed_checks") not in ([], ()):
        errors.append(f"{path} failed its accounting validator")
    if arm != "protocol_B_four_record":
        if row.get("protocol") != "A_GLOBAL_EXACT_SCAN":
            errors.append(f"{path} has the wrong protocol label")
        if row.get("identity_specific_summaries_inspected") != 8 \
                or row.get("total_retrieval_node_equivalents") != 8:
            errors.append(f"{path} must charge all eight record summaries")
        expected_shortlist = 8 if arm == "exact_all_record" else 4
        expected_incomplete = arm != "exact_all_record"
        if row.get("shortlist_size") != expected_shortlist:
            errors.append(f"{path} has the wrong shortlist size for {arm}")
        if row.get("incomplete_retrieval") is not expected_incomplete:
            errors.append(f"{path} misreports shortlist incompleteness")
        if row.get("four_node_claim") is not False:
            errors.append(f"{path} turns a top-four shortlist into a four-node claim")
        if row.get("total_retrieval_bytes") != row.get("index_bytes_scanned"):
            errors.append(f"{path} does not charge its exact index once")
    else:
        if row.get("protocol") != "B_FOUR_RECORD_COARSE_NOMINATION":
            errors.append(f"{path} has the wrong protocol label")
        if row.get("identity_specific_summaries_inspected", 5) > 4 \
                or row.get("total_retrieval_node_equivalents", 5) > 4:
            errors.append(f"{path} exceeds four retrieved records")
        if row.get("shortlist_size") != 4:
            errors.append(f"{path} Protocol B shortlist is not four")
        if row.get("incomplete_retrieval") is not True \
                or row.get("four_node_claim") is not True:
            errors.append(f"{path} hides Protocol B incompleteness")
        if row.get("total_retrieval_bytes") != \
                row.get("index_bytes_scanned", 0) + row.get(
                    "sketch_bytes_loaded", 0):
            errors.append(f"{path} Protocol B does not charge coarse plus loaded bytes")
    return row.get("stream_seed"), row.get("task_id")


def _validate_l13(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    for overlap, split, section in _split_rows(artifact, "retrieval", errors):
        path = f"retrieval.{overlap}.{split}"
        streams = _mapping(section.get("streams"), f"{path}.streams", errors)
        expected_clusters = _seed_count(artifact, split, errors)
        if len(streams) != expected_clusters or not streams:
            errors.append(f"{path}.streams does not match complete-stream seeds")
        for seed, arms in streams.items():
            arms = _mapping(arms, f"{path}.streams.{seed}", errors)
            if set(arms) != set(RETRIEVAL_ARMS):
                errors.append(f"{path}.streams.{seed} has incomplete arms")
            denominators = set()
            for arm_name in RETRIEVAL_ARMS:
                arm = _mapping(arms.get(arm_name),
                               f"{path}.streams.{seed}.{arm_name}", errors)
                curve = arm.get("curve")
                _validate_curve_rows(
                    curve, f"{path}.streams.{seed}.{arm_name}.curve", errors)
                q1 = _mapping(_budget(curve, 1) if isinstance(curve, Mapping)
                              else None,
                              f"{path}.streams.{seed}.{arm_name}.curve[1]",
                              errors)
                if isinstance(q1.get("tasks"), int):
                    denominators.add(q1["tasks"])
                if arm.get("frozen_target_accuracy") != TARGET_ACCURACY:
                    errors.append(f"{path}.{seed}.{arm_name} target changed")
                accuracies = {
                    q: _budget(curve, q).get("task_accuracy")
                    for q in QUERY_BUDGETS
                    if isinstance(curve, Mapping)
                    and isinstance(_budget(curve, q), Mapping)
                }
                expected_q = next((q for q in QUERY_BUDGETS
                                   if _exact(accuracies.get(q))
                                   and Fraction(accuracies[q]) >= TARGET_ACCURACY),
                                  5)
                if arm.get("questions_to_frozen_target_accuracy") != expected_q:
                    errors.append(f"{path}.{seed}.{arm_name} target metric mismatch")
            if len(denominators) > 1:
                errors.append(f"{path}.streams.{seed} arm denominators differ")

        per_task = _mapping(section.get("per_task_accounting"),
                            f"{path}.per_task_accounting", errors)
        if set(per_task) != set(RETRIEVAL_ACCOUNTING_ARMS):
            errors.append(f"{path} lacks per-task accounting for every retrieval arm")
        identifiers = {}
        rows_by_arm: dict[str, dict[tuple[Any, Any], Mapping]] = {}
        for arm_name in RETRIEVAL_ACCOUNTING_ARMS:
            rows = per_task.get(arm_name, ())
            if not isinstance(rows, (list, tuple)) or not rows:
                errors.append(f"{path}.{arm_name} has no per-task rows")
                rows = ()
            ids = []
            indexed = {}
            for index, row in enumerate(rows):
                key = _validate_accounting_row(
                    row, arm_name,
                    f"{path}.per_task_accounting.{arm_name}[{index}]", errors)
                ids.append(key)
                if isinstance(row, Mapping):
                    indexed[key] = row
            if len(ids) != len(set(ids)):
                errors.append(f"{path}.{arm_name} task identifiers are duplicated")
            identifiers[arm_name] = set(ids)
            rows_by_arm[arm_name] = indexed
        identifier_sets = list(identifiers.values())
        if not identifier_sets or any(rows != identifier_sets[0]
                                      for rows in identifier_sets[1:]):
            errors.append(f"{path} retrieval arms do not cover the same tasks")
        matched_fields = RETRIEVAL_ACCOUNTING_FIELDS + (
            "protocol", "incomplete_retrieval", "within_512",
            "four_node_claim",
        )
        for key in identifiers.get("MAIN_protocol_A", ()):
            main_row = rows_by_arm.get("MAIN_protocol_A", {}).get(key, {})
            for control in RESOURCE_MATCHED_RETRIEVAL_CONTROLS:
                control_row = rows_by_arm.get(control, {}).get(key, {})
                if any(main_row.get(field) != control_row.get(field)
                       for field in matched_fields):
                    errors.append(
                        f"{path}.{control} is not resource-matched for task {key}")

        pa = _mapping(section.get("protocol_A"), f"{path}.protocol_A", errors)
        pb = _mapping(section.get("protocol_B"), f"{path}.protocol_B", errors)
        if pa.get("all_within_512") is not True \
                or pa.get("no_four_node_claim") is not True \
                or pa.get("all_main_rows_report_incomplete") is not True \
                or pa.get("all_rows_validated") is not True:
            errors.append(f"{path} Protocol A summary is invalid")
        if pb.get("all_within_512") is not True \
                or pb.get("all_at_most_four") is not True \
                or pb.get("all_incomplete_reported") is not True \
                or pb.get("all_rows_validated") is not True:
            errors.append(f"{path} Protocol B summary is invalid")
        for label, summary in (("protocol_A", pa), ("protocol_B", pb)):
            extrema = _mapping(summary.get("accounting_extrema"),
                               f"{path}.{label}.accounting_extrema", errors)
            if set(extrema) != set(RETRIEVAL_ACCOUNTING_FIELDS):
                errors.append(f"{path}.{label} accounting extrema are incomplete")
            for field in RETRIEVAL_ACCOUNTING_FIELDS:
                bounds = _mapping(extrema.get(field),
                                  f"{path}.{label}.{field}", errors)
                lo, hi = bounds.get("min"), bounds.get("max")
                if not isinstance(lo, int) or isinstance(lo, bool) \
                        or not isinstance(hi, int) or isinstance(hi, bool) \
                        or lo < 0 or lo > hi:
                    errors.append(f"{path}.{label}.{field} extrema are invalid")
        if section.get("main_protocol") != "A_GLOBAL_EXACT_SCAN_TOP4":
            errors.append(f"{path} main protocol was not validation-frozen")
        freeze = _mapping(section.get("main_protocol_freeze"),
                          f"{path}.main_protocol_freeze", errors)
        record = _mapping(freeze.get("record"),
                          f"{path}.main_protocol_freeze.record", errors)
        expected_freeze = {
            "protocol": "A_GLOBAL_EXACT_SCAN_TOP4",
            "ranking": "exact_likelihood",
            "shortlist_size": 4,
            "physical_node_equivalents": 8,
            "byte_limit": 512,
            "incomplete_retrieval": True,
            "frozen_from": "development design; original four-record intent",
            "development_seeds": tuple(
                _mapping(artifact.get("provenance"), "provenance", errors)
                .get("development_stream_seeds", ())),
            "validation_seeds_used_to_choose": (),
        }
        freeze_keys = set(expected_freeze)
        normalized_record = dict(record)
        for name in ("development_seeds",
                     "validation_seeds_used_to_choose"):
            value = normalized_record.get(name)
            if isinstance(value, (list, tuple)):
                normalized_record[name] = tuple(value)
        if set(record) != freeze_keys or normalized_record != expected_freeze:
            errors.append(f"{path} executable retrieval freeze record differs")
        if freeze.get("sha256") != _canon_sha(record):
            errors.append(f"{path} executable retrieval freeze SHA256 differs")
        claim = str(section.get("main_claim", ""))
        if not all(term in claim for term in
                   ("approximate", "global exact-sketch scan", "eight node-",
                    "not four-node")):
            errors.append(f"{path} main claim hides the node limitation")
        if section.get("resource_matched_controls") is not True:
            errors.append(f"{path} retrieval controls are not resource-matched")
        if not isinstance(section.get("coarse_nonsufficiency_collisions"), int) \
                or isinstance(section.get("coarse_nonsufficiency_collisions"), bool) \
                or section.get("coarse_nonsufficiency_collisions", 0) <= 0:
            errors.append(f"{path} Protocol B nonsufficiency is not witnessed")
        if section.get("no_memory_baseline") != (
                "validation-frozen fresh family-prior exact task-information "
                "policy over legal behavioral and semantic questions"):
            errors.append(f"{path} no-memory query baseline is under-specified")
        active = _mapping(section.get("active_bytes"),
                          f"{path}.active_bytes", errors)
        if active.get("all_within_4KiB") is not True \
                or not isinstance(active.get("max"), int) \
                or active.get("max", 4097) > 4096:
            errors.append(f"{path} exceeds the 4-KiB active budget")
        cal = _mapping(section.get("calibration"),
                       f"{path}.calibration", errors)
        if cal.get("fires") is not True \
                or cal.get("coarse_collision_fired") is not True \
                or cal.get("valid_rows_accepted_by_accounting_validator") is not True \
                or cal.get(
                    "free_exact_index_undercharge_rejected_by_same_validator") is not True:
            errors.append(f"{path} retrieval accounting calibration did not fire")
    return _finish(errors)


def _validate_l14(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    accounting = _mapping(artifact.get("query_accounting"),
                          "query_accounting", errors)
    got = CT.validate_q246(accounting)
    errors.extend(f"query_accounting: {error}" for error in got.errors)
    if not CT.calibrate_q246(accounting).get("fires"):
        errors.append("q=2.46 same-validator calibration did not fire")
    if accounting.get("published_curve_reproduced") is not True \
            or accounting.get("metrics_internally_consistent") is not True:
        errors.append("legacy curve/accounting flags are false")
    cal = _mapping(accounting.get("calibration"),
                   "query_accounting.calibration", errors)
    if cal.get("fires") is not True \
            or cal.get("old_zero_query_label_rejected") is not True:
        errors.append("legacy query calibration did not fire")
    budgets = _mapping(accounting.get("budgets"),
                       "query_accounting.budgets", errors)
    tasks = accounting.get("tasks")
    for q in (0, 1, 3):
        row = _mapping(_budget(budgets, q),
                       f"query_accounting.budgets[{q}]", errors)
        _required(row, ("query_budget", "queries_offered",
                        "queries_actually_asked", "mean_over_all_tasks",
                        "mean_over_ambiguous_tasks",
                        "mean_over_scored_returning_tasks",
                        "total_per_stream", "task_accuracy",
                        "metric_denominator", "query_type"),
                  f"query_accounting.budgets[{q}]", errors)
        asked = row.get("queries_actually_asked")
        if row.get("query_budget") != q:
            errors.append(f"legacy q={q} budget label is wrong")
        if not isinstance(asked, int) or isinstance(asked, bool) or asked < 0:
            errors.append(f"legacy q={q} asked count is invalid")
        elif isinstance(tasks, int) and row.get("mean_over_all_tasks") != \
                Fraction(asked, tasks):
            errors.append(f"legacy q={q} all-task mean is inconsistent")
        totals = row.get("total_per_stream", ())
        if isinstance(totals, (list, tuple)) and isinstance(asked, int) \
                and sum(totals) != asked:
            errors.append(f"legacy q={q} stream totals do not sum to asked")
        types = _mapping(row.get("query_type"),
                         f"query_accounting.budgets[{q}].query_type", errors)
        for name in ("semantic", "identity", "convention", "task", "cause"):
            if not isinstance(types.get(name), int) or types.get(name, -1) < 0:
                errors.append(f"legacy q={q} missing {name} query accounting")
        if row.get("metric_denominator") != \
                "returning+ambiguous+misleading":
            errors.append(f"legacy q={q} denominator is mislabeled")
    return _finish(errors)


def _effect_rows(row: Mapping, path: str, errors: list[str]) -> dict[str, tuple]:
    records = row.get("per_question_resolution_effects", ())
    if not isinstance(records, (list, tuple)):
        errors.append(f"{path}.per_question_resolution_effects must be a sequence")
        return {}
    out: dict[str, tuple] = {}
    for index, record in enumerate(records):
        record = _mapping(record, f"{path}.effects[{index}]", errors)
        digest = record.get("task_digest")
        effects = record.get("effects")
        if not isinstance(digest, str) or not digest:
            errors.append(f"{path}.effects[{index}] has no task digest")
            continue
        if digest in out:
            errors.append(f"{path} duplicates task digest {digest}")
        if not isinstance(effects, (list, tuple)):
            errors.append(f"{path}.effects[{index}] is not a sequence")
            effects = ()
        for effect_index, effect in enumerate(effects):
            effect = _mapping(
                effect, f"{path}.effects[{index}][{effect_index}]", errors)
            _required(effect, ("event", "changed", "support",
                               "resolved_quantities"),
                      f"{path}.effects[{index}][{effect_index}]", errors)
            changed = _mapping(effect.get("changed"),
                               f"{path}.effect.changed", errors)
            if set(changed) != {"identity", "convention", "task", "cause"} \
                    or not all(isinstance(v, bool) for v in changed.values()):
                errors.append(f"{path} effect has incomplete change flags")
            resolved = tuple(effect.get("resolved_quantities", ()))
            expected = tuple(name for name in
                             ("identity", "convention", "task", "cause")
                             if changed.get(name) is True)
            if resolved != expected:
                errors.append(f"{path} effect resolution labels are not derived")
        out[digest] = tuple(effects)
    return out


def _validate_l15(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    for overlap, split, section in _split_rows(artifact, "memoryless", errors):
        path = f"memoryless.{overlap}.{split}"
        tasks = section.get("tasks")
        if not isinstance(tasks, int) or isinstance(tasks, bool) or tasks <= 0:
            errors.append(f"{path}.tasks must be positive")
            tasks = 0
        if section.get("population") != "all_matched_scored":
            errors.append(f"{path} uses a positional/partial population")
        policies = _mapping(section.get("policies"), f"{path}.policies", errors)
        if set(policies) != set(MEMORYLESS_POLICIES):
            errors.append(f"{path} does not contain exactly seven policies")
        for policy in MEMORYLESS_POLICIES:
            curve = _mapping(policies.get(policy),
                             f"{path}.policies.{policy}", errors)
            _validate_curve_rows(curve, f"{path}.policies.{policy}", errors,
                                 tasks)
            previous: dict[str, tuple] = {}
            for q in QUERY_BUDGETS:
                row = _mapping(_budget(curve, q),
                               f"{path}.policies.{policy}[{q}]", errors)
                if row.get("query_budget") != q:
                    errors.append(f"{path}.{policy}[{q}] budget label is wrong")
                if row.get("answers_applied") is not True:
                    errors.append(f"{path}.{policy}[{q}] has unapplied answers")
                if row.get("latent_quantity") != LATENT_QUANTITIES[policy]:
                    errors.append(f"{path}.{policy}[{q}] latent quantity is wrong")
                asked = row.get("queries_actually_asked")
                offered = row.get("queries_offered")
                if not isinstance(asked, int) or isinstance(asked, bool) \
                        or asked < 0:
                    errors.append(f"{path}.{policy}[{q}] asked count is invalid")
                    asked = 0
                if not isinstance(offered, int) or isinstance(offered, bool) \
                        or offered < asked:
                    errors.append(f"{path}.{policy}[{q}] offered count is invalid")
                query_types = _mapping(row.get("query_types"),
                                       f"{path}.{policy}[{q}].query_types", errors)
                if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                       for v in query_types.values()) \
                        or sum(query_types.values()) != asked:
                    errors.append(f"{path}.{policy}[{q}] query types do not sum")
                effects = _effect_rows(row, f"{path}.{policy}[{q}]", errors)
                if len(effects) != tasks:
                    errors.append(f"{path}.{policy}[{q}] effect population mismatch")
                if sum(len(values) for values in effects.values()) != asked:
                    errors.append(f"{path}.{policy}[{q}] asked/effect counts differ")
                if q == 0 and any(effects.values()):
                    errors.append(f"{path}.{policy}[0] has a post-query effect")
                if previous and set(previous) != set(effects):
                    errors.append(f"{path}.{policy} changes population across q")
                for digest, prior in previous.items():
                    current = effects.get(digest, ())
                    if tuple(current[:len(prior)]) != tuple(prior):
                        errors.append(
                            f"{path}.{policy} is not prefix-consistent for {digest}")
                previous = effects
                for entropy_name in ("convention_entropy", "task_entropy"):
                    entropy = _mapping(row.get(entropy_name),
                                       f"{path}.{policy}[{q}].{entropy_name}",
                                       errors)
                    if not all(isinstance(entropy.get(k), str)
                               for k in ("before", "after")):
                        errors.append(f"{path}.{policy}[{q}] entropy is incomplete")
                classes = _mapping(row.get("candidate_class_count"),
                                   f"{path}.{policy}[{q}].candidate_class_count",
                                   errors)
                _fraction(classes.get("before"),
                          f"{path}.{policy}[{q}].candidate.before", errors)
                _fraction(classes.get("after"),
                          f"{path}.{policy}[{q}].candidate.after", errors)
                resolved = _mapping(row.get("resolved_latent_quantities"),
                                    f"{path}.{policy}[{q}].resolved", errors)
                if set(resolved) != {"identity", "convention", "task", "cause"} \
                        or any(not isinstance(v, int) or isinstance(v, bool)
                               or v < 0 for v in resolved.values()):
                    errors.append(f"{path}.{policy}[{q}] resolved counts are invalid")
                else:
                    derived = {name: 0 for name in
                               ("identity", "convention", "task", "cause")}
                    for task_effects in effects.values():
                        for effect in task_effects:
                            for name in effect.get("resolved_quantities", ()):
                                if name in derived:
                                    derived[name] += 1
                    if dict(resolved) != derived:
                        errors.append(
                            f"{path}.{policy}[{q}] resolved counts are not "
                            "derived from per-question effects")
        if section.get("oracle_legal_query_improves") is not True:
            errors.append(f"{path} oracle legal query does not improve")
        for policy in ("no_memory_exact_task_information_gain",
                       "no_memory_exact_convention_task_information_gain"):
            curve = policies.get(policy, {})
            q0, q4 = _budget(curve, 0), _budget(curve, 4)
            if not isinstance(q0, Mapping) or not isinstance(q4, Mapping) \
                    or not _exact(q0.get("task_accuracy")) \
                    or not _exact(q4.get("task_accuracy")) \
                    or Fraction(q4["task_accuracy"]) <= Fraction(
                        q0["task_accuracy"]):
                errors.append(f"{path} {policy} does not improve from q=0 to q=4")
        if section.get("nonoracle_policy_improves") is not True:
            errors.append(f"{path} non-oracle improvement summary is false")
        if section.get("fresh_equals_stable_fresh") is not True:
            errors.append(f"{path} fresh stable-ID control is mismatched")
        if section.get("all_answers_applied") is not True:
            errors.append(f"{path} counted unapplied clarification answers")
        fresh = policies.get("fresh_x64h_family_prior", {})
        stable = policies.get("stable_id_fresh_no_posterior", {})
        for q in QUERY_BUDGETS:
            fr, sr = _budget(fresh, q), _budget(stable, q)
            if isinstance(fr, Mapping) and isinstance(sr, Mapping) \
                    and (fr.get("task_accuracy") != sr.get("task_accuracy")
                         or fr.get("queries_actually_asked") !=
                         sr.get("queries_actually_asked")):
                errors.append(f"{path} fresh controls differ at q={q}")
        cal = _mapping(section.get("calibration"),
                       f"{path}.calibration", errors)
        if cal.get("fires") is not True or cal.get("valid_input") is not True \
                or not cal.get("rejected") \
                or not all(cal.get("rejected", {}).values()):
            errors.append(f"{path} answer-application calibration did not fire")
    return _finish(errors)


def _validate_aggregate_curve(curve: Any, path: str,
                              errors: list[str]) -> None:
    curve = _mapping(curve, path, errors)
    if curve.get("frozen_target_accuracy") != TARGET_ACCURACY:
        errors.append(f"{path} target changed")
    if curve.get("prefix_consistent") is not True:
        errors.append(f"{path} is not prefix-consistent")
    tasks = curve.get("population_tasks")
    if not isinstance(tasks, int) or isinstance(tasks, bool) or tasks <= 0:
        errors.append(f"{path} population is empty")
        tasks = None
    rows = curve.get("budgets")
    _validate_curve_rows(rows, f"{path}.budgets", errors, tasks)
    if isinstance(rows, Mapping):
        expected = next((q for q in QUERY_BUDGETS
                         if isinstance(_budget(rows, q), Mapping)
                         and _exact(_budget(rows, q).get("task_accuracy"))
                         and Fraction(_budget(rows, q)["task_accuracy"])
                         >= TARGET_ACCURACY), None)
        if curve.get("minimum_questions_to_frozen_target") != expected:
            errors.append(f"{path} minimum target budget is inconsistent")


def _validate_l16(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    for overlap, split, section in _split_rows(
            artifact, "active_query", errors):
        path = f"active_query.{overlap}.{split}"
        expected = _seed_count(artifact, split, errors)
        got = CT.validate_active_intervals(section, expected)
        errors.extend(f"{path}: {error}" for error in got.errors)
        if not CT.calibrate_active_intervals(section, expected).get("fires"):
            errors.append(f"{path} same-validator interval calibration failed")
        if section.get("frozen_target_accuracy") != TARGET_ACCURACY \
                or section.get("not_reached_censor_value") != 5:
            errors.append(f"{path} changed target or censor semantics")
        if "aggregate" not in str(section.get("questions_metric_definition", "")):
            errors.append(f"{path} uses a truth-aware/per-task question metric")
        streams = _mapping(section.get("streams"), f"{path}.streams", errors)
        for seed, arms in streams.items():
            arms = _mapping(arms, f"{path}.streams.{seed}", errors)
            if set(arms) != {"information_gain", "random"}:
                errors.append(f"{path}.streams.{seed} does not contain paired arms")
            for arm in ("information_gain", "random"):
                row = _mapping(arms.get(arm),
                               f"{path}.streams.{seed}.{arm}", errors)
                _validate_aggregate_curve(
                    row.get("accuracy_curve"),
                    f"{path}.streams.{seed}.{arm}.accuracy_curve", errors)
                expected_q = _mapping(row.get("accuracy_curve"),
                                      f"{path}.{seed}.{arm}.curve", errors).get(
                                          "minimum_questions_to_frozen_target")
                censored = expected_q if expected_q is not None else 5
                if row.get("questions_at_matched_accuracy") != censored:
                    errors.append(f"{path}.{seed}.{arm} target questions mismatch")
        cal = _mapping(section.get("calibration"),
                       f"{path}.calibration", errors)
        if cal.get("fires") is not True or cal.get("valid_input") is not True \
                or not cal.get("rejected") \
                or not all(cal.get("rejected", {}).values()):
            errors.append(f"{path} stored interval calibration did not fire")
    return _finish(errors)


def _validate_l17(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    for overlap, split, section in _split_rows(
            artifact, "new_identity", errors):
        path = f"new_identity.{overlap}.{split}"
        streams = section.get("streams")
        if streams != _seed_count(artifact, split, errors) or not streams:
            errors.append(f"{path} does not cover every complete stream")
            streams = 0
        exact_expected = {
            "precision": Fraction(1), "recall": Fraction(1),
            "false_new_rate_returning": Fraction(0),
            "forced_assimilation_rate": Fraction(0),
            "unresolved_new_rate": Fraction(0),
        }
        for name, expected in exact_expected.items():
            value = _fraction(section.get(name), f"{path}.{name}", errors)
            if value is not None and value != expected:
                errors.append(f"{path}.{name} expected {expected}, got {value}")
        questions = _fraction(section.get("questions_to_grounded_creation"),
                              f"{path}.questions_to_grounded_creation", errors)
        if questions is not None and questions <= 0:
            errors.append(f"{path} creates records without grounding questions")
        for name in ("successfully_promoted_new_records",
                     "later_reuse_of_new_records"):
            if section.get(name) != streams:
                errors.append(f"{path}.{name} does not equal stream count")
        if section.get("contamination_during_creation") != 0:
            errors.append(f"{path} contaminated an established record")
        bytes_added = section.get("record_bytes_added", ())
        if not isinstance(bytes_added, (list, tuple)) \
                or len(bytes_added) != streams \
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                       for v in bytes_added):
            errors.append(f"{path} record-byte additions are incomplete")
        arms = _mapping(section.get("arms"), f"{path}.arms", errors)
        if set(arms) != set(NEW_CONTROL_ARMS):
            errors.append(f"{path} NEW control arms are incomplete")
        if isinstance(arms.get("always_reuse_nearest"), Mapping) \
                and arms["always_reuse_nearest"].get(
                    "forced_assimilation_rate") != 1:
            errors.append(f"{path} always-reuse calibration did not force assimilation")
        if isinstance(arms.get("no_new_unresolved"), Mapping) \
                and arms["no_new_unresolved"].get("recall") != 0:
            errors.append(f"{path} unresolved control is counted as recall")
        if isinstance(arms.get("no_new_forced"), Mapping) \
                and arms["no_new_forced"].get("forced_assimilation_rate") != 1:
            errors.append(f"{path} no-NEW forced arm did not fire")
        if isinstance(arms.get("always_create_new"), Mapping) \
                and arms["always_create_new"].get("returning_false_new", 0) <= 0:
            errors.append(f"{path} always-create false-new arm did not fire")
        oracle = arms.get("oracle_new_returning_status")
        if isinstance(oracle, Mapping) and (oracle.get("precision") != 1
                                            or oracle.get("recall") != 1):
            errors.append(f"{path} oracle new/returning control failed")
        active = _mapping(section.get("active_bytes"),
                          f"{path}.active_bytes", errors)
        if active.get("all_within_4KiB") is not True \
                or not isinstance(active.get("max"), int) \
                or active.get("max", 4097) > 4096:
            errors.append(f"{path} promoted record exceeds active budget")
        rows = section.get("stream_rows", ())
        if not isinstance(rows, (list, tuple)) or len(rows) != streams:
            errors.append(f"{path}.stream_rows has the wrong length")
        else:
            for index, row in enumerate(rows):
                row = _mapping(row, f"{path}.stream_rows[{index}]", errors)
                if row.get("constructible") is not True \
                        or row.get("main_classification_gate") is not True \
                        or row.get("contamination_during_creation") != 0 \
                        or row.get("successfully_promoted_new_records") != 1 \
                        or row.get("later_reuse_of_new_records") != 1:
                    errors.append(f"{path}.stream_rows[{index}] failed creation/reuse")
                fired = _mapping(row.get("calibration_fired"),
                                 f"{path}.stream_rows[{index}].calibration", errors)
                if not fired or not all(value is True for value in fired.values()):
                    errors.append(f"{path}.stream_rows[{index}] calibration failed")
        cal = _mapping(section.get("calibration"),
                       f"{path}.calibration", errors)
        if cal.get("fires") is not True:
            errors.append(f"{path} NEW calibration did not fire")
    return _finish(errors)


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _validate_l18(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    section = _mapping(artifact.get("scope_audit"), "scope_audit", errors)
    if set(section) != set(OVERLAPS):
        errors.append("scope audit must contain exactly both strata")
    for overlap in OVERLAPS:
        row = _mapping(section.get(overlap), f"scope_audit.{overlap}", errors)
        construction = _mapping(row.get("constructibility"),
                                f"scope_audit.{overlap}.constructibility", errors)
        required = ("out_of_family_convention",
                    "out_of_family_transfer_utterance",
                    "out_of_family_grounded_event", "UNKNOWN_MEANING",
                    "MISSING_REPRESENTATION",
                    "restricted_query_indistinguishable")
        _required(construction, required,
                  f"scope_audit.{overlap}.constructibility", errors)
        convention = _mapping(construction.get("out_of_family_convention"),
                              f"scope_audit.{overlap}.convention", errors)
        if convention.get("constructible") is not True \
                or convention.get("family_membership_count") != 0 \
                or not isinstance(convention.get("convention"), Mapping) \
                or convention.get("tested_via") != \
                "authored noninjective role map, then grounded":
            errors.append(f"{overlap} out-of-family convention is vacuous")
        contradiction = _mapping(convention.get("grounded_contradiction"),
                                 f"scope_audit.{overlap}.contradiction", errors)
        if not contradiction.get("events") \
                or contradiction.get("zero_survivors") is not True:
            errors.append(f"{overlap} convention has no grounded contradiction")
        transfer = _mapping(construction.get("out_of_family_transfer_utterance"),
                            f"scope_audit.{overlap}.transfer", errors)
        if overlap == "shared":
            if transfer.get("constructible") is not False \
                    or transfer.get("utterance") is not None \
                    or transfer.get("scope") != \
                    "untestable in frozen shared two-token alphabet":
                errors.append("shared transfer OOF case must be explicitly untestable")
        elif transfer.get("constructible") is not True \
                or transfer.get("zero_family_likelihood") is not True \
                or transfer.get("utterance") is None:
            errors.append("disjoint transfer OOF case is not nonvacuous")
        grounded = _mapping(construction.get("out_of_family_grounded_event"),
                            f"scope_audit.{overlap}.grounded", errors)
        if grounded.get("constructible") is not True \
                or grounded.get("zero_survivors") is not True \
                or not grounded.get("events") \
                or not isinstance(grounded.get("minimum_event_count"), int) \
                or grounded.get("minimum_event_count", 0) <= 0 \
                or grounded.get("source") != \
                "authored_out_of_family_convention":
            errors.append(f"{overlap} grounded OOF event is vacuous")
        if grounded.get("events") != contradiction.get("events"):
            errors.append(f"{overlap} grounded OOF event differs from its convention")
        unknown = _mapping(construction.get("UNKNOWN_MEANING"),
                           f"scope_audit.{overlap}.UNKNOWN_MEANING", errors)
        if unknown.get("constructible") is not True \
                or unknown.get("derived_live_count") != 0 \
                or not unknown.get("demonstrations"):
            errors.append(f"{overlap} UNKNOWN_MEANING case is vacuous")
        missing = _mapping(construction.get("MISSING_REPRESENTATION"),
                           f"scope_audit.{overlap}.MISSING", errors)
        if missing.get("constructible") is not True \
                or missing.get("tested") is not True \
                or missing.get("outcome") != "MISSING_REPRESENTATION":
            errors.append(f"{overlap} MISSING_REPRESENTATION was not observed")
        cause = _mapping(missing.get("cause_posterior"),
                         f"scope_audit.{overlap}.cause", errors)
        if cause.get("MISSING_REPRESENTATION") != 1:
            errors.append(f"{overlap} MISSING cause posterior is not exact")
        cause_total = Fraction(0)
        for name, mass in cause.items():
            value = _fraction(mass, f"scope_audit.{overlap}.cause.{name}",
                              errors)
            if value is not None:
                cause_total += value
        if cause and cause_total != 1:
            errors.append(f"{overlap} MISSING cause posterior is not normalized")
        restricted = _mapping(
            construction.get("restricted_query_indistinguishable"),
            f"scope_audit.{overlap}.restricted", errors)
        if restricted.get("constructible") is not True \
                or not isinstance(restricted.get("case"), Mapping):
            errors.append(f"{overlap} restricted indistinguishable case is absent")
        restricted_case = _mapping(
            restricted.get("case"), f"scope_audit.{overlap}.restricted.case",
            errors)
        query_set = tuple(restricted_case.get("query_set", ()))
        if restricted_case.get("restricted_equal") is not True \
                or restricted_case.get("globally_equal") is not False \
                or not query_set \
                or restricted_case.get("outside_witness") in set(query_set):
            errors.append(f"{overlap} restricted case has no global distinguisher")

        scope = _mapping(row.get("restricted_scope"),
                         f"scope_audit.{overlap}.restricted_scope", errors)
        if scope.get("constructible") is not True \
                or not isinstance(scope.get("promotions"), int) \
                or scope.get("promotions", 0) <= 0:
            errors.append(f"{overlap} restricted scope promotion is absent")
        if scope.get("false_global_promotions") != 0:
            errors.append(f"{overlap} contains false global promotions")
        if scope.get("calibration_false_global_promotions") != 1:
            errors.append(f"{overlap} false-global calibration did not fire")
        if scope.get("case") != restricted_case:
            errors.append(f"{overlap} scoped promotion uses a different challenge case")
        promoted = _mapping(scope.get("record"),
                            f"scope_audit.{overlap}.record", errors)
        verification = _mapping(promoted.get("scope"),
                                f"scope_audit.{overlap}.record.scope", errors)
        _required(verification, ("challenge_universe_digest",
                                 "query_set_digest", "validity_scope",
                                 "status"),
                  f"scope_audit.{overlap}.record.scope", errors)
        if not _valid_digest(verification.get("challenge_universe_digest")) \
                or not _valid_digest(verification.get("query_set_digest")):
            errors.append(f"{overlap} promoted scope digests are invalid")
        if verification.get("validity_scope") != \
                "controlled authored X64H semantic family" \
                or verification.get("status") != "empirical":
            errors.append(f"{overlap} restricted promotion is falsely global")
        validation = _mapping(scope.get("validation"),
                              f"scope_audit.{overlap}.validation", errors)
        validation_checks = _mapping(
            validation.get("checks"),
            f"scope_audit.{overlap}.validation.checks", errors)
        if validation.get("passed") is not True or not validation_checks \
                or not all(value is True for value in validation_checks.values()):
            errors.append(f"{overlap} scoped promotion failed its validator")
        calibration = _mapping(scope.get("calibration"),
                               f"scope_audit.{overlap}.calibration", errors)
        planted_validation = _mapping(
            calibration.get("same_validator"),
            f"scope_audit.{overlap}.calibration.same_validator", errors)
        planted_checks = _mapping(
            planted_validation.get("checks"),
            f"scope_audit.{overlap}.calibration.same_validator.checks",
            errors)
        if calibration.get("fires") is not True \
                or not isinstance(calibration.get("plant"), Mapping) \
                or planted_validation.get("passed") is not False \
                or not planted_checks \
                or all(value is True for value in planted_checks.values()):
            errors.append(
                f"{overlap} scope plant was not rejected by the same validator")
    return _finish(errors)


def _validate_l19(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    prov = _mapping(artifact.get("provenance"), "provenance", errors)
    margins = _mapping(prov.get("validation_frozen_margins"),
                       "provenance.validation_frozen_margins", errors)
    margin = _fraction(margins.get("L10_negative_transfer"),
                       "L10_negative_transfer margin", errors)
    for overlap, split, section in _split_rows(
            artifact, "negative_transfer", errors):
        path = f"negative_transfer.{overlap}.{split}"
        if margin is not None and section.get("frozen_margin") != margin:
            errors.append(f"{path} did not use the frozen margin")
        if section.get("all_conditions_reported") is not True \
                or section.get("main_noninferior") is not True:
            errors.append(f"{path} headline negative-transfer gate failed")
        if section.get("matched_protocol") is not True:
            errors.append(f"{path} aggregate protocol match is false")
        if section.get("noninferiority_failures") not in ([], ()):
            errors.append(f"{path} contains noninferiority failures")
        if section.get("main_established_corruption") != 0:
            errors.append(f"{path} corrupts an established record")
        conditions = _mapping(section.get("conditions"),
                              f"{path}.conditions", errors)
        # Canonical JSON serialization sorts mapping keys, so presence is
        # checked as a set; ``all_conditions_reported`` was computed before
        # serialization and separately attests the preregistered order.
        if set(conditions) != set(NEGATIVE_CONDITIONS):
            errors.append(f"{path} does not report all eight ordered conditions")
        for name in NEGATIVE_CONDITIONS:
            row = _mapping(conditions.get(name),
                           f"{path}.conditions.{name}", errors)
            _required(row, ("constructible_transfer", "tested_via",
                            "task_accuracy", "accuracy_delta",
                            "excess_questions", "false_confident_actions",
                            "established_record_corruption",
                            "provisional_branches", "unresolved_outcomes",
                            "noninferior", "matched_protocol"),
                      f"{path}.conditions.{name}", errors)
            if row.get("matched_protocol") is not True \
                    or row.get("noninferior") is not True:
                errors.append(f"{path}.{name} is unmatched or inferior")
            if row.get("established_record_corruption") != 0:
                errors.append(f"{path}.{name} corrupts established memory")
            false_confident = row.get("false_confident_actions")
            if not isinstance(false_confident, int) \
                    or isinstance(false_confident, bool) \
                    or false_confident < 0:
                errors.append(f"{path}.{name} false-confident count is invalid")
            accuracy = _mapping(row.get("task_accuracy"),
                                f"{path}.{name}.task_accuracy", errors)
            main, no_memory = accuracy.get("MAIN"), accuracy.get("no_memory")
            if (main is None) != (no_memory is None):
                errors.append(f"{path}.{name} reports only one matched arm")
            if main is not None:
                main_f = _fraction(main, f"{path}.{name}.MAIN", errors)
                base_f = _fraction(no_memory, f"{path}.{name}.no_memory", errors)
                delta = _fraction(row.get("accuracy_delta"),
                                  f"{path}.{name}.accuracy_delta", errors)
                if main_f is not None and base_f is not None and delta is not None:
                    if delta != main_f - base_f:
                        errors.append(f"{path}.{name} accuracy delta is wrong")
                    if margin is not None and delta < -margin:
                        errors.append(f"{path}.{name} violates noninferiority")
            elif row.get("accuracy_delta") is not None \
                    or row.get("constructible_transfer") is not False:
                errors.append(f"{path}.{name} hides an unscored constructible case")
            if not row.get("tested_via"):
                errors.append(f"{path}.{name} does not state how it was tested")
        rows = section.get("stream_rows", ())
        expected = _seed_count(artifact, split, errors)
        if not isinstance(rows, (list, tuple)) or len(rows) != expected:
            errors.append(f"{path}.stream_rows has the wrong cluster count")
        else:
            for index, row in enumerate(rows):
                row = _mapping(row, f"{path}.stream_rows[{index}]", errors)
                if row.get("phase") != "X65A-L1-negative-transfer" \
                        or row.get("overlap") != overlap \
                        or row.get("query_budget") != 1 \
                        or row.get("metric_denominator") != \
                        "all_tasks_per_constructible_condition":
                    errors.append(f"{path}.stream_rows[{index}] protocol header differs")
                gates = _mapping(row.get("gates"),
                                 f"{path}.stream_rows[{index}].gates", errors)
                if not gates or not all(value is True for value in gates.values()):
                    errors.append(f"{path}.stream_rows[{index}] has a failed gate")
                calibrations = _mapping(
                    row.get("calibrations"),
                    f"{path}.stream_rows[{index}].calibrations", errors)
                if calibrations.get("all_fire") is not True:
                    errors.append(f"{path}.stream_rows[{index}] calibration failed")
                condition_rows = row.get("conditions", ())
                if not isinstance(condition_rows, (list, tuple)) \
                        or len(condition_rows) != len(NEGATIVE_CONDITIONS):
                    errors.append(f"{path}.stream_rows[{index}] lacks eight conditions")
                    condition_rows = ()
                names = []
                contract_fields = {
                    "same_current_task", "same_truthful_answer_channel",
                    "same_query_budget", "same_zero_one_task_loss",
                    "same_budget_exhaustion_stopping_rule",
                    "same_metric_denominator", "main_legal_query_types",
                    "no_memory_legal_query_types",
                    "different_query_universes_explicit",
                    "answers_applied_to_current_posterior",
                    "both_within_budget",
                }
                for condition_index, condition_result in enumerate(condition_rows):
                    cpath = (f"{path}.stream_rows[{index}].conditions"
                             f"[{condition_index}]")
                    condition_result = _mapping(condition_result, cpath, errors)
                    condition = _mapping(condition_result.get("condition"),
                                         f"{cpath}.condition", errors)
                    names.append(condition.get("name"))
                    if condition_result.get("matched_protocol") is not True:
                        errors.append(f"{cpath} protocol is unmatched")
                    protocol = _mapping(condition_result.get("protocol_contract"),
                                        f"{cpath}.protocol_contract", errors)
                    if not contract_fields.issubset(set(protocol)):
                        errors.append(f"{cpath} protocol contract fields are incomplete")
                    bool_fields = contract_fields - {
                        "main_legal_query_types", "no_memory_legal_query_types"}
                    if any(protocol.get(name) is not True for name in bool_fields):
                        errors.append(f"{cpath} protocol contract has a false clause")
                    if tuple(protocol.get("main_legal_query_types", ())) != \
                            ("semantic",) \
                            or tuple(protocol.get(
                                "no_memory_legal_query_types", ())) != \
                            ("behavioral", "semantic"):
                        errors.append(f"{cpath} legal query universes are not explicit")
                    main = _mapping(condition_result.get("main"),
                                    f"{cpath}.main", errors)
                    no_memory = _mapping(condition_result.get("no_memory"),
                                         f"{cpath}.no_memory", errors)
                    if main.get("established_record_corruption") != 0 \
                            or no_memory.get("established_record_corruption") != 0:
                        errors.append(f"{cpath} corrupts established state")
                    scored = main.get("task_accuracy") is not None
                    if scored:
                        retrieval = _mapping(
                            protocol.get("main_retrieval"),
                            f"{cpath}.protocol_contract.main_retrieval", errors)
                        selected = protocol.get("main_retrieval_selected_keys")
                        if not isinstance(selected, (list, tuple)) or not selected \
                                or len(selected) != len(set(selected)):
                            errors.append(f"{cpath} lacks an explicit MAIN shortlist")
                        for field in RETRIEVAL_ACCOUNTING_FIELDS:
                            value = retrieval.get(field)
                            if not isinstance(value, int) or isinstance(value, bool) \
                                    or value < 0:
                                errors.append(
                                    f"{cpath}.main_retrieval.{field} is invalid")
                        inspected = retrieval.get(
                            "identity_specific_summaries_inspected")
                        if retrieval.get("protocol") != "A_GLOBAL_EXACT_SCAN" \
                                or not isinstance(inspected, int) \
                                or inspected <= 0 \
                                or retrieval.get(
                                    "identity_likelihoods_evaluated") != inspected \
                                or retrieval.get(
                                    "total_retrieval_node_equivalents") != inspected \
                                or retrieval.get("full_records_loaded") != 0 \
                                or retrieval.get("four_node_claim") is not False \
                                or retrieval.get("within_512") is not True \
                                or retrieval.get("total_retrieval_bytes", 513) > 512 \
                                or retrieval.get("total_retrieval_bytes") != \
                                retrieval.get("index_bytes_scanned"):
                            errors.append(
                                f"{cpath} MAIN retrieval contract is not explicit/exact-scan")
                        if isinstance(selected, (list, tuple)) \
                                and retrieval.get("shortlist_size") != len(selected):
                            errors.append(f"{cpath} MAIN shortlist/accounting differ")
                if tuple(names) != NEGATIVE_CONDITIONS:
                    errors.append(f"{path}.stream_rows[{index}] condition order differs")
        intervals = _mapping(section.get("intervals"),
                             f"{path}.intervals", errors)
        for name in (SAFETY_INTERVALS[0], SAFETY_INTERVALS[2]):
            _validate_interval(intervals.get(name), expected,
                               f"{path}.intervals.{name}", errors)
            row = intervals.get(name, {})
            if isinstance(row, Mapping) and _exact(row.get("hi")) \
                    and Fraction(row["hi"]) >= 0:
                errors.append(f"{path}.{name} safety interval is not below zero")
        cal = _mapping(section.get("calibration"),
                       f"{path}.calibration", errors)
        if cal.get("fires") is not True \
                or not all(isinstance(cal.get(name), int)
                           and cal.get(name, 0) > 0
                           for name in ("immediate_MAP_corruption",
                                        "forced_new_assimilation",
                                        "no_confirmation_contamination")):
            errors.append(f"{path} negative-transfer calibration did not fire")
        if cal.get("same_main_safety_predicate_rejected_every_plant") is not True:
            errors.append(f"{path} safety plants bypassed the MAIN predicate")
    return _finish(errors)


def _canon_sha(value: Any) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def _validate_exact_posterior(value: Any, path: str, errors: list[str],
                              expected_size: int | None = None) -> Mapping:
    posterior = _mapping(value, path, errors)
    if expected_size is not None and len(posterior) != expected_size:
        errors.append(f"{path} has {len(posterior)} rather than {expected_size} entries")
    total = Fraction(0)
    for key, mass in posterior.items():
        exact = _fraction(mass, f"{path}.{key}", errors)
        if exact is not None:
            if exact < 0:
                errors.append(f"{path}.{key} is negative")
            total += exact
    if posterior and total != 1:
        errors.append(f"{path} is not exactly normalized")
    return posterior


def _validate_restart_case(case: Any, path: str, errors: list[str]) \
        -> tuple[str, str, int] | None:
    case = _mapping(case, path, errors)
    if set(case) != set(RESTART_CASE_FIELDS):
        missing = sorted(set(RESTART_CASE_FIELDS) - set(case))
        extra = sorted(set(case) - set(RESTART_CASE_FIELDS))
        errors.append(f"{path} restart state fields differ "
                      f"missing={missing} extra={extra}")
    overlap, split, seed = (case.get("overlap"), case.get("split"),
                            case.get("seed"))
    if overlap not in OVERLAPS or split not in SPLITS \
            or not isinstance(seed, int) or isinstance(seed, bool):
        errors.append(f"{path} has an invalid stream key")
        key = None
    else:
        key = (overlap, split, seed)
    if case.get("state_schema") != RL1.SCHEMA:
        errors.append(f"{path}.state_schema is not {RL1.SCHEMA}")
    if not isinstance(case.get("state_step"), int) \
            or isinstance(case.get("state_step"), bool) \
            or case.get("state_step", -1) < 0:
        errors.append(f"{path}.state_step is invalid")
    history = case.get("post_query_history")
    if not isinstance(history, (list, tuple)) or not history \
            or any(not isinstance(event, (list, tuple)) or len(event) != 2
                   or any(not isinstance(v, int) or isinstance(v, bool)
                          for v in event) for event in history):
        errors.append(f"{path}.post_query_history is missing or malformed")

    state_payload = {
        "schema": case.get("state_schema"), "overlap": overlap,
        "step": case.get("state_step"),
        **{field: case.get(field) for field in RESTART_FIELDS},
    }
    state = None
    try:
        # This is the production child loader's pure path.  It verifies every
        # component seal, reconstructs selection weights/retrieval from the
        # public task and all eight confirmed records, replays query history,
        # and demands exact active identity/convention/support equality.
        state = RL1.state_from_payload(state_payload)
    except Exception as exc:
        errors.append(f"{path} is not an exactly reconstructible MAIN state: {exc}")

    shortlist = case.get("retrieval_shortlist")
    shortlist_size = len(shortlist) if isinstance(shortlist, (list, tuple)) else 0
    identity = _validate_exact_posterior(
        case.get("identity_posterior"), f"{path}.identity_posterior", errors,
        shortlist_size + 2 if shortlist_size else None)
    if "NEW_IDENTITY" not in identity or "OUT_OF_FAMILY" not in identity:
        errors.append(f"{path} identity posterior omits NEW/OUT")
    new_mass = _fraction(case.get("new_mass"), f"{path}.new_mass", errors)
    out_mass = _fraction(case.get("out_mass"), f"{path}.out_mass", errors)
    if new_mass is not None and identity.get("NEW_IDENTITY") != new_mass:
        errors.append(f"{path}.new_mass disagrees with identity posterior")
    if out_mass is not None and identity.get("OUT_OF_FAMILY") != out_mass:
        errors.append(f"{path}.out_mass disagrees with identity posterior")

    convention_rows = case.get("record_convention_posteriors", ())
    if not isinstance(convention_rows, (list, tuple)) \
            or len(convention_rows) != shortlist_size:
        errors.append(f"{path} convention posteriors do not match top-four support")
        convention_rows = ()
    active_keys = []
    for index, value in enumerate(convention_rows):
        row = _mapping(value, f"{path}.conventions[{index}]", errors)
        if set(row) != {"record_key", "support", "posterior"}:
            errors.append(f"{path}.conventions[{index}] fields are incomplete")
        record_key = row.get("record_key")
        if isinstance(record_key, str) and record_key:
            active_keys.append(record_key)
        else:
            errors.append(f"{path}.conventions[{index}] lacks a record key")
        support = row.get("support")
        if not isinstance(support, (list, tuple)) \
                or tuple(sorted(set(support))) != tuple(support):
            errors.append(f"{path}.conventions[{index}] support is invalid")
        _validate_exact_posterior(
            row.get("posterior"), f"{path}.conventions[{index}].posterior",
            errors)
    if tuple(active_keys) != tuple(shortlist or ()):
        errors.append(f"{path} convention rows differ from retrieval shortlist")

    confirmed = case.get("confirmed_records", ())
    if not isinstance(confirmed, (list, tuple)) or len(confirmed) != 8:
        errors.append(f"{path} does not contain eight confirmed records")
        confirmed = ()
    confirmed_keys: list[str] = []
    confirmed_fields = {
        "record_key", "grounded", "verification_domain",
        "challenge_digest", "query_set_digest", "validity_scope",
        "status", "version", "evidence_ids",
    }
    for index, value in enumerate(confirmed):
        row = _mapping(value, f"{path}.confirmed[{index}]", errors)
        if set(row) != confirmed_fields:
            errors.append(f"{path}.confirmed[{index}] fields are incomplete")
        key_value = row.get("record_key")
        if isinstance(key_value, str) and key_value:
            confirmed_keys.append(key_value)
        else:
            errors.append(f"{path}.confirmed[{index}] lacks a record key")
        if row.get("status") != "CONFIRMED" \
                or not isinstance(row.get("version"), int) \
                or row.get("version", 0) < 1:
            errors.append(f"{path}.confirmed[{index}] status/version is invalid")
        if not _valid_digest(row.get("challenge_digest")) \
                or not _valid_digest(row.get("query_set_digest")):
            errors.append(f"{path}.confirmed[{index}] scope digests are invalid")
        grounded, evidence = row.get("grounded"), row.get("evidence_ids")
        if not isinstance(grounded, (list, tuple)) or not grounded \
                or not isinstance(evidence, (list, tuple)) \
                or len(evidence) != len(set(evidence)):
            errors.append(f"{path}.confirmed[{index}] evidence is invalid")
    if len(confirmed_keys) != len(set(confirmed_keys)) \
            or not set(active_keys).issubset(set(confirmed_keys)):
        errors.append(f"{path} active convention keys escape confirmed records")

    branches = case.get("provisional_branches", ())
    branch_fields = {
        "branch_id", "identity_posterior", "record_support", "new_support",
        "new_mass", "out_mass", "evidence_ids", "asked", "query_universe",
        "policy", "status", "update_budget",
    }
    if not isinstance(branches, (list, tuple)) or not branches:
        errors.append(f"{path} loses provisional identity branches")
        branches = ()
    for index, value in enumerate(branches):
        row = _mapping(value, f"{path}.branches[{index}]", errors)
        if set(row) != branch_fields:
            errors.append(f"{path}.branches[{index}] fields are incomplete")
        branch_identity = _validate_exact_posterior(
            row.get("identity_posterior"),
            f"{path}.branches[{index}].identity_posterior", errors,
            shortlist_size + 2 if shortlist_size else None)
        if set(branch_identity) != set(identity):
            errors.append(f"{path}.branches[{index}] identity support differs")
        if row.get("new_mass") != branch_identity.get("NEW_IDENTITY") \
                or row.get("out_mass") != branch_identity.get("OUT_OF_FAMILY"):
            errors.append(f"{path}.branches[{index}] NEW/OUT mass differs")
        asked = row.get("asked")
        if index == 0 and asked != history:
            errors.append(f"{path} provisional history differs from MAIN")
        if not isinstance(row.get("update_budget"), int) \
                or isinstance(row.get("update_budget"), bool) \
                or (isinstance(asked, (list, tuple))
                    and row.get("update_budget", -1) < len(asked)):
            errors.append(f"{path}.branches[{index}] update budget is invalid")

    if not isinstance(shortlist, (list, tuple)) \
            or not 0 < len(shortlist) <= 4 \
            or len(shortlist) != len(set(shortlist)) \
            or not set(shortlist).issubset(set(confirmed_keys)):
        errors.append(f"{path}.retrieval_shortlist is invalid")

    accounting = _mapping(case.get("retrieval_accounting"),
                          f"{path}.retrieval_accounting", errors)
    accounting_fields = {
        "protocol", *RETRIEVAL_ACCOUNTING_FIELDS, "incomplete_retrieval",
        "within_512", "four_node_claim",
    }
    if set(accounting) != accounting_fields:
        errors.append(f"{path}.retrieval_accounting fields are incomplete")
    if accounting.get("protocol") != "A_GLOBAL_EXACT_SCAN" \
            or accounting.get("identity_specific_summaries_inspected") != 8 \
            or accounting.get("identity_likelihoods_evaluated") != 8 \
            or accounting.get("total_retrieval_node_equivalents") != 8 \
            or accounting.get("shortlist_size") != shortlist_size \
            or accounting.get("full_records_loaded") != 0 \
            or accounting.get("total_retrieval_bytes", 513) > 512 \
            or accounting.get("within_512") is not True \
            or accounting.get("incomplete_retrieval") is not True \
            or accounting.get("four_node_claim") is not False:
        errors.append(f"{path} does not charge the exact eight-node MAIN scan")

    supports = _mapping(case.get("post_query_record_supports"),
                        f"{path}.post_query_record_supports", errors)
    if set(supports) != set(shortlist or ()) or any(
            not isinstance(value, (list, tuple))
            or tuple(sorted(set(value))) != tuple(value)
            for value in supports.values()):
        errors.append(f"{path} reconstructing active supports are incomplete")
    new_support = case.get("post_query_new_support")
    if not isinstance(new_support, (list, tuple)) \
            or tuple(sorted(set(new_support))) != tuple(new_support):
        errors.append(f"{path} reconstructing NEW support is invalid")
    task = _mapping(case.get("task_evidence"), f"{path}.task_evidence", errors)
    if set(task) != {"kind", "demos", "live", "u", "pool", "tie",
                     "accepted"} or "z" in task:
        errors.append(f"{path} task evidence is incomplete or leaks truth")
    priors = _mapping(case.get("inference_priors"),
                      f"{path}.inference_priors", errors)
    if set(priors) != {"p_new", "p_out", "record_prior", "with_new",
                       "with_out"} \
            or priors.get("with_new") is not True \
            or priors.get("with_out") is not True:
        errors.append(f"{path} inference priors do not reconstruct open world")
    weights = _mapping(case.get("selection_weights"),
                       f"{path}.selection_weights", errors)
    if weights.get("reconstruction") != \
            "recompute_exact_selection_weights_from_task" \
            or not all(_valid_digest(weights.get(name)) for name in (
                "scaled_sha256", "task_input_sha256",
                "family_signature_sha256")):
        errors.append(f"{path} selection weights are not reconstructible")
    policy = _mapping(case.get("query_policy_state"),
                      f"{path}.query_policy_state", errors)
    if policy.get("policy") != "information_gain" \
            or policy.get("history") != history \
            or policy.get("stop_when_identity_decisive") is not False \
            or not isinstance(policy.get("query_budget"), int) \
            or policy.get("query_budget", 0) <= len(history or ()):
        errors.append(f"{path} query policy cannot resume exact MAIN")

    serialized = _mapping(case.get("serialized_hashes"),
                          f"{path}.serialized_hashes", errors)
    if set(serialized) != set(RESTART_STATE_HASH_FIELDS) \
            or not all(_valid_digest(value) for value in serialized.values()):
        errors.append(f"{path} serialized hashes are incomplete")
    for field in HASHED_RESTART_FIELDS:
        if field in case and serialized.get(field) != _canon_sha(case[field]):
            errors.append(f"{path} serialized hash for {field} is wrong")
    metadata = {"schema": case.get("state_schema"), "overlap": overlap,
                "step": case.get("state_step")}
    if serialized.get("metadata") != _canon_sha(metadata):
        errors.append(f"{path} serialized metadata hash is wrong")

    cycle = _mapping(case.get("cycle"), f"{path}.cycle", errors)
    cycle_required = {
        "ok", "overlap", "parent_pid", "child_pid", "parent_pid_gone",
        "checkpoint_sha256", "checkpoint_bytes", "checkpoint_hashes",
        "child_loaded_parent_state", "uninterrupted_final_sha256",
        "restarted_final_sha256", "final_hashes_identical",
        "final_state_identical", "loaded_step", "final_step",
        "real_main_continuation", "continuation_policy",
        "continuation_queries", "continuation_answers",
        "forbidden_channel_closed", "child_env_size", "calibrations",
        "all_calibrations_rejected",
    }
    if set(cycle) != cycle_required:
        errors.append(f"{path}.cycle fields are incomplete or unexpected")
    if cycle.get("ok") is not True or cycle.get("overlap") != overlap:
        errors.append(f"{path}.cycle did not succeed")
    if not isinstance(cycle.get("parent_pid"), int) \
            or not isinstance(cycle.get("child_pid"), int) \
            or cycle.get("parent_pid") == cycle.get("child_pid") \
            or cycle.get("parent_pid_gone") is not True:
        errors.append(f"{path}.cycle is not a genuine parent-death restart")
    if cycle.get("child_loaded_parent_state") is not True \
            or cycle.get("forbidden_channel_closed") is not True \
            or not isinstance(cycle.get("child_env_size"), int) \
            or cycle.get("child_env_size", 999) > 5:
        errors.append(f"{path}.cycle child load/scrub checks failed")
    if not isinstance(cycle.get("checkpoint_bytes"), int) \
            or isinstance(cycle.get("checkpoint_bytes"), bool) \
            or cycle.get("checkpoint_bytes", 0) <= 0:
        errors.append(f"{path}.cycle checkpoint byte count is invalid")
    for name in ("checkpoint_sha256", "uninterrupted_final_sha256",
                 "restarted_final_sha256"):
        if not _valid_digest(cycle.get(name)):
            errors.append(f"{path}.cycle.{name} is invalid")
    if cycle.get("uninterrupted_final_sha256") != \
            cycle.get("restarted_final_sha256") \
            or cycle.get("final_hashes_identical") is not True \
            or cycle.get("final_state_identical") is not True:
        errors.append(f"{path}.cycle restarted exact state differs")
    if cycle.get("checkpoint_sha256") != _canon_sha(state_payload):
        errors.append(f"{path}.cycle checkpoint hash differs from exact state")
    if cycle.get("checkpoint_hashes") != serialized:
        errors.append(f"{path}.cycle component hashes differ from exact state")
    queries, answers = (cycle.get("continuation_queries"),
                        cycle.get("continuation_answers"))
    if cycle.get("loaded_step") != case.get("state_step") \
            or cycle.get("final_step") != case.get("state_step", -1) + 1 \
            or cycle.get("real_main_continuation") is not True \
            or cycle.get("continuation_policy") != "information_gain" \
            or not isinstance(queries, (list, tuple)) or len(queries) != 1 \
            or not isinstance(answers, (list, tuple)) or len(answers) != 1 \
            or any(not isinstance(value, int) or isinstance(value, bool)
                   for value in tuple(queries or ()) + tuple(answers or ())):
        errors.append(f"{path}.cycle is not an exact one-step MAIN continuation")
    elif state is not None:
        transition = {
            "type": "main_clarification",
            "policy": cycle["continuation_policy"],
            "query": queries[0], "answer": answers[0],
            "evidence_id": (
                f"restart:{overlap}:{case.get('state_step')}:"
                f"{queries[0]}:{answers[0]}"),
        }
        try:
            expected_final = RL1.advance_state(state, transition)
        except Exception as exc:
            errors.append(f"{path}.cycle continuation is not legal MAIN: {exc}")
        else:
            expected_final_sha = _canon_sha(expected_final)
            if cycle.get("uninterrupted_final_sha256") != expected_final_sha \
                    or cycle.get("restarted_final_sha256") != expected_final_sha:
                errors.append(
                    f"{path}.cycle final hash is not the reconstructed MAIN state")
    if cycle.get("calibrations") != {} \
            or cycle.get("all_calibrations_rejected") is not True:
        errors.append(f"{path}.cycle should delegate plants to matrix calibration")

    plants = case.get("calibration_plants")
    if not isinstance(plants, (list, tuple)):
        errors.append(f"{path}.calibration_plants must be a sequence")
        plants = ()
    plant_keys = set()
    for index, plant in enumerate(plants):
        if not isinstance(plant, (list, tuple)) or len(plant) != 2 \
                or plant[0] not in RESTART_FIELDS or plant[1] != "mutate":
            errors.append(f"{path}.calibration_plants[{index}] is invalid")
            continue
        plant_keys.add(f"{plant[1]}:{plant[0]}")
    corruptions = _mapping(case.get("corrupt_checkpoint_calibrations"),
                           f"{path}.corruptions", errors)
    if set(corruptions) != plant_keys:
        errors.append(f"{path} scheduled corrupt-checkpoint rows differ")
    for name, value in corruptions.items():
        row = _mapping(value, f"{path}.corruptions.{name}", errors)
        if row.get("rejected") is not True \
                or not isinstance(row.get("returncode"), int) \
                or row.get("returncode") == 0 \
                or row.get("same_child_validator") != \
                "x65a.restart_l1 child/state_from_payload":
            errors.append(f"{path} corrupt checkpoint {name} was not rejected")
    return key


def _validate_restart_matrix(artifact: Mapping, errors: list[str]) -> None:
    restart = _mapping(artifact.get("restart"), "restart", errors)
    if set(restart) != {"schema", "contract", "cases", "validation",
                       "calibration"}:
        errors.append("restart matrix fields are incomplete or unexpected")
    if restart.get("schema") != RESTART_MATRIX_SCHEMA:
        errors.append("restart matrix schema is wrong")
    provenance = _mapping(artifact.get("provenance"), "provenance", errors)
    development = tuple(provenance.get("development_stream_seeds", ()))
    validation_seeds = tuple(provenance.get("validation_stream_seeds", ()))
    contract = _mapping(restart.get("contract"), "restart.contract", errors)
    expected_contract = {
        "development_seeds", "validation_seeds", "overlaps",
        "identities_per_stream", "required_corruption_fields",
        "corruption_mode", "expected_streams",
    }
    if set(contract) != expected_contract:
        errors.append("restart matrix contract fields are incomplete")
    if tuple(contract.get("development_seeds", ())) != development \
            or tuple(contract.get("validation_seeds", ())) != validation_seeds:
        errors.append("restart matrix contract seeds differ from provenance")
    if set(contract.get("overlaps", ())) != set(OVERLAPS) \
            or contract.get("identities_per_stream") != 8 \
            or tuple(contract.get("required_corruption_fields", ())) != \
            RESTART_FIELDS \
            or contract.get("corruption_mode") != "mutate":
        errors.append("restart matrix contract scope is wrong")
    expected_keys = {
        (overlap, split, int(seed))
        for overlap in OVERLAPS
        for split, seeds in (("development", development),
                             ("validation", validation_seeds))
        for seed in seeds
    }
    # The frozen L1 matrix is four development plus four validation streams in
    # each of two strata: 2 * (4 + 4) = 16 genuine restart cycles.
    if len(expected_keys) != 16 or contract.get("expected_streams") != 16:
        errors.append("restart contract is not the frozen full 16-case matrix")
    cases = restart.get("cases")
    if not isinstance(cases, (list, tuple)):
        errors.append("restart matrix cases must be a sequence")
        cases = ()
    elif len(cases) != 16:
        errors.append("restart matrix must contain exactly 16 cases")
    actual_keys = []
    seen_plants = set()
    for index, case in enumerate(cases):
        key = _validate_restart_case(case, f"restart.cases[{index}]", errors)
        if key is not None:
            actual_keys.append(key)
        if isinstance(case, Mapping):
            seen_plants.update(case.get("corrupt_checkpoint_calibrations", {}))
    if set(actual_keys) != expected_keys or len(actual_keys) != len(set(actual_keys)):
        errors.append("restart cases omit, duplicate, or add stream keys")
    if seen_plants != {f"mutate:{field}" for field in RESTART_FIELDS}:
        errors.append("restart matrix does not calibrate every required state field")

    validation = _mapping(restart.get("validation"),
                          "restart.validation", errors)
    if set(validation) != {"checks", "case_checks", "errors", "passed"}:
        errors.append("restart matrix validation fields are incomplete")
    checks = _mapping(validation.get("checks"),
                      "restart.validation.checks", errors)
    if set(checks) != set(RESTART_MATRIX_CHECKS) \
            or not all(value is True for value in checks.values()):
        errors.append("restart matrix aggregate validation did not pass")
    case_checks = _mapping(validation.get("case_checks"),
                           "restart.validation.case_checks", errors)
    expected_labels = {f"{overlap}/{split}/{seed}"
                       for overlap, split, seed in expected_keys}
    if set(case_checks) != expected_labels:
        errors.append("restart matrix validation lacks all 16 case rows")
    for label, value in case_checks.items():
        row = _mapping(value, f"restart.validation.case_checks.{label}", errors)
        if set(row) != set(RESTART_MATRIX_CASE_CHECKS) \
                or not all(flag is True for flag in row.values()):
            errors.append(f"restart matrix validator rejected {label}")
    if validation.get("errors") not in ([], ()) \
            or validation.get("passed") is not True:
        errors.append("restart matrix validation headline did not pass")

    calibration = _mapping(restart.get("calibration"),
                            "restart.calibration", errors)
    if calibration.get("fires") is not True \
            or calibration.get("valid_matrix_accepted") is not True \
            or calibration.get(
                "accepted_corruption_rejected_by_same_matrix_validator") \
                is not True:
        errors.append("restart matrix same-validator calibration did not fire")
    planted = _mapping(calibration.get("planted_validation"),
                       "restart.calibration.planted_validation", errors)
    if planted.get("passed") is not False \
            or not planted.get("errors"):
        errors.append("restart planted accepted-corruption report was not rejected")

    storage = _mapping(artifact.get("restart_storage_accounting"),
                       "restart_storage_accounting", errors)
    expected_storage = {
        "category", "checkpoints", "minimum_checkpoint_bytes",
        "maximum_checkpoint_bytes", "total_checkpoint_bytes",
        "active_semantic_memory_budget_bytes",
        "counted_against_active_semantic_memory", "boundary",
        "case_sizes_match",
    }
    if set(storage) != expected_storage:
        errors.append("restart storage accounting fields are incomplete")
    sizes = tuple(
        case.get("cycle", {}).get("checkpoint_bytes")
        for case in cases if isinstance(case, Mapping))
    valid_sizes = bool(sizes) and all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in sizes)
    if storage.get("category") != "serialized_restart_and_audit_state" \
            or storage.get("checkpoints") != len(cases) \
            or not valid_sizes \
            or storage.get("minimum_checkpoint_bytes") != min(sizes or (0,)) \
            or storage.get("maximum_checkpoint_bytes") != max(sizes or (0,)) \
            or storage.get("total_checkpoint_bytes") != sum(sizes or (0,)) \
            or storage.get("active_semantic_memory_budget_bytes") != 4096 \
            or storage.get("counted_against_active_semantic_memory") is not False \
            or storage.get("case_sizes_match") is not True:
        errors.append("restart storage accounting disagrees with case payloads")
    boundary = storage.get("boundary")
    if not isinstance(boundary, str) \
            or "no bounded-total-memory claim" not in boundary:
        errors.append("restart storage boundary omits the total-memory limitation")


def _validate_l110(artifact: Mapping) -> CT.ValidationResult:
    errors: list[str] = []
    # L1.10 is independently fail-closed: a replicated point estimate is not
    # evidence for a resource-matched retrieval effect if L1.3's executable
    # accounting/freeze contract has been corrupted.
    retrieval_contract = _validate_l13(artifact)
    errors.extend(f"central retrieval contract: {error}"
                  for error in retrieval_contract.errors)
    provenance = _mapping(artifact.get("provenance"), "provenance", errors)
    margins = _mapping(provenance.get("validation_frozen_margins"),
                       "validation_frozen_margins", errors)
    l3_margin = _fraction(margins.get("L3_retrieval_noninferiority"),
                          "L3 retrieval margin", errors)
    required = _mapping(artifact.get("required_intervals"),
                        "required_intervals", errors)
    retrieval = _mapping(artifact.get("retrieval"), "retrieval", errors)
    negative = _mapping(artifact.get("negative_transfer"),
                        "negative_transfer", errors)
    if set(required) != set(OVERLAPS):
        errors.append("required intervals do not cover both strata")
    for overlap in OVERLAPS:
        by_split = _mapping(required.get(overlap),
                            f"required_intervals.{overlap}", errors)
        if set(by_split) != set(SPLITS):
            errors.append(f"required intervals {overlap} lack both splits")
        for split in SPLITS:
            path = f"required_intervals.{overlap}.{split}"
            rows = _mapping(by_split.get(split), path, errors)
            expected_names = set(CENTRAL_INTERVALS + SAFETY_INTERVALS
                                 + ("restart_difference",))
            if set(rows) != expected_names:
                errors.append(f"{path} has missing or unexpected effects")
            clusters = _seed_count(artifact, split, errors)
            for name in expected_names:
                _validate_interval(rows.get(name), clusters,
                                   f"{path}.{name}", errors)
                interval = rows.get(name)
                if isinstance(interval, Mapping) \
                        and (not isinstance(interval.get("resamples"), int)
                             or isinstance(interval.get("resamples"), bool)
                             or interval.get("resamples", 0) < 3000):
                    errors.append(f"{path}.{name} has fewer than 3000 resamples")
            for name in CENTRAL_INTERVALS[:3]:
                row = rows.get(name, {})
                if isinstance(row, Mapping) and _exact(row.get("lo")) \
                        and Fraction(row["lo"]) <= 0:
                    errors.append(f"{path}.{name} lower bound is not > 0")
            exact = rows.get("MAIN_minus_exact_accuracy", {})
            if isinstance(exact, Mapping) and _exact(exact.get("lo")) \
                    and l3_margin is not None \
                    and Fraction(exact["lo"]) < -l3_margin:
                errors.append(f"{path} MAIN is inferior to exact all-record")
            queries = rows.get("MAIN_minus_no_memory_query_count", {})
            if isinstance(queries, Mapping) and _exact(queries.get("hi")) \
                    and Fraction(queries["hi"]) >= 0:
                errors.append(f"{path} query advantage upper bound is not < 0")
            for name in SAFETY_INTERVALS:
                row = rows.get(name, {})
                if isinstance(row, Mapping) and _exact(row.get("hi")) \
                        and Fraction(row["hi"]) >= 0:
                    errors.append(f"{path}.{name} upper bound is not < 0")
            restart_iv = rows.get("restart_difference", {})
            if isinstance(restart_iv, Mapping) and tuple(
                    restart_iv.get(name) for name in ("lo", "delta", "hi")) \
                    != (0, 0, 0):
                errors.append(f"{path} restart interval is not exactly zero")
            rsplit = _mapping(
                _mapping(retrieval.get(overlap),
                         f"retrieval.{overlap}", errors).get(split),
                f"retrieval.{overlap}.{split}", errors)
            nsplit = _mapping(
                _mapping(negative.get(overlap),
                         f"negative_transfer.{overlap}", errors).get(split),
                f"negative_transfer.{overlap}.{split}", errors)
            central = _mapping(rsplit.get("intervals"),
                               f"retrieval.{overlap}.{split}.intervals", errors)
            safety = _mapping(nsplit.get("intervals"),
                              f"negative_transfer.{overlap}.{split}.intervals",
                              errors)
            for name in CENTRAL_INTERVALS:
                if rows.get(name) != central.get(name):
                    errors.append(f"{path}.{name} differs from retrieval evidence")
            stream_rows = _mapping(
                rsplit.get("streams"), f"retrieval.{overlap}.{split}.streams",
                errors)
            provenance_seeds = tuple(_mapping(
                artifact.get("provenance"), "provenance", errors).get(
                    "development_stream_seeds" if split == "development"
                    else "validation_stream_seeds", ()))
            central_metrics = {
                "MAIN_minus_random_accuracy":
                    ("random_retrieval", "task_accuracy"),
                "MAIN_minus_recency_accuracy":
                    ("recency", "task_accuracy"),
                "MAIN_minus_surface_accuracy":
                    ("surface_nearest", "task_accuracy"),
                "MAIN_minus_exact_accuracy":
                    ("exact_all_record", "task_accuracy"),
                "MAIN_minus_no_memory_query_count":
                    ("no_memory", "questions_to_frozen_target_accuracy"),
            }
            for interval_name, (control, metric) in central_metrics.items():
                left, right = [], []
                for seed in provenance_seeds:
                    stream = stream_rows.get(seed, stream_rows.get(str(seed), {}))
                    if not isinstance(stream, Mapping):
                        errors.append(f"{path} lacks stream {seed} for paired interval")
                        continue
                    left.append(_mapping(
                        stream.get("MAIN_protocol_A"),
                        f"{path}.stream.{seed}.MAIN", errors).get(metric))
                    right.append(_mapping(
                        stream.get(control),
                        f"{path}.stream.{seed}.{control}", errors).get(metric))
                interval = _mapping(rows.get(interval_name),
                                    f"{path}.{interval_name}", errors)
                _validate_replicated_interval(
                    interval, left, right, clusters,
                    f"{path}.{interval_name}", errors)
            for name in (SAFETY_INTERVALS[0], SAFETY_INTERVALS[2]):
                if rows.get(name) != safety.get(name):
                    errors.append(f"{path}.{name} differs from safety evidence")
            safety_vectors = _mapping(
                nsplit.get("paired_safety_vectors"),
                f"negative_transfer.{overlap}.{split}.paired_safety_vectors",
                errors)
            safety_pairs = {
                SAFETY_INTERVALS[0]: (
                    "provisional_assignment_corruption",
                    "immediate_MAP_corruption"),
                SAFETY_INTERVALS[2]: (
                    "confirmation_contamination",
                    "no_confirmation_contamination"),
            }
            for interval_name, (left_name, right_name) in safety_pairs.items():
                _validate_replicated_interval(
                    rows.get(interval_name), safety_vectors.get(left_name),
                    safety_vectors.get(right_name), clusters,
                    f"{path}.{interval_name}", errors)
            for cal_path, source in (("retrieval", rsplit),
                                     ("negative_transfer", nsplit)):
                cal = _mapping(source.get("calibration"),
                               f"{cal_path}.{overlap}.{split}.calibration",
                               errors)
                if cal.get("fires") is not True:
                    errors.append(f"{cal_path}.{overlap}.{split} calibration failed")
                if cal_path == "negative_transfer" and (
                        cal.get(
                            "same_main_safety_predicate_rejected_every_plant")
                        is not True
                        or not all(isinstance(cal.get(name), int)
                                   and not isinstance(cal.get(name), bool)
                                   and cal.get(name, 0) > 0
                                   for name in (
                                       "immediate_MAP_corruption",
                                       "forced_new_assimilation",
                                       "no_confirmation_contamination"))):
                    errors.append(
                        f"negative_transfer.{overlap}.{split} safety calibration is weak")
            news = _mapping(
                _mapping(artifact.get("new_identity"), "new_identity", errors)
                .get(overlap), f"new_identity.{overlap}", errors)
            new_split = _mapping(news.get(split),
                                 f"new_identity.{overlap}.{split}", errors)
            if _mapping(new_split.get("calibration"),
                        f"new_identity.{overlap}.{split}.calibration",
                        errors).get("fires") is not True:
                errors.append(f"new_identity.{overlap}.{split} calibration failed")
            new_calibration = _mapping(
                new_split.get("calibration"),
                f"new_identity.{overlap}.{split}.calibration", errors)
            details = new_calibration.get("details", ())
            if not isinstance(details, (list, tuple)) \
                    or len(details) != clusters \
                    or any(not isinstance(row, Mapping) or not row
                           or not all(value is True for value in row.values())
                           for row in details):
                errors.append(
                    f"new_identity.{overlap}.{split} per-stream calibrations failed")
            new_vectors = _mapping(
                new_split.get("paired_forced_assimilation_vectors"),
                f"new_identity.{overlap}.{split}.paired_vectors", errors)
            _validate_replicated_interval(
                rows.get("NEW_minus_no_NEW_forced_assimilation"),
                new_vectors.get("MAIN"), new_vectors.get("no_NEW_forced"),
                clusters,
                f"{path}.NEW_minus_no_NEW_forced_assimilation", errors)
            if new_split.get("main_classification_gate") is not True \
                    or new_split.get("later_reuse_failures") not in ([], ()):
                errors.append(f"new_identity.{overlap}.{split} has creation/reuse failures")
            restart_cases = _mapping(
                artifact.get("restart"), "restart", errors).get("cases", ())
            if not isinstance(restart_cases, (list, tuple)):
                restart_cases = ()
            restart_observed = []
            for case in restart_cases:
                if not isinstance(case, Mapping) \
                        or case.get("overlap") != overlap \
                        or case.get("split") != split:
                    continue
                cycle = _mapping(case.get("cycle"),
                                 f"{path}.restart_case", errors)
                restart_observed.append(int(
                    cycle.get("final_hashes_identical") is not True
                    or cycle.get("uninterrupted_final_sha256") !=
                    cycle.get("restarted_final_sha256")))
            _validate_replicated_interval(
                rows.get("restart_difference"), tuple(restart_observed),
                tuple(Fraction(0) for _ in restart_observed), clusters,
                f"{path}.restart_difference", errors)

    _validate_restart_matrix(artifact, errors)
    return _finish(errors)


_VALIDATORS: dict[str, Callable[[Mapping], CT.ValidationResult]] = {
    "L1.0": _validate_l10,
    "L1.1": _validate_l11,
    "L1.2": _validate_l12,
    "L1.3": _validate_l13,
    "L1.4": _validate_l14,
    "L1.5": _validate_l15,
    "L1.6": _validate_l16,
    "L1.7": _validate_l17,
    "L1.8": _validate_l18,
    "L1.9": _validate_l19,
    "L1.10": _validate_l110,
}


def validate_gate(artifact: Mapping, gate: str) -> CT.ValidationResult:
    """Run one pure L1 evidence validator."""
    if gate not in _VALIDATORS:
        raise ValueError(f"unknown L1 gate {gate!r}")
    if not isinstance(artifact, Mapping):
        return _finish(("artifact must be a mapping",))
    return _VALIDATORS[gate](artifact)


def _first_split(artifact: MutableMapping, section: str) -> MutableMapping:
    return artifact[section][OVERLAPS[0]][SPLITS[0]]


def _m_l10_missing_provenance(art):
    art["provenance"].pop("task_conditions_per_stream", None)


def _m_l10_final_seed(art):
    art["provenance"]["final_stream_seed_sampled"] = True


def _m_l10_prereq(art):
    art["x64h_prerequisite"]["passed"] = False


def _m_l11_risk(art):
    row = _first_split(art, "matched_risk")["taskwise_rows"][0]["q0"]
    row["stable_risk"], row["latent_action_risk_under_stable"] = 1, 0


def _m_l11_history(art):
    _first_split(art, "matched_risk")["taskwise_rows"][0]["q1"][
        "matched_history"] = False


def _m_l11_calibration(art):
    _first_split(art, "matched_risk")["calibration"]["fires"] = False


def _m_l12_proof(art):
    art["sketch_sufficiency"]["proof"]["valid"] = False


def _m_l12_mismatch(art):
    mismatches = art["sketch_sufficiency"]["differential"]["shared"][
        "mismatches"]
    mismatches[next(iter(mismatches))] = 1


def _m_l12_incomplete(art):
    art["sketch_sufficiency"]["incomplete_retrieval"] = True


def _m_l12_countermodel_calibration(art):
    art["sketch_sufficiency"]["calibration"][
        "same_premise_validator_rejects_hidden_nonindicator_weight"] = False


def _m_l13_undercharge(art):
    row = _first_split(art, "retrieval")["per_task_accounting"][
        "MAIN_protocol_A"][0]
    row["identity_specific_summaries_inspected"] = 4
    row["identity_likelihoods_evaluated"] = 4
    row["total_retrieval_node_equivalents"] = 4


def _m_l13_bytes(art):
    row = _first_split(art, "retrieval")["per_task_accounting"][
        "MAIN_protocol_A"][0]
    row["total_retrieval_bytes"] = 513


def _m_l13_calibration(art):
    _first_split(art, "retrieval")["calibration"]["fires"] = False


def _m_l13_freeze(art):
    _first_split(art, "retrieval")["main_protocol_freeze"]["record"][
        "shortlist_size"] = 8


def _m_l13_unmatched_control(art):
    row = _first_split(art, "retrieval")["per_task_accounting"][
        "random_retrieval"][0]
    row["total_retrieval_node_equivalents"] = 4


def _m_l14_total(art):
    _budget(art["query_accounting"]["budgets"], 3)[
        "queries_actually_asked"] = 294


def _m_l14_label(art):
    art["query_accounting"]["published_curve"] = "0.631 -> 0.952 at q=2.46"


def _m_l14_calibration(art):
    art["query_accounting"]["calibration"]["fires"] = False


def _m_l15_policy(art):
    _first_split(art, "memoryless")["policies"].pop(MEMORYLESS_POLICIES[0])


def _m_l15_unapplied(art):
    _budget(_first_split(art, "memoryless")["policies"][
        MEMORYLESS_POLICIES[0]], 1)["answers_applied"] = False


def _m_l15_effect(art):
    row = _budget(_first_split(art, "memoryless")["policies"][
        MEMORYLESS_POLICIES[0]], 1)
    row["per_question_resolution_effects"][0]["effects"] = []


def _m_l15_task_ig_flat(art):
    curve = _first_split(art, "memoryless")["policies"][
        "no_memory_exact_task_information_gain"]
    _budget(curve, 4)["task_accuracy"] = _budget(
        curve, 0)["task_accuracy"]


def _m_l15_joint_ig_flat(art):
    curve = _first_split(art, "memoryless")["policies"][
        "no_memory_exact_convention_task_information_gain"]
    _budget(curve, 4)["task_accuracy"] = _budget(
        curve, 0)["task_accuracy"]


def _m_l16_interval(art):
    row = _first_split(art, "active_query")["intervals"]["task_accuracy"]
    row["lo"], row["delta"], row["hi"] = 1, 0, -1


def _m_l16_prefix(art):
    stream = next(iter(_first_split(art, "active_query")["streams"].values()))
    stream["information_gain"]["accuracy_curve"]["prefix_consistent"] = False


def _m_l16_calibration(art):
    _first_split(art, "active_query")["calibration"]["fires"] = False


def _m_l17_unresolved(art):
    _first_split(art, "new_identity")["unresolved_new_rate"] = Fraction(1, 2)


def _m_l17_reuse(art):
    _first_split(art, "new_identity")["later_reuse_of_new_records"] = 0


def _m_l17_calibration(art):
    _first_split(art, "new_identity")["calibration"]["fires"] = False


def _m_l18_shared_transfer(art):
    row = art["scope_audit"]["shared"]["constructibility"][
        "out_of_family_transfer_utterance"]
    row["constructible"] = True


def _m_l18_scope_digest(art):
    art["scope_audit"]["shared"]["restricted_scope"]["record"]["scope"].pop(
        "query_set_digest")


def _m_l18_global(art):
    art["scope_audit"]["shared"]["restricted_scope"][
        "false_global_promotions"] = 1


def _m_l18_missing(art):
    row = art["scope_audit"]["shared"]["constructibility"][
        "MISSING_REPRESENTATION"]
    row["outcome"] = "UNRESOLVED"
    row["cause_posterior"]["MISSING_REPRESENTATION"] = Fraction(0)


def _m_l18_vacuous_convention(art):
    art["scope_audit"]["shared"]["constructibility"][
        "out_of_family_convention"]["family_membership_count"] = 1


def _m_l18_vacuous_grounded(art):
    art["scope_audit"]["shared"]["constructibility"][
        "out_of_family_grounded_event"]["zero_survivors"] = False


def _m_l18_vacuous_unknown(art):
    art["scope_audit"]["shared"]["constructibility"][
        "UNKNOWN_MEANING"]["derived_live_count"] = 1


def _m_l18_vacuous_restricted(art):
    art["scope_audit"]["shared"]["constructibility"][
        "restricted_query_indistinguishable"]["case"][
            "globally_equal"] = True


def _m_l19_inferior(art):
    _first_split(art, "negative_transfer")["conditions"][
        NEGATIVE_CONDITIONS[0]]["noninferior"] = False


def _m_l19_corruption(art):
    _first_split(art, "negative_transfer")["conditions"][
        NEGATIVE_CONDITIONS[0]]["established_record_corruption"] = 1


def _m_l19_calibration(art):
    _first_split(art, "negative_transfer")["calibration"]["fires"] = False


def _m_l19_unmatched_protocol(art):
    row = _first_split(art, "negative_transfer")
    row["stream_rows"][0]["conditions"][0]["protocol_contract"][
        "same_query_budget"] = False


def _m_l110_random_bound(art):
    _first_split(art, "required_intervals")[
        "MAIN_minus_random_accuracy"]["lo"] = 0


def _m_l110_exact_margin(art):
    _first_split(art, "required_intervals")[
        "MAIN_minus_exact_accuracy"]["lo"] = Fraction(-1, 10)


def _m_l110_query_bound(art):
    _first_split(art, "required_intervals")[
        "MAIN_minus_no_memory_query_count"]["hi"] = 0


def _m_l110_safety_bound(art):
    _first_split(art, "required_intervals")[SAFETY_INTERVALS[0]]["hi"] = 0


def _m_l110_restart_interval(art):
    _first_split(art, "required_intervals")["restart_difference"]["delta"] = 1


def _m_l110_restart(art):
    art["restart"]["cases"][0]["cycle"]["ok"] = False


def _m_l110_restart_storage(art):
    art["restart_storage_accounting"][
        "counted_against_active_semantic_memory"] = True


def _reseal_restart_case(case: MutableMapping) -> None:
    """Reseal a planted state so reconstruction, not a stale hash, rejects."""
    hashes = {field: _canon_sha(case[field])
              for field in HASHED_RESTART_FIELDS}
    hashes["metadata"] = _canon_sha({
        "schema": case["state_schema"], "overlap": case["overlap"],
        "step": case["state_step"],
    })
    case["serialized_hashes"] = hashes
    state_payload = {
        "schema": case["state_schema"], "overlap": case["overlap"],
        "step": case["state_step"],
        **{field: case[field] for field in RESTART_FIELDS},
    }
    case["cycle"]["checkpoint_hashes"] = copy.deepcopy(hashes)
    case["cycle"]["checkpoint_sha256"] = _canon_sha(state_payload)
    case["cycle"]["checkpoint_bytes"] = len(encode(state_payload))


def _m_l110_resealed_retrieval_undercharge(art):
    case = art["restart"]["cases"][0]
    case["retrieval_accounting"][
        "total_retrieval_node_equivalents"] = 4
    _reseal_restart_case(case)


def _m_l110_resealed_new_support(art):
    case = art["restart"]["cases"][0]
    support = list(case["post_query_new_support"])
    case["post_query_new_support"] = support[1:]
    _reseal_restart_case(case)


def _m_l110_resealed_record_support(art):
    case = art["restart"]["cases"][0]
    key = next(iter(case["post_query_record_supports"]))
    support = list(case["post_query_record_supports"][key])
    case["post_query_record_supports"][key] = support[1:]
    _reseal_restart_case(case)


def _m_l110_resealed_selection_weights(art):
    case = art["restart"]["cases"][0]
    case["selection_weights"]["scaled_sha256"] = "0" * 64
    _reseal_restart_case(case)


def _m_l110_resealed_priors(art):
    case = art["restart"]["cases"][0]
    case["inference_priors"]["p_new"] = Fraction(0)
    _reseal_restart_case(case)


def _m_l110_resealed_task_truth(art):
    case = art["restart"]["cases"][0]
    case["task_evidence"]["z"] = 0
    _reseal_restart_case(case)


def _m_l110_resealed_policy(art):
    case = art["restart"]["cases"][0]
    case["query_policy_state"]["policy"] = "random"
    for branch in case["provisional_branches"]:
        branch["policy"] = "random"
    _reseal_restart_case(case)


def _m_l110_final_state(art):
    art["restart"]["cases"][0]["cycle"]["final_state_identical"] = False


def _m_l110_fake_continuation(art):
    art["restart"]["cases"][0]["cycle"]["real_main_continuation"] = False


def _m_l110_wrong_continuation_query(art):
    art["restart"]["cases"][0]["cycle"]["continuation_queries"][0] = 31


def _m_l110_fabricated_bound(art):
    row = _first_split(art, "required_intervals")[
        "MAIN_minus_random_accuracy"]
    row["hi"] += Fraction(1, 100)


def _m_l110_unmatched_control(art):
    row = _first_split(art, "retrieval")["per_task_accounting"][
        "random_retrieval"][0]
    row["total_retrieval_node_equivalents"] = 4


_MUTATIONS: dict[str, tuple[tuple[str, Callable[[MutableMapping], None]], ...]] = {
    "L1.0": (("missing_task_conditions", _m_l10_missing_provenance),
             ("final_seed_sampled", _m_l10_final_seed),
             ("failed_prerequisite", _m_l10_prereq)),
    "L1.1": (("stable_risk_worse", _m_l11_risk),
             ("unmatched_history", _m_l11_history),
             ("dead_calibration", _m_l11_calibration)),
    "L1.2": (("invalid_proof", _m_l12_proof),
             ("differential_mismatch", _m_l12_mismatch),
             ("approximate_sketch", _m_l12_incomplete),
             ("dead_countermodel_calibration",
              _m_l12_countermodel_calibration)),
    "L1.3": (("smuggled_index_nodes", _m_l13_undercharge),
             ("over_byte_budget", _m_l13_bytes),
             ("dead_calibration", _m_l13_calibration),
             ("changed_validation_freeze", _m_l13_freeze),
             ("unmatched_random_control", _m_l13_unmatched_control)),
    "L1.4": (("wrong_q246_total", _m_l14_total),
             ("wrong_metric_label", _m_l14_label),
             ("dead_calibration", _m_l14_calibration)),
    "L1.5": (("missing_baseline", _m_l15_policy),
             ("unapplied_answer", _m_l15_unapplied),
             ("missing_resolution_effect", _m_l15_effect),
             ("flat_exact_task_IG", _m_l15_task_ig_flat),
             ("flat_exact_joint_IG", _m_l15_joint_ig_flat)),
    "L1.6": (("reversed_interval", _m_l16_interval),
             ("nonprefix_curve", _m_l16_prefix),
             ("dead_calibration", _m_l16_calibration)),
    "L1.7": (("unresolved_new", _m_l17_unresolved),
             ("never_reused", _m_l17_reuse),
             ("dead_calibration", _m_l17_calibration)),
    "L1.8": (("vacuous_shared_transfer", _m_l18_shared_transfer),
             ("missing_scope_digest", _m_l18_scope_digest),
             ("false_global_promotion", _m_l18_global),
             ("missing_representation_not_observed", _m_l18_missing),
             ("vacuous_convention", _m_l18_vacuous_convention),
             ("vacuous_grounded_event", _m_l18_vacuous_grounded),
             ("vacuous_unknown_meaning", _m_l18_vacuous_unknown),
             ("vacuous_restricted_case", _m_l18_vacuous_restricted)),
    "L1.9": (("inferior_condition", _m_l19_inferior),
             ("established_corruption", _m_l19_corruption),
             ("dead_calibration", _m_l19_calibration),
             ("unmatched_protocol_contract", _m_l19_unmatched_protocol)),
    "L1.10": (("nonstrict_random_bound", _m_l110_random_bound),
              ("exact_inferiority", _m_l110_exact_margin),
              ("nonstrict_query_bound", _m_l110_query_bound),
              ("nonstrict_safety_bound", _m_l110_safety_bound),
              ("nonzero_restart_interval", _m_l110_restart_interval),
              ("failed_restart", _m_l110_restart),
              ("misclassified_restart_storage", _m_l110_restart_storage),
              ("resealed_retrieval_undercharge",
               _m_l110_resealed_retrieval_undercharge),
              ("resealed_record_support", _m_l110_resealed_record_support),
              ("resealed_new_support", _m_l110_resealed_new_support),
              ("resealed_selection_weights",
               _m_l110_resealed_selection_weights),
              ("resealed_inference_priors", _m_l110_resealed_priors),
              ("resealed_task_truth_leak", _m_l110_resealed_task_truth),
              ("resealed_nonmain_policy", _m_l110_resealed_policy),
              ("different_exact_final_state", _m_l110_final_state),
              ("fake_main_continuation", _m_l110_fake_continuation),
              ("wrong_main_continuation_query",
               _m_l110_wrong_continuation_query),
              ("fabricated_bootstrap_bound", _m_l110_fabricated_bound),
              ("unmatched_retrieval_control", _m_l110_unmatched_control)),
}


def calibrate_gate(artifact: Mapping, gate: str) -> dict:
    """Plant gate-specific defects and require the same validator to reject."""
    if gate not in _VALIDATORS:
        raise ValueError(f"unknown L1 gate {gate!r}")
    baseline = validate_gate(artifact, gate)
    cases = {}
    errors = {}
    for name, mutation in _MUTATIONS[gate]:
        planted = copy.deepcopy(dict(artifact))
        try:
            mutation(planted)
            result = validate_gate(planted, gate)
            cases[name] = not result.ok
            errors[name] = list(result.errors)
        except Exception as exc:  # a malformed baseline is never green
            cases[name] = False
            errors[name] = [f"calibration could not be executed: {exc}"]
    return {
        "valid_input": baseline.ok,
        "rejected": cases,
        "rejection_errors": errors,
        "same_validator": True,
        "fires": baseline.ok and bool(cases) and all(cases.values()),
    }


def evaluate_gate(artifact: Mapping, gate: str) -> GateResult:
    return GateResult(gate, validate_gate(artifact, gate),
                      calibrate_gate(artifact, gate))


def evaluate_l1_artifact(artifact: Mapping) -> L1GateReport:
    """Evaluate L1.0--L1.10 without trusting runner-supplied gate booleans."""
    return L1GateReport(tuple(evaluate_gate(artifact, gate)
                              for gate in GATE_NAMES))
