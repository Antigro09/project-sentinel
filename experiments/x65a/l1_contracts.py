"""Pure, executable artifact contracts for X65A-L1.

Every validator is side-effect free.  Its paired calibration starts from a
valid object, plants a named defect, and requires the *same validator* to
reject it.  This keeps ``fires=True`` from becoming a hand-written assertion.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping


REQUIRED_ACTIVE_METRICS = frozenset({
    "task_accuracy",
    "equivalence_retrieval",
    "questions_at_matched_accuracy",
    "false_confident_answers",
})
EXPECTED_FULL_SUITE_PASSED = 640
EXPECTED_FULL_SUITE_SKIPPED = 1
FULL_SUITE_COMMAND = ("uv", "run", "pytest", "-q")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...]

    def canon(self) -> dict:
        return {"ok": self.ok, "errors": list(self.errors)}


def _result(errors) -> ValidationResult:
    errors = tuple(dict.fromkeys(str(e) for e in errors))
    return ValidationResult(not errors, errors)


def _is_exact_number(value) -> bool:
    return (isinstance(value, (int, Fraction))
            and not isinstance(value, bool))


def _budget(mapping, q: int):
    if q in mapping:
        return mapping[q]
    return mapping.get(str(q))


def validate_provenance(prov: Mapping) -> ValidationResult:
    """Validate every required section-0 provenance field exactly."""
    errors = []
    required = {
        "x65a_l_commit_full_hash", "current_HEAD", "branch",
        "tracked_tree_clean", "tracked_status", "untracked_tree_clean",
        "untracked_count", "untracked_status",
        "development_stream_seeds", "validation_stream_seeds",
        "streams_per_alphabet_stratum", "identities_per_stream",
        "task_conditions_per_stream", "query_budgets",
        "validation_frozen_margins", "experiment_runtime_ms",
        "full_suite_runtime_ms", "full_suite_evidence", "artifact_paths",
        "final_manifest_written", "final_stream_seed_sampled",
    }
    missing = sorted(required - set(prov))
    errors.extend(f"missing provenance field {k}" for k in missing)
    if missing:
        return _result(errors)

    for name in ("x65a_l_commit_full_hash", "current_HEAD"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(prov[name])):
            errors.append(f"{name} is not a full git hash")
    if not str(prov["branch"]):
        errors.append("branch is empty")

    tracked = tuple(prov["tracked_status"])
    untracked = tuple(prov["untracked_status"])
    if bool(prov["tracked_tree_clean"]) != (len(tracked) == 0):
        errors.append("tracked clean flag disagrees with tracked status")
    if int(prov["untracked_count"]) != len(untracked):
        errors.append("untracked count disagrees with untracked status")
    if bool(prov["untracked_tree_clean"]) != (len(untracked) == 0):
        errors.append("untracked clean flag disagrees with untracked status")

    dev = tuple(prov["development_stream_seeds"])
    val = tuple(prov["validation_stream_seeds"])
    if not dev or not val or set(dev) & set(val):
        errors.append("development/validation seeds must be nonempty and disjoint")
    streams = prov["streams_per_alphabet_stratum"]
    if (streams.get("development") != len(dev)
            or streams.get("validation") != len(val)):
        errors.append("stream counts disagree with declared seeds")
    if not isinstance(prov["identities_per_stream"], int) \
            or prov["identities_per_stream"] <= 0:
        errors.append("identities_per_stream must be positive")

    conditions = prov["task_conditions_per_stream"]
    for overlap in ("shared", "disjoint_op"):
        if overlap not in conditions:
            errors.append(f"task conditions missing stratum {overlap}")
            continue
        for split, seeds in (("development", dev), ("validation", val)):
            rows = conditions[overlap].get(split, {})
            for seed in seeds:
                row = rows.get(seed, rows.get(str(seed)))
                if not isinstance(row, Mapping) or not row:
                    errors.append(
                        f"task conditions missing {overlap}/{split}/{seed}")

    if tuple(prov["query_budgets"]) != (0, 1, 2, 3, 4):
        errors.append("query budgets are not the frozen q=0..4 curve")
    margins = prov["validation_frozen_margins"]
    for key in ("L3_retrieval_noninferiority", "L10_negative_transfer"):
        if key not in margins or not isinstance(margins[key], Fraction):
            errors.append(f"missing exact validation-frozen margin {key}")

    for key in ("experiment_runtime_ms", "full_suite_runtime_ms"):
        if not isinstance(prov[key], int) or prov[key] <= 0:
            errors.append(f"{key} must be a measured positive integer")
    suite = prov["full_suite_evidence"]
    if not isinstance(suite, Mapping):
        errors.append("full_suite_evidence must be a mapping")
    else:
        expected_keys = {
            "command", "exit_code", "passed", "skipped", "runtime_ms",
            "result_line",
        }
        if set(suite) != expected_keys:
            errors.append("full_suite_evidence fields are incomplete")
        if tuple(suite.get("command", ())) != FULL_SUITE_COMMAND:
            errors.append("full suite command is not the frozen repository suite")
        if suite.get("exit_code") != 0:
            errors.append("full suite did not exit successfully")
        if suite.get("passed") != EXPECTED_FULL_SUITE_PASSED:
            errors.append("full suite pass count is not the frozen L1 count")
        if suite.get("skipped") != EXPECTED_FULL_SUITE_SKIPPED:
            errors.append("full suite skip count is not the frozen L1 count")
        if suite.get("runtime_ms") != prov["full_suite_runtime_ms"]:
            errors.append("full suite runtime evidence disagrees with provenance")
        expected_line = (
            f"{EXPECTED_FULL_SUITE_PASSED} passed, "
            f"{EXPECTED_FULL_SUITE_SKIPPED} skipped in "
            f"{prov['full_suite_runtime_ms']} ms")
        if suite.get("result_line") != expected_line:
            errors.append("full suite result line disagrees with measured fields")
    paths = prov["artifact_paths"]
    for key in ("runner", "authoritative_json", "readme", "inference",
                "evaluation", "contracts", "retrieval", "safety",
                "negative_transfer", "statistics", "restart"):
        value = paths.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            errors.append(f"artifact path {key} is missing or not absolute")
    if prov["final_manifest_written"] is not False:
        errors.append("final manifest must not be written")
    if prov["final_stream_seed_sampled"] is not False:
        errors.append("final stream seed must not be sampled")
    return _result(errors)


def calibrate_provenance(prov: Mapping) -> dict:
    valid = validate_provenance(prov)
    cases = {}
    mutations = {
        "omitted_untracked_status": lambda x: x.pop("untracked_status", None),
        "zero_full_suite_runtime": lambda x: x.__setitem__(
            "full_suite_runtime_ms", 0),
        "failed_full_suite_exit": lambda x: x["full_suite_evidence"].__setitem__(
            "exit_code", 1),
        "wrong_full_suite_count": lambda x: x["full_suite_evidence"].__setitem__(
            "passed", EXPECTED_FULL_SUITE_PASSED - 1),
        "omitted_task_conditions": lambda x: x.pop(
            "task_conditions_per_stream", None),
        "short_base_hash": lambda x: x.__setitem__(
            "x65a_l_commit_full_hash", "5205543"),
    }
    for name, mutate in mutations.items():
        planted = copy.deepcopy(dict(prov))
        mutate(planted)
        cases[name] = not validate_provenance(planted).ok
    return {"valid_input": valid.ok, "rejected": cases,
            "fires": valid.ok and all(cases.values())}


def validate_q246(accounting: Mapping) -> ValidationResult:
    """Validate the exact legacy relation 295/120 = 59/24 and its curve."""
    errors = []
    if accounting.get("overlap") != "disjoint_op":
        errors.append("q=2.46 audit must identify the disjoint_op stratum")
    if accounting.get("tasks") != 120:
        errors.append("legacy q=2.46 denominator must be 120 probe rows")
    budgets = accounting.get("budgets", {})
    q0, q1, q3 = (_budget(budgets, q) for q in (0, 1, 3))
    if not all(isinstance(row, Mapping) for row in (q0, q1, q3)):
        return _result(errors + ["legacy budgets 0, 1, and 3 are required"])
    expected = (
        (q0.get("query_budget"), 0, "q0 budget label"),
        (q0.get("queries_actually_asked"), 0, "q0 questions asked"),
        (q0.get("task_accuracy"), Fraction(53, 84), "q0 accuracy"),
        (q1.get("query_budget"), 1, "q1 budget label"),
        (q1.get("task_accuracy"), Fraction(80, 84), "q1 accuracy"),
        (q3.get("queries_actually_asked"), 295, "q3 questions asked"),
        (q3.get("mean_over_all_tasks"), Fraction(59, 24), "q3 mean"),
        (q3.get("query_budget"), 3, "q3 budget label"),
    )
    for got, want, name in expected:
        if got != want:
            errors.append(f"{name} mismatch: expected {want}, got {got}")
    if q3.get("queries_actually_asked") != 295 \
            or q3.get("mean_over_all_tasks") != Fraction(295, 120):
        errors.append("q=2.46 exact total/denominator identity failed")
    if accounting.get("published_curve") != \
            "0.631 -> 0.952 at budget one":
        errors.append("published curve label is inconsistent")
    return _result(errors)


def calibrate_q246(accounting: Mapping) -> dict:
    valid = validate_q246(accounting)
    cases = {}
    for name in ("wrong_total", "wrong_denominator", "wrong_budget_label",
                 "old_zero_query_label"):
        planted = copy.deepcopy(dict(accounting))
        budgets = planted["budgets"]
        q3 = _budget(budgets, 3)
        if name == "wrong_total":
            q3["queries_actually_asked"] = 294
        elif name == "wrong_denominator":
            q3["mean_over_all_tasks"] = Fraction(295, 119)
        else:
            if name == "wrong_budget_label":
                q3["query_budget"] = 2
            else:
                q0 = _budget(budgets, 0)
                q0["queries_actually_asked"] = _budget(
                    budgets, 1)["queries_actually_asked"]
        cases[name] = not validate_q246(planted).ok
    return {
        "valid_input": valid.ok,
        "rejected": cases,
        "old_zero_query_label_rejected": cases["old_zero_query_label"],
        "fires": valid.ok and all(cases.values()),
    }


def validate_memoryless_answer_application(row) -> ValidationResult:
    """Require a one-to-one chain: asked -> applied history -> effect record."""
    errors = []
    if hasattr(row, "canon"):
        data = row.canon()
        state_history = tuple(row.state.history)
        effect_events = tuple(e.event for e in row.resolution_effects)
        if tuple(row.evaluation.asked) != state_history:
            errors.append("evaluation history differs from applied state history")
        if effect_events != state_history:
            errors.append("resolution effects do not match applied answers")
        if row.evaluation.validation_errors():
            errors.extend(row.evaluation.validation_errors())
    elif isinstance(row, Mapping):
        data = row
    else:
        return _result(["memoryless row must be PolicyRun or canonical mapping"])

    asked = data.get("queries_asked")
    budget = data.get("query_budget")
    effects = data.get("resolution_effects", ())
    types = data.get("query_types", {})
    if not isinstance(asked, int) or asked < 0:
        errors.append("queries_asked must be a nonnegative integer")
    else:
        if not isinstance(budget, int) or asked > budget:
            errors.append("queries asked exceed budget")
        if len(effects) != asked:
            errors.append("one resolution effect is required per asked query")
        if sum(int(v) for v in types.values()) != asked:
            errors.append("query-type counts disagree with asked queries")
    if data.get("answers_applied") is not True:
        errors.append("answers_applied is false")
    return _result(errors)


def calibrate_memoryless_answer_application(row) -> dict:
    valid = validate_memoryless_answer_application(row)
    base = row.canon() if hasattr(row, "canon") else copy.deepcopy(dict(row))
    cases = {}
    planted = copy.deepcopy(base)
    planted["queries_asked"] += 1
    cases["counted_without_answer"] = not \
        validate_memoryless_answer_application(planted).ok
    planted = copy.deepcopy(base)
    if planted["resolution_effects"]:
        planted["resolution_effects"].pop()
    else:
        planted["queries_asked"] = 1
        planted["query_budget"] = max(1, planted["query_budget"])
    cases["missing_resolution_effect"] = not \
        validate_memoryless_answer_application(planted).ok
    planted = copy.deepcopy(base)
    planted["answers_applied"] = False
    cases["unapplied_answer"] = not \
        validate_memoryless_answer_application(planted).ok
    return {"valid_input": valid.ok, "rejected": cases,
            "fires": valid.ok and all(cases.values())}


def validate_active_intervals(bundle: Mapping,
                              expected_clusters: int) -> ValidationResult:
    """Validate the four paired operational intervals and status semantics."""
    errors = []
    intervals = bundle.get("intervals", {})
    keys = frozenset(intervals)
    if keys != REQUIRED_ACTIVE_METRICS:
        errors.append("active interval keys are incomplete or unexpected")
    includes = []
    for key in sorted(REQUIRED_ACTIVE_METRICS & keys):
        row = intervals[key]
        for name in ("lo", "delta", "hi"):
            if not _is_exact_number(row.get(name)):
                errors.append(f"{key}.{name} is not exact")
        if all(_is_exact_number(row.get(name))
               for name in ("lo", "delta", "hi")):
            if not row["lo"] <= row["delta"] <= row["hi"]:
                errors.append(f"{key} interval does not contain its estimate")
            includes.append(row["lo"] <= 0 <= row["hi"])
        if row.get("clusters") != expected_clusters:
            errors.append(f"{key} cluster count mismatch")
        if row.get("unit") != "complete_stream_or_latent_identity":
            errors.append(f"{key} resampling unit is wrong")
        if not isinstance(row.get("resamples"), int) \
                or row["resamples"] <= 0:
            errors.append(f"{key} resample count is invalid")
    all_include = (len(includes) == len(REQUIRED_ACTIVE_METRICS)
                   and all(includes))
    if bundle.get("all_operational_intervals_include_zero") != all_include:
        errors.append("all-include-zero flag disagrees with intervals")
    expected_status = ("not_measured_in_X65A-L" if all_include else "measured")
    if bundle.get("component_status") != expected_status:
        errors.append("active component status disagrees with intervals")
    streams = bundle.get("streams", {})
    if not isinstance(streams, Mapping) or len(streams) != expected_clusters:
        errors.append("active stream clusters are missing or duplicated")
    return _result(errors)


def calibrate_active_intervals(bundle: Mapping,
                               expected_clusters: int) -> dict:
    valid = validate_active_intervals(bundle, expected_clusters)
    cases = {}
    planted = copy.deepcopy(dict(bundle))
    planted["intervals"].pop("task_accuracy", None)
    cases["missing_metric"] = not validate_active_intervals(
        planted, expected_clusters).ok
    planted = copy.deepcopy(dict(bundle))
    row = planted["intervals"]["task_accuracy"]
    # Make the plant red even when the valid interval is the all-zero null.
    row["lo"], row["delta"], row["hi"] = (
        Fraction(1), Fraction(0), Fraction(-1))
    cases["reversed_interval"] = not validate_active_intervals(
        planted, expected_clusters).ok
    planted = copy.deepcopy(dict(bundle))
    planted["intervals"]["task_accuracy"]["clusters"] = \
        expected_clusters + 1
    cases["wrong_cluster_count"] = not validate_active_intervals(
        planted, expected_clusters).ok
    planted = copy.deepcopy(dict(bundle))
    planted["component_status"] = (
        "measured" if bundle.get("component_status") ==
        "not_measured_in_X65A-L" else "not_measured_in_X65A-L")
    cases["wrong_component_status"] = not validate_active_intervals(
        planted, expected_clusters).ok
    return {"valid_input": valid.ok, "rejected": cases,
            "fires": valid.ok and all(cases.values())}
